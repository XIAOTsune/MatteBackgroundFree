import os
import cv2
import numpy as np
import torch
import gc
import traceback
from datetime import datetime
from PIL import Image

from src.utils.logger import logger
from src.config import PRED_OUTPUT_DIR
from src.models import segment_image, model_manager
from src.image_processing import (
    estimate_soft_alpha_inside_mask, 
    _boost_veil_color, 
    replace_background_with_mask, 
    create_transparent_result,
    _save_image_safe,
    estimate_background_color, # New
    compute_alpha_unified      # New
)

# Helpers for ROI
def _bbox_from_mask(mask_u8: np.ndarray):
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return int(x0), int(y0), int(x1), int(y1)

def _expand_box(x0, y0, x1, y1, pad: int, W: int, H: int):
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
    if x1 <= x0: x1 = min(W, x0 + 1)
    if y1 <= y0: y1 = min(H, y0 + 1)
    return x0, y0, x1, y1

def _filter_mask_by_strokes(mask_f: np.ndarray, strokes_f: np.ndarray, thresh: float = 0.5):
    """
    基于用户笔迹过滤掩码连通域。
    mask_f: 模型预测的 0-1 float 掩码
    strokes_f: 用户涂抹的 0-1 float 笔迹
    """
    # 1. 二值化
    bin_mask = (mask_f > thresh).astype(np.uint8)
    bin_strokes = (strokes_f > 0.05).astype(np.uint8)
    
    # 2. 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    
    # 3. 统计哪些 Label 被笔迹碰到了
    keep_labels = set()
    for i in range(1, num_labels): # 0 是背景
        # 提取当前对象的掩码作为 ROI
        obj_mask = (labels == i).astype(np.uint8)
        # 如果笔迹在该对象区域内有像素，则保留
        if np.any((obj_mask > 0) & (bin_strokes > 0)):
            keep_labels.add(i)
    
    if not keep_labels:
        # 如果没涂到任何识别出的物体，返回空掩码
        return np.zeros_like(mask_f)
        
    # 4. 构建最终掩码 (保留原有的软边缘概率值)
    final_mask = np.zeros_like(mask_f)
    for i in keep_labels:
        obj_mask_bool = (labels == i)
        final_mask[obj_mask_bool] = mask_f[obj_mask_bool]
        
    return final_mask

def apply_background_replacement(
    image,
    mask=None,           # Added to fix NameError
    model_name='General', # Restored
    input_size=(1024, 1024),
    semi_transparent: bool = False,
    semi_strength: float = 0.5,
    semi_mode: str = 'auto',
    remove_white_halo: bool = False,
    defringe_strength: float = 0.7,
    # === ROI 新增参数 ===
    roi_mask_fullres: np.ndarray | None = None,
    roi_crop_before: bool = True,
    roi_pad_px: int = 16,
):
    try:
        # 1) 参数容错
        if isinstance(model_name, (int, float)):
            model_name = "General"

        # 2) 图像规范化
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255)
            image_array = (image_array * 255.0 if image_array.max() <= 1.0 else image_array).astype(np.uint8)

        if image_array.ndim == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.ndim == 3 and image_array.shape[2] >= 4:
            image_array = image_array[:, :, :3]

        H, W = image_array.shape[:2]

        # 3) 获取原始 Mask (Float 0.0-1.0)
        if mask is None:
            # === ROI 逻辑 ===
            if roi_mask_fullres is not None and roi_crop_before:
                roi_bin = (roi_mask_fullres > 0).astype(np.uint8)
                bbox = _bbox_from_mask(roi_bin)
                if bbox is not None:
                    x0, y0, x1, y1 = _expand_box(*bbox, pad=int(roi_pad_px), W=W, H=H)
                    crop_img = image_array[y0:y1, x0:x1]
                    m_crop = segment_image(crop_img, model_name=model_name, input_size=input_size)
                    if m_crop is None: raise ValueError("无法生成分割mask（ROI裁剪）")
                    
                    if m_crop.dtype == np.uint8: m_crop = m_crop.astype(np.float32) / 255.0
                    
                    m_crop = cv2.resize(m_crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LANCZOS4)
                    m = np.zeros((H, W), np.float32)
                    m[y0:y1, x0:x1] = m_crop
                else:
                    m = segment_image(image_array, model_name=model_name, input_size=input_size)
            else:
                m = segment_image(image_array, model_name=model_name, input_size=input_size)
            
            if m is None: raise ValueError("Mask生成失败")
        else:
            m = np.asarray(mask)

        # 统一 Mask 格式 (Float)
        if m.dtype == np.uint8: m = m.astype(np.float32) / 255.0
        if m.ndim == 3: m = m[:, :, 0]
        
        # 缩放回原图 (Lanczos)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_CUBIC)

        m = np.clip(m, 0.0, 1.0)

        # === ROI 约束 (Smart ROI: 连通域过滤) ===
        if roi_mask_fullres is not None:
            # roi_mask_fullres 是用户涂抹的笔迹
            strokes = np.asarray(roi_mask_fullres)
            if strokes.dtype != np.uint8:
                strokes = (strokes * 255).astype(np.uint8)
            if strokes.ndim == 3: strokes = strokes[:, :, 0]
            if strokes.shape[:2] != (H, W):
                strokes = cv2.resize(strokes, (W, H), interpolation=cv2.INTER_NEAREST)
            
            # 使用算法过滤
            m = _filter_mask_by_strokes(m, strokes.astype(np.float32) / 255.0)

        # === 核心逻辑分支 ===
        
        if semi_transparent:
            # === 半透明模式 (V4.0 Unified) ===
            # 不再区分模式，统一使用色差估算算法
            strength_val = float(semi_strength)
            
            # 1. 估算背景色
            # 传入原始 m (float 0-1) 用于辅助判断背景区域
            bg_color_est = estimate_background_color(image_array, (m * 255).astype(np.uint8))
            
            # 2. 计算 Alpha
            if bg_color_est[3]: # 如果背景估算有效
                logger.info(f"🎨 估算背景色 (RGB): {bg_color_est[:3]} (基于 Mask 外部采样)")
                m_final_u8 = compute_alpha_unified(image_array, m, strength=strength_val, bg_color=bg_color_est[:3])
            else:
                # 估算失败（全屏物体等），回退到旧的 Gamma 方案
                logger.warning("⚠️ 背景估算失败（背景区域过小），回退到通用半透明算法")
                m_final_u8 = estimate_soft_alpha_inside_mask(image_array, m, strength=strength_val, mode="auto")

            # 3. 颜色修正 (可选，目前 V4 统一算法主要依赖 Alpha 准确性，暂时跳过复杂的 Veil Boost，因为色差发已经处理了大部分)
            image_array_processed = image_array
        
        else:
            # === 基础模式 (绝对信任模型) ===
            image_array_processed = image_array 
            
            # 仅在需要时应用引导滤波（通常 BiRefNet 不需要，但为了平滑边缘可保留，参数设为保守）
            # 如果您不需要对边缘做任何额外平滑，可以把 try/except 块与其后的 m_refined 逻辑全部注释掉，直接 use m.
            # 为了“边缘质量”，建议保留极轻微的 GuidedFilter 或直接用 m。
            # 这里我们选择：直接使用原始 mask 的 float 值，仅做极轻微的高斯模糊抗锯齿（如果不希望有任何模糊，只需 clip）
            
            # 方案：信任模型输出，不做额外的形态学操作。
            # 仅做 uint8 转换
            m_refined = m
            m_final_u8 = (np.clip(m_refined, 0.0, 1.0) * 255).astype(np.uint8)

        # 预览与合成
        mask_preview = Image.fromarray(m_final_u8).convert('RGB')

        # [UPDATE]: 移除自定义背景替换功能，统一输出透明 PNG
        result = create_transparent_result(
            image_array=image_array_processed,
            mask=m_final_u8,
            remove_white_halo=remove_white_halo,
            defringe_strength=defringe_strength
        )

        return result, mask_preview

    except Exception as e:
        error_msg = f"处理失败: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None, error_msg


def process_single_image(
    image,
    semi_enable=False,
    semi_strength=0.5,
    semi_mode='auto',
    defringe=False,
    defringe_strength=0.7,
    _model_name=None,
    _resolution=None,
    # === 新增（仅在内部转发到底层） ===
    _roi_mask_fullres=None,
    _roi_crop_before=True,
    _roi_pad_px=16,
):
    """执行单张图片的前后处理和融合。返回：result_image, mask_preview"""
    if image is None:
        return None, "请上传图片"

    try:
        logger.info("开始处理图片")

        # 1) 解析模型名 & 分辨率（从全局兜底）
        model_name = (
            _model_name
            or model_manager.current_loaded_model_name
            or "General"
        )
        try:
            resolution = int(
                _resolution
                or (
                    model_manager.current_loaded_resolution[0]
                    if isinstance(model_manager.current_loaded_resolution, tuple)
                    else None
                )
                or 1024
            )
        except Exception:
            resolution = 1024
        if resolution <= 0:
            resolution = 1024
        input_size = (resolution, resolution)

        # 2) 调统一后端（用关键字入参更稳健）
        result, mask_preview = apply_background_replacement(
            image=image,
            model_name=model_name,
            input_size=input_size,
            semi_transparent=semi_enable,
            semi_strength=semi_strength,
            semi_mode=semi_mode,
            remove_white_halo=defringe,
            defringe_strength=defringe_strength,
            # 传递 ROI
            roi_mask_fullres=_roi_mask_fullres,
            roi_crop_before=_roi_crop_before,
            roi_pad_px=_roi_pad_px,
        )
        
        if result is None:
            # apply_background_replacement 出错时第二返回值是错误信息字符串
            return None, mask_preview if isinstance(mask_preview, str) else "处理失败"

        logger.info("图片处理完成")

        # 3) 自动保存（更稳健的兜底，不影响其他功能）
        try:
            out_dir = PRED_OUTPUT_DIR
            os.makedirs(out_dir, exist_ok=True)

            # 尝试获取原文件名；否则用时间戳
            base_name = None
            if hasattr(image, "filename") and getattr(image, "filename"):
                base_name = os.path.splitext(os.path.basename(image.filename))[0]
            elif hasattr(image, "name") and getattr(image, "name"):
                base_name = os.path.splitext(os.path.basename(image.name))[0]
            if not base_name:
                base_name = datetime.now().strftime("%Y%m%d_%H%M%S")

            save_path = os.path.join(out_dir, f"single_{base_name}.png")

            _save_image_safe(result, save_path)

            logger.info(f"🖼️ 单张结果已自动保存：{save_path}")
        except Exception as se:
            logger.warning(f"单张结果保存失败（不影响前端展示）：{se}")

        return result, mask_preview

    except Exception as e:
        error_msg = f"处理失败: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None, error_msg

    finally:
        # === 强制显存/内存回收 ===
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def process_batch_files(
    files,
    file_type='image',
    progress=None,
    *,
    model_name='General',
    input_size=(1024, 1024),
    semi_enable=False,
    semi_strength=0.5,
    semi_mode='auto',
    defringe=False,            # 仅图片用
    defringe_strength=0.7,     # 仅图片用
    bg_color="#00FF00"         # 仅视频用
):
    """
    批量处理（图片/视频）。
    Note: Video batch processing logic delegates to process_single_video in video_processor
    """
    import zipfile
    
    results = []
    
    total = len(files)

    # ================= 批量图片 =================
    if file_type == 'image':
        for idx, f in enumerate(files, 1):
            if progress:
                progress((idx - 1) / max(1, total), desc=f"图片 {idx}/{total}")

            # 读取
            try:
                if hasattr(f, "read"):
                    img = Image.open(f.name).convert("RGB")
                    img = np.array(img)
                elif isinstance(f, str):
                    img = Image.open(f).convert("RGB")
                    img = np.array(img)
                else:
                    img = np.asarray(f)
            except Exception as e:
                logger.error(f"读取图片失败 {f}: {e}")
                continue

            # 处理
            result_img, _ = apply_background_replacement(
                image=img,
                model_name=model_name,
                input_size=input_size,
                semi_transparent=semi_enable,
                semi_strength=semi_strength,
                semi_mode=semi_mode,
                remove_white_halo=defringe,
                defringe_strength=defringe_strength,
                # is_transparent_mode applies only to internal logic inside apply_background_replacement
            )

            # 保存
            if result_img is not None:
                fname = getattr(f, 'name', f if isinstance(f, str) else f"img_{idx}")
                base = os.path.basename(fname)
                name, _ = os.path.splitext(base)
                out_path = os.path.join(PRED_OUTPUT_DIR, f"{name}_result.png")
                try:
                    # 安全保存
                    if hasattr(result_img, "save"):
                        result_img.save(out_path)
                    else:
                        Image.fromarray(result_img).save(out_path)
                    results.append(out_path)
                except Exception:
                    pass
            
            gc.collect()

    elif file_type == 'video':
        # Need to import video processor here to avoid circular dep if video processor imports this file
        # But actually video processor might depend on apply_background_replacement...
        # So we should put video processing in video_processor.py and import it here.
        from .video_processor import process_single_video
        
        for idx, f in enumerate(files, 1):
            if progress:
                progress((idx - 1) / max(1, total), desc=f"视频 {idx}/{total}")
            
            v_path = f.name if hasattr(f, 'name') else str(f)
            
            # 调用单视频处理逻辑
            out_path, msg = process_single_video(
                input_video=v_path,
                model_name=model_name,
                input_size=input_size,
                bg_color=bg_color,
                progress=None, 
                semi_enable=semi_enable,
                semi_strength=semi_strength,
                semi_mode=semi_mode
            )
            
            if out_path and os.path.exists(out_path):
                results.append(out_path)
            
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    else:
        return None, "⚠️ 未识别的文件类型"

    # ================= 打包下载 =================
    if results:
        zip_name = f"batch_{file_type}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_output_path = os.path.join(PRED_OUTPUT_DIR, zip_name)
        with zipfile.ZipFile(zip_output_path, "w") as zipf:
            for f in results:
                zipf.write(f, os.path.basename(f))
        return zip_output_path, f"✅ 批量处理完成，共 {len(results)} 个结果，已打包"
    else:
        return None, "⚠️ 未产生任何有效结果"
