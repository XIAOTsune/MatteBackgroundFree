import os
import numpy as np
import torch
import traceback
import gc

# Try importing moviepy
try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("警告: moviepy 未安装，视频处理功能将不可用")

from src.utils.logger import logger
from src.config import PRED_OUTPUT_DIR
import cv2  # Added cv2 import
from src.models import segment_image, model_manager
from src.image_processing import (
    estimate_soft_alpha_inside_mask,
    _boost_veil_color,
    hex_to_rgb,
    _resize_bg_keep_aspect,
    estimate_background_color,
    compute_alpha_unified
)

def process_single_video(
    input_video: str,
    model_name: str,
    input_size: tuple,
    bg_color: str = "#00FF00",
    progress=None,
    # === 新增 ===
    semi_enable=False, semi_strength=0.5, semi_mode='auto'
):
    """处理单个视频（无 Alpha 导出）：逐帧分割并合成到 RGB，输出 MP4(H.264)"""
    if not MOVIEPY_AVAILABLE:
        return None, "视频处理功能不可用：moviepy未安装"
    if input_video is None:
        return None, "请上传视频文件"

    try:
        # === 1) 模型准备 ===
        if model_manager.model is None or (model_manager.current_loaded_model_name != model_name):
            ok = model_manager.load_model(model_name, input_size)
            if not ok:
                raise RuntimeError(f"无法加载模型 {model_name}")

        # === 2) 打开视频 ===
        video = mp.VideoFileClip(input_video)
        total_frames = max(1, int(video.fps * video.duration))
        vw, vh = video.size

        logger.info(f"🎥 视频信息: {total_frames} 帧, {video.fps:.2f} FPS, {video.duration:.2f}s")
        logger.info(f"🧠 当前推理分辨率: {input_size[0]}x{input_size[1]}")

        # === 准备背景 ===
        # [UPDATE]: 仅保留纯色背景选项
        r, g, b = hex_to_rgb(bg_color)
        prepared_bg_arr = np.full((vh, vw, 3), (r, g, b), dtype=np.uint8)
        
        prepared_bg_arr = prepared_bg_arr.astype(np.uint8)
        processed_frames = 0

        # === 定义帧处理函数 ===
        def process_frame(get_frame, t):
            nonlocal processed_frames
            frame = get_frame(t)  # RGB
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)

            # 1) 分割
            try:
                mask = segment_image(frame, model_name=model_name, input_size=input_size)
            except Exception as e:
                logger.error(f"[video] segment_image 失败: {e}")
                mask = None

            if mask is None:
                out = frame
            else:
                m_u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
                
                # 2) 半透明处理 (Alpha 计算)
                if semi_enable:
                    # 使用 V4.0 Unified 算法 (与图片端对齐)
                    bg_color_est = estimate_background_color(frame, m_u8)
                    if bg_color_est[3]:
                        m_u8 = compute_alpha_unified(frame, mask, strength=float(semi_strength), bg_color=bg_color_est[:3])
                    else:
                        m_u8 = estimate_soft_alpha_inside_mask(frame, mask, strength=float(semi_strength), mode="auto")
                    
                    # 移除 MORPH_CLOSE (填补漏洞 Bug)，保留所有半透明细节
                    frame_processed = frame
                else:
                    frame_processed = frame

                # 4) 合成
                m_f = m_u8.astype(np.float32) / 255.0
                m_f = m_f[..., None] # (H,W,1)
                
                # out = fg * a + bg * (1-a)
                out = (frame_processed.astype(np.float32) * m_f + 
                       prepared_bg_arr.astype(np.float32) * (1.0 - m_f)).astype(np.uint8)

                if model_manager.device.type == "cuda" and (processed_frames % 500 == 0):
                    torch.cuda.empty_cache()

            processed_frames += 1
            if total_frames and (processed_frames % 5 == 0):
                try:
                    if progress is not None:
                        progress(processed_frames / total_frames, f"已处理 {processed_frames}/{total_frames} 帧")
                except Exception:
                    pass
            return out

        # === 3) 写入视频 ===
        processed_video = video.fl(lambda gf, t: process_frame(gf, t))
        base_name = f"result_{os.path.basename(input_video)}"
        base, _ = os.path.splitext(os.path.join(PRED_OUTPUT_DIR, base_name))
        os.makedirs(PRED_OUTPUT_DIR, exist_ok=True)
        out_path = base + ".mp4"

        processed_video.write_videofile(
            out_path, codec="libx264", audio=True, audio_codec='aac',
            fps=video.fps, verbose=False, logger=None, threads=4, preset='medium'
        )
        video.close(); processed_video.close()
        
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        return out_path, f"✅ 完成：{os.path.basename(out_path)}"

    except Exception as e:
        logger.error(f"视频处理失败: {e}")
        traceback.print_exc()
        return None, f"视频处理失败：{e}"
