import os
import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
from pathlib import Path
import logging
import traceback
import zipfile
import shutil
import requests
from torchvision.transforms import functional as F
import gc
from typing import Dict, Any, Tuple

from datetime import datetime
try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("警告: moviepy 未安装，视频处理功能将不可用")

try:
    from transformers import AutoModelForImageSegmentation
    import torchvision.transforms as transforms
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers未安装，模型加载功能将不可用")

from typing import Tuple, List, Optional, Union
##带alpha通道视频导出参数预设###


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局变量
model = None
device = None
transform = None
current_loaded_model_name = None
current_loaded_resolution = None

# ===== 新增：旧版 BiRefNet 模型映射 =====
usage_to_weights_file = {
    'General': 'BiRefNet',
    'General-Lite': 'BiRefNet_lite',
    'General-Lite-2K': 'BiRefNet_lite-2K',
    'Matting': 'BiRefNet-matting',
    'Portrait': 'BiRefNet-portrait',
    'DIS': 'BiRefNet-DIS5K',
    'HRSOD': 'BiRefNet-HRSOD',
    'COD': 'BiRefNet-COD',
    'DIS-TR_TEs': 'BiRefNet-DIS5K-TR_TEs',
    'General-legacy': 'BiRefNet-legacy'
}
# ===== 模型说明（显示给用户看的中文备注） =====
model_descriptions = {
    "General": "通用版（BiRefNet） - 适合大多数自然图像",
    "General-Lite": "轻量版（BiRefNet_lite） - 推理速度快，精度略低",
    "General-Lite-2K": "高分辨率版（BiRefNet_lite-2K） - 适合2K图像",
    "Matting": "抠图版（BiRefNet-matting） - 擅长发丝、透明边缘",
    "Portrait": "人像优化版（BiRefNet-portrait） - 擅长人像抠图",
    "DIS": "细节增强版（BiRefNet-DIS5K） - 细节表现更好",
    "HRSOD": "高分辨率分割版（BiRefNet-HRSOD） - 复杂背景效果更佳",
    "COD": "伪装检测版（BiRefNet-COD） - 擅长隐藏/伪装目标",
    "DIS-TR_TEs": "DIS5K训练增强版（BiRefNet-DIS5K-TR_TEs）",
    "General-legacy": "旧版通用模型（兼容性好，权重较旧）"
}

# === 输出目录 ===
PRED_OUTPUT_DIR = os.path.join(os.getcwd(), "preds-BiRefNet")
os.makedirs(PRED_OUTPUT_DIR, exist_ok=True)

#照片读取逻辑
def load_image_safe(path):
    from PIL import Image, ImageOps
    image = Image.open(path)
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    return image

#####################绘制工具函数#####################
def _make_editor_thumbnail(img_pil: Image.Image, long_side: int = 640) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    把原图缩放成缩略图（不加黑边），并构造 ImageEditor 的 EditorValue 初始值。
    返回：(editor_value, meta)，meta 里包含原图与缩略图尺寸。
    """
    w, h = img_pil.size
    scale = min(long_side / max(w, h), 1.0)
    tw, th = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    thumb = img_pil.resize((tw, th), Image.BILINEAR)

    editor_value = {"background": thumb, "layers": [], "composite": None}
    meta = {"ori_w": w, "ori_h": h, "thumb_w": tw, "thumb_h": th}
    return editor_value, meta

# ===== 修改开始：修复 ROI 报错 (兼容 Numpy/PIL) =====
def _editor_layers_to_mask_fullres(editor_value: Dict[str, Any], meta: Dict[str, int]) -> np.ndarray | None:
    """
    从 ImageEditor 的编辑值提取 ROI（二值）并映射回原图尺寸。
    """
    if not editor_value or not meta:
        return None

    tw, th, W, H = meta["thumb_w"], meta["thumb_h"], meta["ori_w"], meta["ori_h"]
    mask_thumb = np.zeros((th, tw), np.uint8)

    # 1) 直接从图层 alpha 聚合
    layers = editor_value.get("layers") or []
    for layer in layers:
        if layer is None:
            continue
        arr = np.array(layer)
        if arr.ndim == 3 and arr.shape[2] == 4:     # RGBA
            alpha = arr[..., 3]
            mask_thumb = np.maximum(mask_thumb, alpha.astype(np.uint8))
        elif arr.ndim == 3 and arr.shape[2] == 3:   # RGB 兜底
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            mask_thumb = np.maximum(mask_thumb, (gray > 0).astype(np.uint8) * 255)

    # 2) 图层为空兜底：composite 与 background 差异
    # 🔧 修复点：增加类型判断，防止对 numpy 数组调用 .convert() 报错
    def _safe_get_rgba(item):
        if item is None: return None
        if isinstance(item, np.ndarray):
            # 如果是 numpy，确保是 RGBA
            if item.ndim == 3 and item.shape[2] == 3:
                return cv2.cvtColor(item, cv2.COLOR_RGB2RGBA)
            return item
        if hasattr(item, "convert"): # PIL Image
            return np.array(item.convert("RGBA"))
        return np.array(item)

    if mask_thumb.max() == 0 and editor_value.get("composite") is not None and editor_value.get("background") is not None:
        bg = _safe_get_rgba(editor_value.get("background"))
        comp = _safe_get_rgba(editor_value.get("composite"))
        
        if bg is not None and comp is not None and bg.shape == comp.shape:
            # 简单阈值差异
            diff = np.abs(comp[..., :3].astype(np.int16) - bg[..., :3].astype(np.int16)).sum(axis=2)
            mask_thumb = (diff > 5).astype(np.uint8) * 255

    if mask_thumb.max() == 0:
        return None

    # 3) 映射回原图 (线性插值 + 阈值)
    mask_full = cv2.resize(mask_thumb, (W, H), interpolation=cv2.INTER_LINEAR)
    return (mask_full > 127).astype(np.uint8) * 255
# ===== 修改结束 =====

def _bbox_from_mask(mask_u8: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return int(x0), int(y0), int(x1), int(y1)

def _expand_box(x0, y0, x1, y1, pad: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
    if x1 <= x0: x1 = min(W, x0 + 1)
    if y1 <= y0: y1 = min(H, y0 + 1)
    return x0, y0, x1, y1
#####################绘制工具函数#####################

# === 安全保存工具函数 ===
def _force_png_path(path: str) -> str:
    """把任意路径的扩展名改成 .png"""
    import os
    root, _ = os.path.splitext(path)
    return root + ".png"

def _save_image_safe(img, save_path: str):
    """
    安全保存图像：
    - RGBA/LA 或包含 transparency → 强制 PNG
    - JPEG 目标但图像非 RGB → 转 RGB 再存
    """
    from PIL import Image
    import os

    # 有透明信息 → 强制改为 PNG
    if getattr(img, "mode", "") in ("RGBA", "LA") or ("transparency" in getattr(img, "info", {})):
        save_path = _force_png_path(save_path)
        img.save(save_path, "PNG")
        return save_path

    # 目标是 jpg/jpeg，但图像不是 RGB → 转 RGB
    ext = os.path.splitext(save_path)[1].lower()
    if ext in (".jpg", ".jpeg") and img.mode != "RGB":
        img = img.convert("RGB")

    # 正常保存；若仍因格式报错，回退为 PNG
    try:
        img.save(save_path)
        return save_path
    except Exception:
        save_path = _force_png_path(save_path)
        img.save(save_path, "PNG")
        return save_path

def safe_progress(p, fraction: float, desc: str):
    """
    安全更新进度条：
    - 仅判断 None，不触发 gr.Progress.__len__
    - 调用失败直接吞掉，避免中断帧处理
    """
    try:
        if p is not None:
            p(fraction, desc)
    except Exception as e:
        logger.debug(f"[DBG] progress 跳过: {e}")

def scan_local_weights():
    """扫描 weights 文件夹下的所有 .pth 文件"""
    weights_dir = "weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    files = [f for f in os.listdir(weights_dir) if f.endswith(".pth")]
    return sorted(files)

# ===== 修改开始：load_model 支持动态模型与分辨率 =====
def load_model(model_name='General', input_size=(1024, 1024)):
    """
    加载 BiRefNet 模型（支持联网与离线双模式）
    - 网络可用：自动从 HuggingFace 拉取
    - 网络不可用：自动从 models_local 或缓存加载
    - 缓存路径统一到 ./models_local
    """
    global model, device, transform, current_loaded_model_name, current_loaded_resolution

    # --- 参数安全检查 ---
    if isinstance(model_name, (int, float)):
        raise TypeError(f"model_name 必须为字符串，收到: {model_name}({type(model_name)})")
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    elif not (isinstance(input_size, tuple) and len(input_size) == 2):
        raise ValueError(f"input_size 必须为 (H, W) tuple，收到: {input_size}")

    # --- 缓存命中检测 ---
    if model is not None and current_loaded_model_name == model_name:
        current_loaded_resolution = input_size
        print(f"✅ 已加载模型：{model_name}（无需重复加载），当前分辨率参数 {input_size}")
        return True

    # 更新加载记录
    current_loaded_model_name = model_name
    current_loaded_resolution = input_size

    from transformers import AutoModelForImageSegmentation
    from torchvision import transforms
    import os, requests, traceback

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 统一缓存目录 ./models_local ===
    project_root = os.getcwd()
    hf_cache_dir = (
        os.environ.get("HF_HOME")
        or os.environ.get("TRANSFORMERS_CACHE")
        or os.path.join(project_root, "models_local")
    )
    hf_cache_dir = os.path.abspath(hf_cache_dir)
    os.makedirs(hf_cache_dir, exist_ok=True)
    print(f"📦 模型缓存目录统一为: {os.path.relpath(hf_cache_dir)}")

    # === 确定模型名称与仓库 ===
    if model_name in usage_to_weights_file:
        repo_name = usage_to_weights_file[model_name]
    else:
        repo_name = model_name
    hf_repo = f"zhengpeng7/{repo_name}"
    local_path = os.path.join("models_local", repo_name)
    os.makedirs(local_path, exist_ok=True)

    # === 检测网络访问能力 ===
    def check_hf_access():
        try:
            r = requests.get("https://huggingface.co", timeout=3)
            return r.status_code == 200
        except Exception:
            return False



    try:
        if check_hf_access():
            print(f"🌐 正在从 HuggingFace 加载模型：{hf_repo}")
            model = AutoModelForImageSegmentation.from_pretrained(
                hf_repo,
                trust_remote_code=True,
                cache_dir=hf_cache_dir
            )
            if torch.cuda.is_available():
                model.to(device)
                # 尝试转换为半精度 (fp16) 以提速和省显存
                try:
                    model.half()
                    print("⚡ 已启用 FP16 半精度推理")
                except Exception:
                    model.float()
                    print("⚠️ FP16 转换失败，回退到 FP32")
            else:
                model.float()            
            model.to(device)
            model.eval()
            print(f"✅ 模型加载完成：{repo_name}，输入尺寸 {input_size}")

            # 自动保存离线副本（确保 config.json 等存在）
            try:
                model.save_pretrained(local_path)
                print(f"📦 已同步模型到本地离线目录: {local_path}")
            except Exception as e:
                print(f"⚠️ 模型离线保存失败: {e}")

        else:
            print("⚠️ 网络访问 HuggingFace 失败，尝试离线加载模型...")

            # --- 优化后的离线加载逻辑 ---
            offline_paths = []

            # ✅ 优先查找标准 Hugging Face 缓存结构
            hf_style_dir = os.path.join(hf_cache_dir, f"models--zhengpeng7--{repo_name}")
            if os.path.exists(hf_style_dir):
                snapshots_dir = os.path.join(hf_style_dir, "snapshots")
                if os.path.exists(snapshots_dir):
                    subdirs = [os.path.join(snapshots_dir, d) for d in os.listdir(snapshots_dir)]
                    subdirs = sorted(subdirs, key=os.path.getmtime, reverse=True)
                    offline_paths.extend(subdirs)
                offline_paths.append(hf_style_dir)

            # ✅ 兼容旧结构（例如 models_local/BiRefNet_lite-2K）
            legacy_path = os.path.join(hf_cache_dir, repo_name)
            if os.path.exists(legacy_path):
                offline_paths.append(legacy_path)

            local_path = os.path.join("models_local", repo_name)
            if os.path.exists(local_path):
                offline_paths.append(local_path)

            loaded = False
            for path in offline_paths:
                cfg = os.path.join(path, "config.json")
                if os.path.exists(cfg):
                    print(f"📂 尝试从离线路径加载模型：{path}")
                    model = AutoModelForImageSegmentation.from_pretrained(
                        path, trust_remote_code=True
                    )
                    if torch.cuda.is_available():
                        model.to(device)
                        # 尝试转换为半精度 (fp16) 以提速和省显存
                        try:
                            model.half()
                            print("⚡ 已启用 FP16 半精度推理")
                        except Exception:
                            model.float()
                            print("⚠️ FP16 转换失败，回退到 FP32")
                    else:
                        model.float()                    
                    model.to(device)
                    model.eval()
                    print(f"✅ 成功离线加载模型：{path}")
                    loaded = True
                    break

            if not loaded:
                raise FileNotFoundError(
                    f"❌ 未找到任何离线模型，请联网一次后再试。\n已检查路径: {offline_paths}"
                )

        # === 标准预处理 ===
        def resize_keep_ratio(img, target_size):
            w, h = img.size
            scale = target_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            return img.resize((new_w, new_h), Image.BILINEAR)

        transform = transforms.Compose([
            transforms.Lambda(lambda img: resize_keep_ratio(img, input_size[0])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        return True

    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        traceback.print_exc()
        return False

def preprocess_image(image):
    """预处理图像"""
    try:
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"图像文件不存在: {image}")
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("不支持的图像格式")
        
        # 转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        logger.error(f"图像预处理失败: {e}")
        return None

# ===== 修改开始：segment_image 优化插值与精度 =====
def segment_image(image, model_name='General', input_size=(1024, 1024)):
    """使用指定模型和分辨率进行图像分割"""
    global model, device, transform

    # ---- 容错：防止 model_name 被误传为 int ----
    if isinstance(model_name, (int, float)):
        sz = int(model_name)
        input_size = (sz, sz)
        model_name = (
            current_model_name
            if 'current_model_name' in globals() and isinstance(current_model_name, str)
            else 'General'
        )
        logger.warning(f"⚠️ 自动纠正参数错位：将 model_name={sz} 修正为 input_size={input_size}, 模型={model_name}")

    # 参数兜底
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    if "lite-2K" in str(model_name) and input_size[0] < 1024:
        logger.warning(f"⚠️ {model_name} 模型在低分辨率下可能结果异常，已自动提升至 1024")
        input_size = (1024, 1024)

    # 检查模型是否已加载
    if (model is None) or (current_loaded_model_name != model_name):
        if not load_model(model_name, input_size):
            return None

    try:
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None

        original_size = processed_image.size  # (W, H)

        resized_image = processed_image.resize(input_size, Image.BILINEAR)
        
        transform_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        
        # 处理输入 Tensor
        input_tensor = transform_pipeline(resized_image).unsqueeze(0).to(device)
        
        # 如果模型是 FP16，输入也要转 FP16
        if torch.cuda.is_available():
            input_tensor = input_tensor.half()

        with torch.no_grad():
            outputs = model(input_tensor)
            pred = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            
            # === 修复核心 BUG 在这里 ===
            # 1. sigmoid: 仍在 GPU 上
            # 2. squeeze: 降维
            # 3. cpu: 转到 CPU
            # 4. float(): ★★★ 强制转回 float32，防止 OpenCV 报错 ★★★
            # 5. numpy(): 转数组
            pred = torch.sigmoid(pred).squeeze().cpu().float().numpy()
            
            # 现在 pred 是 float32，OpenCV 可以正常缩放了
            pred = cv2.resize(pred, original_size, interpolation=cv2.INTER_CUBIC)
            mask = pred  # float32 (0.0 ~ 1.0)

        return mask

    except Exception as e:
        logger.error(f"图像分割失败: {e}")
        logger.error(traceback.format_exc())
        return None
# ===== 修改结束 =====


# ==== 新增：利用 inpaint 估计背景，并在掩码 ROI 内求连续 α ====
def _estimate_background_inpaint(rgb_img: np.ndarray, bin_mask: np.ndarray, radius: int) -> np.ndarray:
    """
    🔧 仅对需要的 ROI 做 inpaint，并在 ROI 内降采样处理，显著提速。
    返回与原图同尺寸的 RGB 背景估计。
    """
    if rgb_img.dtype != np.uint8:
        img8 = (np.clip(rgb_img, 0, 1) * 255).astype(np.uint8)
    else:
        img8 = rgb_img
    H, W = img8.shape[:2]

    # 掩码面积：前景占比过大 → 不做 inpaint，直接模糊回填（快）
    mask_area = float((bin_mask > 0).mean())
    if mask_area > 0.95:
        return cv2.GaussianBlur(img8, (11, 11), sigmaX=4)

    # 只取边带 ROI：避免整图 inpaint
    band_px = max(2, min(6, int(2 + 0.5 * radius)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    dil = cv2.dilate((bin_mask > 0).astype(np.uint8), ker, iterations=1)
    ero = cv2.erode((bin_mask > 0).astype(np.uint8), ker, iterations=1)
    band = cv2.subtract(dil, ero)  # 仅边带

    ys, xs = np.where(band > 0)
    if ys.size == 0:
        # 没有有效边带 → 直接轻模糊
        return cv2.GaussianBlur(img8, (9, 9), sigmaX=3)

    y0, y1 = max(0, ys.min() - 8), min(H, ys.max() + 9)
    x0, x1 = max(0, xs.min() - 8), min(W, xs.max() + 9)

    roi_img = img8[y0:y1, x0:x1]
    roi_msk = band[y0:y1, x0:x1].astype(np.uint8) * 255  # inpaint 需要 0/255

    # 降采样比例：ROI 边长 > 800 时做 0.5 缩放（可按需调大/调小）
    max_side = max(roi_img.shape[:2])
    scale = 0.5 if max_side > 800 else 1.0
    if scale < 1.0:
        new_w = max(1, int((x1 - x0) * scale))
        new_h = max(1, int((y1 - y0) * scale))
        roi_small = cv2.resize(roi_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        m_small = cv2.resize(roi_msk, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        roi_small = roi_img
        m_small = roi_msk

    bgr = cv2.cvtColor(roi_small, cv2.COLOR_RGB2BGR)
    r = max(3, min(12, int(radius)))  # 限制半径，避免超大半径拖慢
    try:
        bgr_bg = cv2.inpaint(bgr, m_small, r, cv2.INPAINT_TELEA)
    except Exception:
        bgr_bg = cv2.inpaint(bgr, m_small, 3, cv2.INPAINT_TELEA)

    # 回放到 ROI 尺寸
    if scale < 1.0:
        bgr_bg = cv2.resize(bgr_bg, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)

    out = img8.copy()
    out[y0:y1, x0:x1] = cv2.cvtColor(bgr_bg, cv2.COLOR_BGR2RGB)
    return out
####发丝保护权重
def _hair_protect_weight(rgb_u8: np.ndarray, m_u8: np.ndarray, band_u8: np.ndarray) -> np.ndarray:
    """
    计算发丝保护权重 w_hair ∈ [0,1]
    修复问题：原版偏向保护暗色物体，导致浅色皮肤/白衣边缘被侵蚀。
    新版策略：仅依赖 梯度(细节) + 距离(薄度)，去除亮度偏见。
    """
    import numpy as np, cv2
    H, W = m_u8.shape[:2]
    
    # 1. 梯度强度 (Sobel) - 检测复杂纹理(如发丝) vs 平滑区域(如皮肤)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    # 增强对比度，让弱纹理归0，强纹理归1
    grad = np.clip(grad * 3.0, 0, 1)

    # 2. “薄度” (Thinness) - 保护细小的结构，不管它是头发还是手指
    outside = (m_u8 == 0).astype(np.uint8)
    dist_in = cv2.distanceTransform(1 - outside, cv2.DIST_L2, 3)
    # 距离边缘 < 3px 的地方认为是很薄的区域
    thin = 1.0 - np.clip(dist_in / 3.0, 0.0, 1.0)

    # 3. 合成权重 (只在 band 区域生效)
    # 提高 thin 的权重，确保所有边缘都不会被过度“啃食”
    # 降低 grad 的权重，防止光滑边缘(皮肤)完全没保护
    w = (0.4 * grad + 0.6 * thin)
    
    w = np.clip(w, 0.0, 1.0)
    w *= (band_u8.astype(np.float32) > 0).astype(np.float32)

    return w.astype(np.float32)
####发丝保护权重




# ===== 优化后的 Alpha 估算 (V3.0 滤镜模式 - 信任模型) =====
def estimate_soft_alpha_inside_mask(
    image_or_array,
    base_mask: np.ndarray | float,
    *,
    strength: float = 0.5,        # 0~1：控制 Gamma 强度和 Luma 权重
    mode: str = "auto"            # "auto" / "暗部优先" / "透色优先"
) -> np.ndarray:
    """
    [V3.0 调优版]
    不再试图重新计算 Alpha，而是基于 BiRefNet 的原始输出进行“曲线重映射”。
    
    策略：
    1. Source of Truth: 信任模型的原始输出（Probability Map）。
    2. Curve (Gamma): 使用 Gamma 曲线压低中间调，使半透明区域更通透。
    3. Luma Masking: (透色优先模式) 依据亮度加权，亮部保持，暗部更透。
    """
    # ---- 1. 输入规整 (全部转为 float32 0.0~1.0) ----
    # 图像
    if isinstance(image_or_array, Image.Image):
        I = np.array(image_or_array.convert("RGB"))
    else:
        I = image_or_array
        if I.ndim == 3 and I.shape[2] == 4:
            I = I[:, :, :3]
    
    I_f = I.astype(np.float32) / 255.0

    # 掩码 (兼容 uint8 和 float)
    if base_mask.dtype == np.uint8:
        alpha = base_mask.astype(np.float32) / 255.0
    else:
        alpha = base_mask.astype(np.float32)
        # 确保范围在 0-1
        if alpha.max() > 1.1: 
            alpha /= 255.0
    
    alpha = np.clip(alpha, 0.0, 1.0)

    # ---- 2. 核心逻辑：曲线重映射 (Gamma Correction) ----
    # Strength 越大，Gamma 值越大，中间调(0.5)会被压得越低，看起来越透
    # 范围设定：Strength 0.0 -> Gamma 1.0 (原样); Strength 1.0 -> Gamma 3.5 (极透)
    gamma_base = 1.0 + (strength * 2.5)
    
    # 针对不同模式微调 Gamma 策略
    if mode in ("暗部优先", "dark"):
        # 暗部模式通常是为了保留阴影，Gamma 不宜过大，否则阴影没了
        final_gamma = gamma_base * 0.8 
    else:
        final_gamma = gamma_base

    # 应用 Gamma
    alpha_processed = np.power(alpha, final_gamma)

    # ---- 3. 核心逻辑：亮度加权 (Luma Masking) ----
    # 仅在 "透色优先" (针对婚纱、冰块、玻璃) 启用
    if mode in ("透色优先", "bleed", "light"):
        # 计算亮度 (Luma)
        luma = 0.299 * I_f[:,:,0] + 0.587 * I_f[:,:,1] + 0.114 * I_f[:,:,2]
        
        # 逻辑：亮度越高(1.0)，Alpha 保持原样；亮度越低(0.0)，Alpha 变得更小(更透)
        # weight = luma * strength + (1 - strength)
        # 当 strength=0 时，weight=1 (无影响)
        # 当 strength=1 时，weight=luma (完全由亮度决定透明度)
        
        # 稍微提升一点基准，防止黑色物体彻底消失
        luma_weight = luma * (0.8 * strength) + (1.0 - (0.8 * strength))
        
        alpha_processed = alpha_processed * luma_weight

    # ---- 4. 输出转换 ----
    return (np.clip(alpha_processed, 0.0, 1.0) * 255).astype(np.uint8)
# ===== 修改结束 =====

# ===== 新增：白纱/烟雾颜色提亮 (去灰) =====
def _boost_veil_color(image_rgb: np.ndarray, alpha_u8: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    针对薄纱/烟雾：强制提亮半透明区域的 RGB，避免发灰。
    """
    if alpha_u8.ndim == 2:
        alpha = alpha_u8.astype(np.float32) / 255.0
        alpha = alpha[..., None]
    else:
        alpha = alpha_u8.astype(np.float32) / 255.0

    img_float = image_rgb.astype(np.float32) / 255.0
    
    # 提亮力度
    boost_factor = 0.5 + 0.5 * strength
    
    # 只提亮亮部，保护暗部
    luminance = 0.299*img_float[...,0] + 0.587*img_float[...,1] + 0.114*img_float[...,2]
    luminance = luminance[..., None]
    luma_mask = np.clip(luminance * 2.0, 0.0, 1.0) 
    
    # 叠加白色层
    white_layer = (1.0 - alpha) * boost_factor * 0.6 * luma_mask
    
    out = img_float + white_layer
    out = np.clip(out, 0.0, 1.0)
    
    return (out * 255).astype(np.uint8)
# ===== 修改结束 =====


###通道抠图实现####
def refine_alpha_with_channel(
    image_or_array,
    base_mask: np.ndarray,
    mode: str = "auto",        # "auto" / "暗部优先" / "透色优先"
    strength: float = 0.5      # 0.0~1.0，建议默认 0.5
) -> np.ndarray:
    """
    基于“PS 通道抠图”思想的 α 估计：在掩码边界环带内按 I=αF+(1-α)B 估 α，
    并与基础掩码做可控融合，输出 0~255 的 8bit α 通道。
    """
    # --- 输入整理 ---
    if isinstance(image_or_array, Image.Image):
        img = np.array(image_or_array.convert("RGB"))
    else:
        img = image_or_array
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
    H, W = img.shape[:2]

    base = base_mask
    if base.dtype != np.uint8:
        base = (np.clip(base, 0, 1) * 255).astype(np.uint8)
    base_alpha = (base.astype(np.float32)) / 255.0

    # --- 形态学区域: 二值掩码/未知环带/实心区 ---
    binary = (base >= 128).astype(np.uint8)
    radius = max(1, int(2 + strength * 10))  # 力度→环带半径
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dil = cv2.dilate(binary, ker, iterations=1)
    ero = cv2.erode(binary, ker, iterations=1)
    unknown = cv2.subtract(dil, ero)  # 边界环带

    expand = max(0, int(strength * 6))  # 进一步外扩
    if expand > 0:
        ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * expand + 1, 2 * expand + 1))
        unknown = cv2.dilate(unknown, ker2, iterations=1)

    solid_r = max(1, radius * 2)  # 更强腐蚀得到“实心”采样区
    ker_solid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * solid_r + 1, 2 * solid_r + 1))
    fg_solid = cv2.erode(binary, ker_solid, iterations=1)
    bg_solid = cv2.erode((1 - binary), ker_solid, iterations=1)

    I = img.astype(np.float32) / 255.0
    F_mean = np.zeros(3, np.float32)
    B_mean = np.zeros(3, np.float32)

    # --- 颜色统计（中位数更稳健） ---
    for c in range(3):
        vals_f = I[:, :, c][fg_solid > 0]
        vals_b = I[:, :, c][bg_solid > 0]
        if vals_f.size == 0:
            vals_f = I[:, :, c][binary > 0]
        if vals_b.size == 0:
            vals_b = I[:, :, c][binary == 0]
        F_mean[c] = np.median(vals_f) if vals_f.size > 0 else 0.8
        B_mean[c] = np.median(vals_b) if vals_b.size > 0 else 0.2

    den = F_mean - B_mean
    weights = np.abs(den)
    sw = float(weights.sum()) + 1e-6
    if sw < 1e-6:
        return base  # 分离度太低，直接返回原掩码
    weights /= sw

    # --- 未知环带 α 估计 ---
    alpha_unknown = np.zeros((H, W), np.float32)
    eps = 1e-4
    for c in range(3):
        if weights[c] <= 1e-6:
            continue
        ac = (I[:, :, c] - B_mean[c]) / (den[c] + eps)
        alpha_unknown += ac * weights[c]
    alpha_unknown = np.clip(alpha_unknown, 0.0, 1.0)

    # --- 模式 → γ 形状控制 ---
    # 暗部优先：更保守（加深），透色优先：更开放（抬高）
    if mode in ("暗部优先", "dark"):
        gamma = 1.2 + 0.8 * (1 - strength)
        alpha_unknown = np.power(alpha_unknown, gamma)
    elif mode in ("透色优先", "bleed"):
        gamma = max(0.5, 1.0 - 0.5 * strength)
        alpha_unknown = np.power(alpha_unknown, gamma)
    # 其它/auto 不做额外曲线

    # --- 和基础掩码融合，仅在未知环带影响 ---
    mixing = 0.35 + 0.55 * strength  # 力度越大越依赖α估计
    mask_unknown = (unknown > 0).astype(np.float32)
    final = base_alpha * (1.0 - mask_unknown) + \
            ((1 - mixing) * base_alpha + mixing * alpha_unknown) * mask_unknown

    # --- 边缘保持平滑（优先 guided filter，退化为双边滤波） ---
    try:
        import cv2.ximgproc as xip
        final = xip.guidedFilter(
            guide=I, src=final.astype(np.float32), radius=radius * 2 + 1, eps=1e-4
        )
    except Exception:
        d = 5 + 2 * radius
        final = cv2.bilateralFilter(final.astype(np.float32), d=d, sigmaColor=0.1, sigmaSpace=radius * 2 + 1)

    final = np.clip(final, 0.0, 1.0)
    return (final * 255).astype(np.uint8)
# 替换原有函数
def hex_to_rgb(hex_color=None):
    """
    将颜色解析为 (R,G,B):
    - 支持 "#RRGGBB" / "#RGB" / "RRGGBB" / "RGB"
    - 支持 (r,g,b) / [r,g,b]
    - 默认返回绿色
    """
    if isinstance(hex_color, (tuple, list)) and len(hex_color) == 3:
        r, g, b = [int(x) for x in hex_color]
        return (max(0, min(r, 255)), max(0, min(g, 255)), max(0, min(b, 255)))

    s = (hex_color or "").strip()
    if not s:
        return (0, 255, 0)
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return (0, 255, 0)
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b)
    except Exception:
        return (0, 255, 0)
####等比填充函数
def _resize_bg_keep_aspect(bg_array: np.ndarray, target_w: int, target_h: int, mode: str = "cover") -> np.ndarray:
    """等比缩放背景到目标画布；mode='cover' 铺满居中裁切，'contain' 等比缩放+留边"""
    h, w = bg_array.shape[:2]
    if h == 0 or w == 0 or target_w == 0 or target_h == 0:
        return cv2.resize(bg_array, (target_w, target_h))

    src_aspect = w / h
    dst_aspect = target_w / target_h

    if mode == "contain":
        # 等比缩放，留边填充（用边缘像素或纯色都行，这里用边缘像素避免色差）
        if src_aspect > dst_aspect:
            new_w = target_w
            new_h = int(new_w / src_aspect)
        else:
            new_h = target_h
            new_w = int(new_h * src_aspect)
        resized = cv2.resize(bg_array, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, bg_array.shape[2]), dtype=bg_array.dtype)
        # 用背景的边缘像素填充（可改成纯色）
        canvas[...] = resized[0,0] if resized.ndim == 3 else 0
        y0 = (target_h - new_h) // 2
        x0 = (target_w - new_w) // 2
        canvas[y0:y0+new_h, x0:x0+new_w] = resized
        return canvas
    else:
        # cover：等比放大后，居中裁切到目标大小
        if src_aspect < dst_aspect:
            # 竖图 → 先让高度对齐，再裁左右
            new_h = target_h
            new_w = int(new_h * src_aspect)
        else:
            # 横图 → 先让宽度对齐，再裁上下
            new_w = target_w
            new_h = int(new_w / src_aspect)
        resized = cv2.resize(bg_array, (new_w, new_h))
        y0 = max(0, (new_h - target_h) // 2)
        x0 = max(0, (new_w - target_w) // 2)
        return resized[y0:y0+target_h, x0:x0+target_w]

def create_background(background_type, background_data, image_size):
    """创建背景图像
    
    Args:
        background_type: 'image', 'color', 'transparent'
        background_data: 背景数据（图片或颜色）
        image_size: (width, height)
    
    Returns:
        背景图像数组或None（透明背景）
    """
    try:
        w, h = image_size
        
        if background_type == 'image' and background_data is not None:
            # 图片背景
            if isinstance(background_data, Image.Image):
                background_array = np.array(background_data)
            else:
                background_array = background_data

            # 新（避免拉伸，等比处理）：
            if background_array.shape[:2] != (h, w):
                background_array = _resize_bg_keep_aspect(background_array, w, h, mode="cover")
            
            return background_array
            
        elif background_type == 'color' and background_data is not None:
            # 纯色背景
            rgb_color = hex_to_rgb(background_data)
            return np.full((h, w, 3), rgb_color, dtype=np.uint8)
            
        elif background_type == 'transparent':
            # 透明背景
            return None
            
        else:
            # 默认透明背景
            return None
            
    except Exception as e:
        logger.error(f"创建背景失败: {e}")
        return None
# ===== 修改开始：对 semi_strength 做健壮化处理 =====
def _as_float_or_default(x, default=0.5):
    try:
        return float(x)
    except Exception:
        logger.debug(f"[semi_strength] 非数值输入 {x!r} → 回退 {default}")
        return default
# ===== 修改结束 =====


# ===== apply_background_replacement (集成去灰处理) =====
def apply_background_replacement(
    image,
    background_image=None,
    mask=None,
    *,
    model_name='General',
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

        # 3) 获取原始 Mask (Float 0.0-1.0) --- 这里的 m 是最宝贵的原始数据
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

        # === 安全锁 1 ===
        m = np.clip(m, 0.0, 1.0)

        # === ROI 约束 ===
        if roi_mask_fullres is not None:
            roi_bin = (roi_mask_fullres > 0).astype(np.float32)
            if roi_bin.shape[:2] != (H, W):
                roi_bin = cv2.resize(roi_bin, (W, H), interpolation=cv2.INTER_NEAREST)
            if roi_bin.ndim == 3:
                roi_bin = roi_bin[:, :, 0]
            m = m * roi_bin

        # === 核心逻辑分支 ===
        
        if semi_transparent:
            # 半透明模式 [V3 改动]
            mode_map = semi_mode if semi_mode in ("auto", "暗部优先", "透色优先") else "auto"
            strength_val = float(semi_strength)
            
            # ★★★ 关键修改：直接传入 m (float)，不要转 uint8 ★★★
            # 这样 estimate_soft_alpha_inside_mask 拿到的是模型的高精度概率图
            m_final_u8 = estimate_soft_alpha_inside_mask(image_array, m, strength=strength_val, mode=mode_map)

            # 【保留】透色优先模式下，对前景色做去灰提亮（这是必须的）
            if mode_map in ("透色优先", "bleed", "light"):
                image_array_processed = _boost_veil_color(image_array, m_final_u8, strength=strength_val)
            else:
                image_array_processed = image_array
        
        else:
            # === 普通模式 (保持原样) ===
            image_array_processed = image_array 

            # 1. 引导滤波 (保留，用于边缘平滑)
            try:
                guide = cv2.GaussianBlur(image_array, (3, 3), 0.5)
                # 修改后：半径随分辨率动态调整，eps 增大以提升平滑度
                radius = max(2, int(max(H, W) / 400))
                eps = 1e-3
                m_refined = cv2.ximgproc.guidedFilter(guide, m, radius, eps)
            except ImportError:
                m_refined = m 
            except Exception:
                m_refined = m

            m_refined = np.clip(m_refined, 0.0, 1.0)
            
            # 2. Gamma 校正 (默认的一点点优化)
            # m_final_float = np.power(m_refined, 0.8)
            # 修改后：默认保持原始 Mask 的分布，不进行硬缩放
            # gamma_val = 1.0 
            # m_final_float = np.power(m_refined, gamma_val)
            # m_final_u8 = (np.clip(m_final_float, 0.0, 1.0) * 255).astype(np.uint8)

            # 2. Gamma 校正 (为了消除颗粒感，建议将 0.8 改为 1.0)
            m_final_float = np.power(m_refined, 1.0) 
            
            # --- 新增：使用 3x3 极小核进行平滑处理，消除边缘颗粒感 ---
            m_final_float = cv2.GaussianBlur(m_final_float, (3, 3), 0)
            
            m_final_u8 = (np.clip(m_final_float, 0.0, 1.0) * 255).astype(np.uint8)
        # 预览与合成
        mask_preview = Image.fromarray(m_final_u8).convert('RGB')

        if background_image is not None:
            if isinstance(background_image, Image.Image):
                bg_arr = np.array(background_image)
            else:
                bg_arr = background_image
            
            result = replace_background_with_mask(
                image_array=image_array_processed,
                background_array=bg_arr,
                mask=m_final_u8,
                remove_white_halo=remove_white_halo,
                defringe_strength=defringe_strength
            )
        else:
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

# ===== 修改开始：replace_background_with_mask (修复毛发空洞，引入发丝保护) =====
def replace_background_with_mask(
    image_array,
    background_array,
    mask,
    remove_white_halo: bool = False,
    defringe_strength: float | None = None,
    *,
    band_px: int = 2,
    strength: float = 0.7,
    erode_px: int = 1
):
    """
    [修复版] 将前景按 mask 融合到背景
    增加了：Mask闭运算预处理（防空洞）、限制侵蚀核大小。
    """
    # ... (前段代码保持不变: 规范化 fg, bg, m) ...
    # ---------- 规范化前景到 RGB uint8 ----------
    fg = np.asarray(image_array)
    if fg.dtype != np.uint8:
        fg = np.clip(fg, 0, 255).astype(np.uint8)
    if fg.ndim == 2:
        fg = cv2.cvtColor(fg, cv2.COLOR_GRAY2RGB)
    elif fg.ndim == 3 and fg.shape[2] >= 4:
        fg = fg[:, :, :3]
    H, W = fg.shape[:2]

    # ---------- 规范化背景到 RGB uint8 ----------
    bg = np.asarray(background_array)
    if bg.dtype != np.uint8:
        bg = np.clip(bg, 0, 255).astype(np.uint8)
    if bg.ndim == 2:
        bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
    elif bg.ndim == 3 and bg.shape[2] >= 4:
        bg = bg[:, :, :3]
    
    # 等比裁切/缩放背景
    if bg.shape[:2] != (H, W):
        bg = _resize_bg_keep_aspect(bg, W, H, mode="cover")

    # ---------- 规范化 mask -> 2D uint8(0..255) ----------
    m = np.asarray(mask)
    if m.ndim == 3:
        if m.shape[2] == 1: m = m[:, :, 0]
        else: m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

    if m.dtype == np.uint8:
        a_u8 = m
    else:
        m = m.astype(np.float32)
        if m.max() <= 1.0 + 1e-6:
            a_u8 = (np.clip(m, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        else:
            a_u8 = np.clip(m, 0.0, 255.0).astype(np.uint8)

    # 修复核心 1: 预先闭运算，填补 Mask 内部的微小噪点孔洞 
    # 防止 erode 操作把原本只有 1px 的噪点扩大成明显的洞
    ker_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    a_u8 = cv2.morphologyEx(a_u8, cv2.MORPH_CLOSE, ker_close)

    # ---------- 可选：去白边 ----------
    if remove_white_halo:
        if defringe_strength is not None:
            mp = _map_defringe_strength(defringe_strength)
            band_px = mp["band_px"]
            strength = mp["strength"]
            erode_px = mp["erode_px"]
        
        # 1) 颜色去污染
        try:
            fg = _color_decontam_edge(fg, a_u8, band_px=band_px, strength=strength)
        except Exception:
            pass
        
        # 2) 智能收边
        if erode_px > 0:
            # ★★★ 修复核心 2: 严格限制 cap，防止高分辨率下侵蚀过度 ★★★
            H_img, W_img = a_u8.shape[:2]
            bd_px = _scale_px_for_image(band_px, H_img, W_img, base=1024, cap=4)
            er_px = _scale_px_for_image(erode_px, H_img, W_img, base=1024, cap=3) # 侵蚀最多3px

            if er_px > 0:
                ker_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * er_px + 1, 2 * er_px + 1))
                ker_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bd_px + 1, 2 * bd_px + 1))

                fg_mask = (a_u8 > 0).astype(np.uint8)
                dil = cv2.dilate(fg_mask, ker_b, iterations=1)
                ero = cv2.erode(fg_mask, ker_b, iterations=1)
                band = cv2.subtract(dil, ero)

                a_eroded = cv2.erode(a_u8, ker_e, iterations=1)

                w_hair = _hair_protect_weight(fg, a_u8, band)
                
                # 混合：w_hair 越大，越保留原 mask
                erode_eff = (1.0 - 0.9 * w_hair) # 稍微增加保护力度 (0.8 -> 0.9)
                
                a_blend = erode_eff * a_eroded.astype(np.float32) + (1.0 - erode_eff) * a_u8.astype(np.float32)
                a_u8 = np.where(band > 0, a_blend, a_u8).astype(np.uint8)

    # ---------- α 融合 ----------
    a = (a_u8.astype(np.float32) / 255.0)[..., None]
    out = fg.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)
    out = np.clip(out, 0, 255).astype(np.uint8)

    return Image.fromarray(out)
# ===== 修改结束 =====

###二值化+轻后处理###
def _to_binary_mask(mask: np.ndarray, *, use_otsu: bool = True) -> np.ndarray:
    """
    将 0~255 的软 mask 变成真正的二值 0/255，并做一次轻量形态学清理，避免小孔/毛刺。
    """
    m = mask
    if m.dtype != np.uint8:
        m = (np.clip(m, 0, 1) * 255).astype(np.uint8)

    # 阈值：默认 Otsu，自适应不同图像；如需固定阈值可把 use_otsu=False 改成固定 128
    if use_otsu:
        _, m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, m = cv2.threshold(m, 128, 255, cv2.THRESH_BINARY)

    # 轻量清理：开运算去毛刺 + 闭运算补小孔
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=1)

    return m.astype(np.uint8)

def create_transparent_result(image_array, mask, remove_white_halo: bool = False, defringe_strength: float = 0.7):

    """
    生成带透明通道的PNG结果（可选去白边）。
    - 兼容灰度 / RGB / RGBA 输入
    - 兼容 0..1 / 0..255 的 mask，且会自动对齐到图像尺寸
    - 先构造 rgba 再处理，保证不会出现“未赋值就引用”的错误
    """
    try:
        # 1) 规范化图像到 RGB
        img = np.asarray(image_array)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            img_rgb = img
        elif img.ndim == 3 and img.shape[2] >= 4:
            img_rgb = img[:, :, :3].copy()
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        H, W = img_rgb.shape[:2]

        # 2) 规范化 mask → 2D uint8(0..255) 且尺寸匹配
        if mask is None:
            # 没有 mask 就返回全不透明（与旧逻辑兼容）
            a = np.full((H, W), 255, dtype=np.uint8)
        else:
            m = np.asarray(mask)
            if m.ndim == 3:
                # 多通道 mask 取单通道；若是 RGB，转灰度更稳
                if m.shape[2] == 1:
                    m = m[:, :, 0]
                else:
                    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

            if m.dtype == np.uint8:
                a = m
            else:
                m = m.astype(np.float32)
                mx = float(m.max()) if m.size else 1.0
                if mx <= 1.0 + 1e-6:
                    a = (np.clip(m, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                else:
                    a = (np.clip(m, 0.0, 255.0) + 0.5).astype(np.uint8)

        # 3) 组装 RGBA（注意：这里用 rgba，随后若需要保持旧变量名，再赋给 input_rgba）
        rgba = np.dstack([img_rgb, a]).astype(np.uint8)
        rgba = np.ascontiguousarray(rgba)

        # 4) 可选：去白边（把“规范化后的 a”一并传入）
        if remove_white_halo:
            params = _map_defringe_strength(defringe_strength)
            rgba = _remove_white_halo_rgba(
                rgba, a,
                band_px=params["band_px"],
                strength=params["strength"],
                erode_px=params["erode_px"]
            )

        # 5) 返回 PIL Image
        return Image.fromarray(rgba[:, :, :4])

    except Exception as e:
        logger.error(f"创建透明背景失败: {e}")
        return None
def _edge_band_from_mask(mask: np.ndarray, band_px: int = 2):
    """
    基于二值 mask 生成“边带”区域与一个 0..1 的边带强度 a_eff（靠近外轮廓更小，靠内更大），
    以便在硬边时也能对边缘做颜色去污染。
    """
    m = mask
    if m.dtype != np.uint8:
        m = (np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8)

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    dil = cv2.dilate(m, ker, iterations=1)
    ero = cv2.erode(m, ker, iterations=1)
    band = cv2.subtract(dil, ero)  # 仅边带 0..255

    # 从外到里做一次距离变换，得到“伪 α”：越靠里值越大
    outside = (m == 0).astype(np.uint8)
    dist = cv2.distanceTransform(1 - outside, cv2.DIST_L2, 3)  # 内部距离
    if dist.max() > 0:
        a_eff = dist / (band_px + 1e-6)
        a_eff = np.clip(a_eff, 0.05, 1.0)
    else:
        a_eff = (m > 0).astype(np.float32)

    a_eff = a_eff.astype(np.float32)
    band_mask = (band > 0).astype(np.uint8)
    return band_mask, a_eff
###srgb线性化###
def _srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def _linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1 / 2.4)) - 0.055)
####按分辨率自适应###
def _scale_px_for_image(px: int, h: int, w: int, base: int = 1024, cap: int = 4) -> int:
    """
     按图像分辨率把像素核做等比放大/缩小
    修复问题：原版 cap=10 太大，导致高分辨率图边缘被切掉太多。
    强制限制最大核为 4px (通常 1-3px 足矣)。
    """
    if px <= 0:
        return 0
    scale = max(h, w) / float(base)
    px_scaled = int(round(px * max(1.0, scale)))
    # 强制限制上限，防止“啃洞”
    return max(1, min(cap, px_scaled))

def _map_defringe_strength(s: float):
    """
    输入力度 0..1 -> 输出内部参数：
      - strength：颜色去污染混合强度（越大越“拉回”前景色）
      - band_px ：边带宽度（像素）
      - erode_px：仅对边带的轻微收边像素（像素）
    新版上限更高：erode_px 最多 4px，band_px 最多 4px（会再按分辨率缩放）
    """
    s = float(max(0.0, min(1.0, s)))

    # 颜色去污染强度：0.45..0.95（稍强一点，让高档位更明显）
    strength = 0.45 + 0.50 * s

    # 边带宽度：1/2/3/4 px，分段更平滑
    if s < 0.30:
        band_px = 1
    elif s < 0.60:
        band_px = 2
    elif s < 0.85:
        band_px = 3
    else:
        band_px = 4

    # 轻收边像素：0..4 px，上限提高
    if s < 0.35:
        erode_px = 0
    elif s < 0.55:
        erode_px = 1
    elif s < 0.75:
        erode_px = 2
    elif s < 0.90:
        erode_px = 3
    else:
        erode_px = 4

    return dict(strength=strength, band_px=band_px, erode_px=erode_px)

##颜色去污染###估计背景色并按混合公式反解前景颜色
def _color_decontam_edge(rgb_u8: np.ndarray, mask: np.ndarray, band_px: int = 2, strength: float = 0.7):
    """
    仅在边带ROI做颜色去污染；对疑似发丝的像素降低强度（保护细节）。
    """
    import numpy as np, cv2
    H, W = rgb_u8.shape[:2]
    rgb = rgb_u8 if rgb_u8.dtype == np.uint8 else np.clip(rgb_u8, 0, 255).astype(np.uint8)

    m = mask
    if m.dtype != np.uint8:
        m = (np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8)

    # band 及 ROI
    band_px = int(max(1, band_px))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    fg = (m > 0).astype(np.uint8)
    dil = cv2.dilate(fg, ker, iterations=1)
    ero = cv2.erode(fg, ker, iterations=1)
    band = cv2.subtract(dil, ero)

    ys, xs = np.where(band > 0)
    if ys.size == 0:
        return rgb

    pad = 8
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    fg_roi  = rgb[y0:y1, x0:x1]
    m_roi   = m[y0:y1, x0:x1]
    band_roi= band[y0:y1, x0:x1]

    # 背景估计（已有降采样 + 早退优化的版本）
    inpaint_r = max(3, int(3 + 4 * band_px + 6 * float(strength)))
    B_full = _estimate_background_inpaint(rgb, m, radius=inpaint_r)
    B_roi  = B_full[y0:y1, x0:x1]

    # 真实α vs 伪α
    if mask.dtype != np.uint8:
        a_real = np.clip(mask, 0, 1).astype(np.float32)[y0:y1, x0:x1]
    else:
        a_real = (m_roi.astype(np.float32) / 255.0)

    outside = (m_roi == 0).astype(np.uint8)
    dist = cv2.distanceTransform(1 - outside, cv2.DIST_L2, 3)
    a_eff = np.clip(dist / (band_px + 1e-6), 0.05, 1.0).astype(np.float32)
    a_use = np.where((a_real > 0) & (a_real < 1), a_real, a_eff)

    # sRGB<->Linear
    def _srgb_to_linear(x):
        x = np.clip(x, 0.0, 1.0)
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    def _linear_to_srgb(x):
        x = np.clip(x, 0.0, 1.0)
        return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1 / 2.4)) - 0.055)

    C_lin = _srgb_to_linear(fg_roi.astype(np.float32) / 255.0)
    B_lin = _srgb_to_linear(B_roi.astype(np.float32) / 255.0)

    eps = 1e-4
    F_lin = (C_lin - (1.0 - a_use)[..., None] * B_lin) / np.maximum(a_use[..., None], eps)
    F_lin = np.clip(F_lin, 0.0, 1.0)
    F = _linear_to_srgb(F_lin)

    # —— 发丝保护：降低颜色去污染强度 ——
    # 在疑似发丝处，把有效强度从 S 降到 S*(1-0.6*w)
    w_hair = _hair_protect_weight(fg_roi, m_roi, band_roi)  # 0..1
    S = float(strength)
    S_loc = (S * (1.0 - 0.6 * w_hair)).astype(np.float32)

    out_roi = (1.0 - S_loc[..., None]) * (fg_roi.astype(np.float32) / 255.0) + S_loc[..., None] * F
    out_roi = np.clip(out_roi * 255.0 + 0.5, 0, 255).astype(np.uint8)

    out = rgb.copy()
    out[y0:y1, x0:x1] = out_roi
    return out

###去白边实现
def _remove_white_halo_rgba(
    rgba: np.ndarray,
    mask: np.ndarray,
    band_px: int = 2,
    strength: float = 0.7,
    erode_px: int = 1,
):
    """
    [修复版] 去白边（透明导出）
    """
    import numpy as _np, cv2
    assert rgba.ndim == 3 and rgba.shape[2] >= 4, "expect HxWx4 RGBA"
    H, W = rgba.shape[:2]

    def _scale_px(px: int, base: int = 1024, cap: int = 10) -> int:
        if px <= 0: return 0
        scale = max(H, W) / float(base)
        return max(1, min(cap, int(round(px * max(1.0, scale)))))

    if mask.dtype != _np.uint8:
        m = (_np.clip(mask, 0, 1) * 255.0 + 0.5).astype(_np.uint8)
    else:
        m = mask

    # ★★★ 修复点 1: 预先闭运算填孔 ★★★
    ker_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker_close)

    rgb = rgba[:, :, :3]
    a_u8_base = rgba[:, :, 3]

    # ★★★ 修复点 2: 限制侵蚀核上限 cap=3 ★★★
    bd_px = _scale_px(int(band_px), cap=4)
    er_px = _scale_px(int(erode_px), cap=3)

    try:
        rgb_fixed = _color_decontam_edge(rgb, m, band_px=max(1, bd_px), strength=float(strength))
    except Exception:
        rgb_fixed = rgb

    if er_px > 0:
        ker_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bd_px + 1, 2 * bd_px + 1))
        ker_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * er_px + 1, 2 * er_px + 1))

        fg = (m > 0).astype(_np.uint8)
        dil = cv2.dilate(fg, ker_b, iterations=1)
        ero = cv2.erode(fg, ker_b,  iterations=1)
        band = cv2.subtract(dil, ero)

        m_er = cv2.erode(m, ker_e, iterations=1)

        w_hair = _hair_protect_weight(rgb, m, band)
        
        # 增加保护力度
        erode_eff = (1.0 - 0.9 * w_hair).astype(_np.float32)
        m_blend = (erode_eff * m_er.astype(_np.float32) + (1.0 - erode_eff) * m.astype(_np.float32))
        a_u8 = _np.where(band > 0, m_blend, m).astype(_np.uint8)
    else:
        a_u8 = a_u8_base.copy()

    # 轻羽化
    a_out = cv2.GaussianBlur(a_u8, (0, 0), sigmaX=0.6, sigmaY=0.6)

    out = _np.dstack([rgb_fixed, a_out]).astype(_np.uint8)
    return out

###轻微收边###
def _defringe_alpha_only(mask: np.ndarray, px: int = 1) -> np.ndarray:
    """
    仅对 alpha/mask 做轻微收边（px 像素），避免白边。
    输入可为 0..255 的 uint8 或 0..1 的 float。
    """
    m = mask.copy()
    if m.dtype != np.uint8:
        m = (np.clip(m, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    if px > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        # 只对边带（0<alpha<255）收边，发丝保守一点
        band = ((m > 0) & (m < 255)).astype(np.uint8)
        er   = cv2.erode(m, ker, iterations=1)
        m    = np.where(band > 0, er, m).astype(np.uint8)
    return m

def replace_background(image, background, mask=None):
    """替换背景（保持向后兼容）"""
    try:
        # 预处理输入图像
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        # 如果没有提供mask，尝试生成
        if mask is None:
            mask = segment_image(image)
            if mask is None:
                raise ValueError("无法生成分割mask")
        
        # 处理背景
        if isinstance(background, Image.Image):
            background_array = np.array(background)
        else:
            background_array = background
        
        # 确保尺寸匹配
        h, w = image_array.shape[:2]
        # 新（避免拉伸，等比处理）：
        if background_array.shape[:2] != (h, w):
            background_array = _resize_bg_keep_aspect(background_array, w, h, mode="cover")
        
        return replace_background_with_mask(image_array, background_array, mask)
        
    except Exception as e:
        logger.error(f"背景替换失败: {e}")
        logger.error(traceback.format_exc())
        return None

def process_files(files, file_type, background_image=None, progress=gr.Progress()):
    """统一的文件处理函数，支持单个和批量处理
    
    Args:
        files: 单个文件或文件列表
        file_type: 'image' 或 'video'
        background_image: 背景图片（可选）
        progress: 进度回调函数
    
    Returns:
        单个文件: (result, status_message)
        批量文件: (zip_path, status_message)
    """
    if not files:
        return None, f"请上传{file_type}文件"
    
    # 检查是否为批量处理
    is_batch = isinstance(files, list) and len(files) > 1
    if not isinstance(files, list):
        files = [files]
    
    try:
        if is_batch:
            return _process_batch_files(files, file_type, background_image, progress)
        else:
            return _process_single_file(files[0], file_type, background_image, progress)
            
    except Exception as e:
        error_msg = f"{file_type}处理失败: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None, error_msg

def _process_single_file(file, file_type, background_image, progress):
    """处理单个文件"""
    try:
        if file_type == 'image':
            return _process_single_image(file, background_image)
        elif file_type == 'video':
            # 兜底用全局当前配置（与 process_video 保持一致）
            model_name = globals().get("current_model_name", "General")
            res = globals().get("current_resolution", 1024)
            return _process_single_video(
                input_video=file,
                bg_img=background_image,
                model_name=model_name,
                input_size=(res, res),
                progress=progress
            )
        else:
            return None, f"不支持的文件类型: {file_type}"
            
    except Exception as e:
        error_msg = f"处理{file_type}失败: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def _process_single_image(
    image,
    background_image,
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
            or globals().get("current_model_name")
            or globals().get("current_loaded_model_name")
            or "General"
        )
        try:
            resolution = int(
                _resolution
                or globals().get("current_resolution")
                or (
                    globals().get("current_loaded_resolution")[0]
                    if isinstance(globals().get("current_loaded_resolution"), tuple)
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
            background_image=background_image,
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
            import os
            try:
                from datetime import datetime
            except Exception:
                datetime = None  # 没导入也不影响主流程

            out_dir = globals().get("PRED_OUTPUT_DIR") or os.path.join(os.getcwd(), "preds-BiRefNet")
            os.makedirs(out_dir, exist_ok=True)

            # 尝试获取原文件名；否则用时间戳
            base_name = None
            if hasattr(image, "filename") and getattr(image, "filename"):
                import os as _os
                base_name = _os.path.splitext(_os.path.basename(image.filename))[0]
            elif hasattr(image, "name") and getattr(image, "name"):
                import os as _os
                base_name = _os.path.splitext(_os.path.basename(image.name))[0]
            if not base_name:
                base_name = datetime.now().strftime("%Y%m%d_%H%M%S") if datetime else "result"

            save_path = os.path.join(out_dir, f"single_{base_name}.png")

            # 优先使用工程里的安全保存函数；没有就兜底保存
            _save_fn = globals().get("_save_image_safe")
            if callable(_save_fn):
                _save_fn(result, save_path)
            else:
                try:
                    # 直接保存 PIL.Image
                    if hasattr(result, "save"):
                        result.save(save_path)
                    else:
                        # 兜底：numpy -> PIL 再保存
                        from PIL import Image as _PILImage
                        import numpy as _np
                        _PILImage.fromarray(_np.asarray(result)).save(save_path)
                except Exception as _e:
                    logger.warning(f"结果保存失败（不影响前端展示）：{_e}")

            logger.info(f"🖼️ 单张结果已自动保存：{save_path}")
        except Exception as se:
            logger.warning(f"单张结果保存失败（不影响前端展示）：{se}")

        return result, mask_preview

    except Exception as e:
        import traceback
        error_msg = f"处理失败: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None, error_msg

    finally:
        # === 强制显存/内存回收 ===
        # 无论处理成功还是失败，都清理战场，防止 8G 显存溢出
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def _process_single_video(
    input_video: str,
    bg_img,
    model_name: str,
    input_size: Tuple[int, int],
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

    import numpy as np, os, traceback

    try:
        # === 1) 模型准备 ===
        global model
        if model is None or (globals().get("current_loaded_model_name") != model_name):
            ok = load_model(model_name, input_size)
            if not ok:
                raise RuntimeError(f"无法加载模型 {model_name}")

        # === 2) 打开视频 ===
        video = mp.VideoFileClip(input_video)
        total_frames = max(1, int(video.fps * video.duration))
        vw, vh = video.size

        logger.info(f"🎥 视频信息: {total_frames} 帧, {video.fps:.2f} FPS, {video.duration:.2f}s")
        logger.info(f"🧠 当前推理分辨率: {input_size[0]}x{input_size[1]}")

        # === 准备背景 ===
        prepared_bg_arr = None
        if bg_img is not None:
            arr = np.array(bg_img)
            if arr.ndim == 2: arr = np.stack([arr, arr, arr], axis=2)
            elif arr.ndim == 3 and arr.shape[2] >= 4: arr = arr[:, :, :3]
            prepared_bg_arr = _resize_bg_keep_aspect(arr, vw, vh, mode="cover")
        else:
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
                    # 使用 V2.0 算法
                    mode_map = semi_mode if semi_mode in ("auto", "暗部优先", "透色优先") else "auto"
                    m_u8 = estimate_soft_alpha_inside_mask(
                        frame, m_u8, strength=float(semi_strength), mode=mode_map
                    )
                    
                    # 3) 【修复点】颜色修正 (去灰/提亮)
                    # 只有在开启半透明且模式匹配时才处理前景颜色
                    if mode_map in ("透色优先", "bleed", "light"):
                        frame_processed = _boost_veil_color(frame, m_u8, strength=float(semi_strength))
                    else:
                        frame_processed = frame
                else:
                    frame_processed = frame

                # 4) 合成
                m_f = m_u8.astype(np.float32) / 255.0
                m_f = m_f[..., None] # (H,W,1)
                
                # out = fg * a + bg * (1-a)
                out = (frame_processed.astype(np.float32) * m_f + 
                       prepared_bg_arr.astype(np.float32) * (1.0 - m_f)).astype(np.uint8)

                if device.type == "cuda" and (processed_frames % 500 == 0):
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
        
        import gc
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        return out_path, f"✅ 完成：{os.path.basename(out_path)}"

    except Exception as e:
        logger.error(f"视频处理失败: {e}")
        traceback.print_exc()
        return None, f"视频处理失败：{e}"

def _process_batch_files(
    files,
    file_type='image',
    background_image=None,
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
    """
    import os, zipfile, traceback, gc
    from datetime import datetime

    results = []
    PRED_OUTPUT_DIR = os.path.join(os.getcwd(), "preds-BiRefNet")
    os.makedirs(PRED_OUTPUT_DIR, exist_ok=True)
    
    total = len(files)

    # ================= 批量图片 =================
    if file_type == 'image':
        for idx, f in enumerate(files, 1):
            if progress:
                progress((idx - 1) / max(1, total), desc=f"图片 {idx}/{total}")

            # 读取
            try:
                if hasattr(f, "read"):
                    import PIL.Image as _PIL
                    img = _PIL.open(f.name).convert("RGB")
                    img = np.array(img)
                elif isinstance(f, str):
                    import PIL.Image as _PIL
                    img = _PIL.open(f).convert("RGB")
                    img = np.array(img)
                else:
                    img = np.asarray(f)
            except Exception as e:
                logger.error(f"读取图片失败 {f}: {e}")
                continue

            # 处理
            result_img, _ = apply_background_replacement(
                image=img,
                background_image=background_image,
                model_name=model_name,
                input_size=input_size,
                semi_transparent=semi_enable,
                semi_strength=semi_strength,
                semi_mode=semi_mode,
                remove_white_halo=defringe,
                defringe_strength=defringe_strength
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

    # ================= 批量视频 =================
    elif file_type == 'video':
        for idx, f in enumerate(files, 1):
            if progress:
                progress((idx - 1) / max(1, total), desc=f"视频 {idx}/{total}")
            
            v_path = f.name if hasattr(f, 'name') else str(f)
            
            # 调用单视频处理逻辑
            out_path, msg = _process_single_video(
                input_video=v_path,
                bg_img=background_image,
                model_name=model_name,
                input_size=input_size,
                bg_color=bg_color,
                progress=None, # 内部不更新总进度，以免冲突
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

def process_single_frame(frame, background_image, model_name='General', input_size=(1024, 1024)):
    """处理单帧视频"""
    try:
        result, _ = apply_background_replacement(
            frame, background_image,
            model_name=model_name, input_size=input_size
        )
        return np.array(result) if result is not None else frame
    except Exception as e:
        logger.error(f"处理帧失败: {e}")
        return frame

# 保持向后兼容的包装函数
def process_image(input_image, background_image=None):
    """处理单张图片的包装函数"""
    return _process_single_image(input_image, background_image)

# ===== 修改开始：记录全局状态 =====
# ===== 修改开始：修复分辨率不生效 BUG =====
def process_image_with_settings(
    image,
    background_image=None,
    semi_enable=False,
    semi_strength=0.5,
    semi_mode='auto',
    defringe=False,
    defringe_strength=0.7,
    # === ROI 参数 ===
    roi_enable=False,
    roi_editor_value=None,
    roi_meta: dict | None = None,
    roi_crop_before=True,
    roi_pad_px=16,
    ui_resolution=1024,
):
    """
    单张图片处理入口（UI回调）
    返回：result_image, mask_preview
    """
    # 1. ★★★ 修复点：直接信任 UI 传进来的分辨率，不要再读取全局变量覆盖它了 ★★★
    if ui_resolution is not None and int(ui_resolution) >= 256:
        res = int(ui_resolution)
    else:
        res = 1024

    # 打印一下确认日志，方便你调试
    print(f"🚀 开始处理：目标分辨率={res}x{res} (如果这里显示 2048 说明生效了)")

    # 获取当前模型名
    model_name = (globals().get("current_model_name")
                  or globals().get("current_loaded_model_name")
                  or "General")

    # 生成 full-res ROI mask（若启用且有值）
    roi_mask_fullres = None
    try:
        if roi_enable and roi_editor_value and roi_meta:
            roi_mask_fullres = _editor_layers_to_mask_fullres(roi_editor_value, roi_meta)
    except Exception as _e:
        logger.warning(f"[ROI] 提取失败，忽略 ROI：{_e}")

    # 调用实际处理
    return _process_single_image(
        image=image,
        background_image=background_image,
        semi_enable=semi_enable,
        semi_strength=semi_strength,
        semi_mode=semi_mode,
        defringe=defringe,
        defringe_strength=defringe_strength,
        _model_name=model_name,
        _resolution=res,    # <--- 传入修复后的 res
        # 贯通 ROI
        _roi_mask_fullres=roi_mask_fullres,
        _roi_crop_before=bool(roi_crop_before),
        _roi_pad_px=int(roi_pad_px),
    )
# ===== 修改结束 =====


def process_video(input_video, background_image=None, bg_color="#00FF00",
                  progress=gr.Progress(),
                  semi_enable=False, semi_strength=0.5, semi_mode='auto'):
    global current_model_name, current_resolution
    model_name = current_model_name if 'current_model_name' in globals() else 'General'
    resolution = current_resolution if 'current_resolution' in globals() else 1024
    return _process_single_video(
        input_video=input_video,
        bg_img=background_image,
        model_name=model_name,
        input_size=(resolution, resolution),
        bg_color=bg_color,
        progress=progress,
        semi_enable=semi_enable,
        semi_strength=semi_strength,
        semi_mode=semi_mode
    )

def process_batch_images(
    files,
    background_image=None,
    # ↓ 同样去掉*，让Gradio可以按位置传参
    semi_enable=False,
    semi_strength=0.5,
    semi_mode='auto',
    defringe=False,
    defringe_strength=0.7,
    progress=None,
):
    """
    批量图片处理入口（UI回调）
    返回：zip_path(或None), 状态文本
    """
    # 全局：模型名 & 分辨率（滑杆）
    model_name = (globals().get("current_model_name")
                  or globals().get("current_loaded_model_name")
                  or "General")
    res = (globals().get("current_resolution")
           or (globals().get("current_loaded_resolution")[0]
               if isinstance(globals().get("current_loaded_resolution"), tuple) else None)
           or 1024)
    try:
        res = int(res)
    except Exception:
        res = 1024
    input_size = (res, res)

    # 交给批量主函数（它本身不是Gradio回调，保留原签名即可）
    return _process_batch_files(
        files=files,
        file_type='image',
        background_image=background_image,
        progress=progress,
        model_name=model_name,
        input_size=input_size,
        semi_enable=semi_enable,
        semi_strength=semi_strength,
        semi_mode=semi_mode,
        defringe=defringe,
        defringe_strength=defringe_strength
    )

def process_batch_videos(files, background_image=None, resolution=1024,
                         bg_color="#00FF00",
                         progress=gr.Progress(), model_name=None,
                         semi_enable=False, semi_strength=0.5, semi_mode='auto'):
    active_model = model_name or globals().get("current_model_name") \
                   or globals().get("current_loaded_model_name") or "General"
    return _process_batch_files(
        files, 'video', background_image, progress,
        model_name=active_model, input_size=(int(resolution), int(resolution)),
        bg_color=bg_color,
        semi_enable=semi_enable, semi_strength=semi_strength, semi_mode=semi_mode
    )

# ===== 半透明扣除：说明文案（复用） =====
SEMI_TIP = """
**扣除半透明**  
- 开关：默认关闭以保持旧版本行为。  

**力度 / 区域大小（0–1）** 影响 inpaint 半径、融合强度、平滑半径。  
建议：**烟雾** 0.6–0.8；**薄纱/纱网** 0.4–0.6；**玻璃/水面** 0.3–0.5。

**模式**  
- **auto**：自动选择，不再额外弯曲 α 曲线。  
- **暗部优先**：适合阴影、烟雾略压暗背景（更保守，防止过度透明）。  
- **透色优先**：适合薄纱、雾气高亮/低饱和（更开放，通透感更强）。  
半身人像/发丝建议先选 **Matting**，再开启本功能。

"""
# —— 根据模式返回对应注释（只显示当前选择的那一条） ——
def _semi_mode_hint_text(mode: str) -> str:
    mapping = {
        "auto": "🧠 **auto**：自动选择，不再额外处理。",
        "暗部优先": "🌑 **暗部优先**：适合阴影、烟雾略压暗背景（更保守，防止过度透明）。",
        "透色优先": "✨ **透色优先**：适合薄纱、雾气高亮/低饱和（更开放，通透感更强）。",
    }
    return mapping.get(mode, mapping["auto"])

def build_semi_controls():
    """创建“扣除半透明”开关"""
    semi_enable = gr.Checkbox(label="扣除半透明", value=False)

    with gr.Group(visible=False) as semi_opts:
        semi_strength = gr.Slider(
            label="透明度增强 / 区域阈值",
            minimum=0.0, maximum=1.0, step=0.05, value=0.5,
            # 更新了说明
            info="基于模型预测进行增强：值越大，半透明区域越通透（Gamma 压制）。建议：薄纱/冰块 0.5+，普通物体 0.2。"
        )
        semi_mode = gr.Radio(
            label="处理模式",
            choices=["auto", "暗部优先", "透色优先"],
            value="auto"
        )
        mode_hint = gr.Markdown(_semi_mode_hint_text("auto"))

    semi_enable.change(
        fn=lambda on: gr.update(visible=on),
        inputs=semi_enable,
        outputs=semi_opts
    )

    semi_mode.change(
        fn=_semi_mode_hint_text,
        inputs=semi_mode,
        outputs=mode_hint
    )

    return semi_enable, semi_strength, semi_mode

def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(
        title="BiRefNet 背景移除工具",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .tab-nav {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        .tab-nav button {
            color: white !important;
            font-weight: bold !important;
        }
        .tab-nav button.selected {
            background: rgba(255,255,255,0.2) !important;
        }
        """
    ) as interface:
        
        gr.Markdown(
            """
            # 🎯 BiRefNet 背景移除工具
            
            **功能特点：**
            - 🖼️ 支持单张图片和批量图片处理
            - 🎬 支持单个视频和批量视频处理
            - 🎨 支持自定义背景图片或默认绿色背景
            - 📦 批量处理结果自动打包下载
            - ⚡ 高性能GPU加速推理
           
            
            **使用说明：** 上传图片或视频，可选择背景图片，系统将自动移除原背景并替换为指定背景（默认绿色）。
            """
        )
                # ===== 修改开始：新增模型与分辨率设置UI =====
        # ===== 简化后的模型与分辨率设置 =====
        with gr.Accordion("⚙️ 模型与分辨率设置", open=True):
            # === 模型下拉框：显示备注 ===
            # 构建带描述的可视化选项
            model_choices = [f"{key} - {desc}" for key, desc in model_descriptions.items()]

            model_choice = gr.Dropdown(
                label="选择模型任务",
                choices=model_choices,
                value=model_choices[0],
                info="选择适合任务的模型，系统会自动加载对应权重"
            )

            resolution = gr.Slider(
                label="输入分辨率",
                minimum=256,
                maximum=2048,
                step=64,
                value=1024,
                info="设置模型推理输入分辨率"
            )
            resolution_info = gr.Markdown(
                value="⚙️ 当前输入分辨率：1024×1024\n💨 推理速度：中等（推荐）\n🎯 预估精度：高",
                label="分辨率性能提示"
            )

            status_box = gr.Textbox(label="状态", interactive=False)

            def on_model_change(selected_model):
                print(f"🪄 用户选择了模型：{selected_model}")
                status = "正在加载模型，请稍候..."
                ok = load_model(selected_model, (1024, 1024))
                if ok:
                    status = f"✅ 模型已加载：{selected_model}"
                else:
                    status = f"❌ 模型加载失败：{selected_model}"
                return status
            def on_resolution_change(res):
                """根据滑块值动态提示性能、精度与显存预估"""
                res = int(res)
                # 估算显存消耗（经验值）
                base_res = 1024
                base_mem_gb = 2.5  # 在 RTX3090 上 1024×1024 大约占 2.5 GB
                estimated_mem = base_mem_gb * (res / base_res) ** 2

                # 设置性能描述
                if res <= 512:
                    speed = "🚀 非常快"
                    quality = "⚪ 精度较低"
                    note = "适合实时预览或低显存设备"
                elif res <= 1024:
                    speed = "⚡ 中等（推荐）"
                    quality = "🟢 精度高"
                    note = "适合大多数任务"
                elif res <= 1536:
                    speed = "🐢 稍慢"
                    quality = "🔵 精度更高"
                    note = "适合高质量抠图"
                else:
                    speed = "🐌 较慢"
                    quality = "🟣 极高精度"
                    note = "适合静态图片的最高质量输出"

                msg = (
                    f"⚙️ 当前输入分辨率：{res}×{res}\n"
                    f"{speed} · {quality}\n"
                    f"🧠 预估显存占用：约 {estimated_mem:.1f} GB\n"
                    f"💡 {note}"
                )

                logger.info(f"🎚️ 分辨率滑块调整为 {res}x{res}，预估显存 {estimated_mem:.1f} GB")
                return msg

            def on_model_change(selected):
                """解析真实模型名并加载"""
                # 提取短名（例如 "General - 通用版" → "General"）
                short_name = selected.split(" - ")[0].strip()
                print(f"🧠 用户选择模型: {short_name}")
                status = f"正在加载模型 {short_name} ..."
                ok = load_model(short_name, (1024, 1024))
                if ok:
                    status = f"✅ 模型加载成功：{short_name}"
                else:
                    status = f"❌ 模型加载失败：{short_name}"
                return status

            model_choice.change(
                fn=on_model_change,
                inputs=[model_choice],
                outputs=[status_box]
            )
            resolution.change(
                fn=on_resolution_change,
                inputs=[resolution],
                outputs=[resolution_info]
            )
            def update_resolution_limit(selected_model):
                """
                根据选择的模型动态限制分辨率范围。
                Lite 模型在低于 1024 分辨率下表现不稳定。
                """
                min_res, max_res = 256, 2048
                default_value = 1024

                if "lite-2K" in str(selected_model):
                    min_res = 1024
                    logger.info(f"⚠️ {selected_model} 模型仅支持分辨率 >=1024，已调整滑块下限")
                    return gr.update(
                        minimum=min_res,
                        maximum=max_res,
                        value=max(default_value, min_res),
                        step=64,
                        label="输入分辨率 (Lite 模型限制 ≥1024)"
                    )
                else:
                    return gr.update(
                        minimum=256,
                        maximum=2048,
                        value=1024,
                        step=64,
                        label="输入分辨率"
                    )

            # 绑定模型选择变化时的滑块更新
            model_choice.change(
                fn=update_resolution_limit,
                inputs=model_choice,
                outputs=resolution
            )
        # ===== 修改结束 =====

        with gr.Tabs():
            # 单张图片处理标签页
            with gr.Tab("🖼️ 单张图片处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            label="上传图片",
                            type="pil",
                            height=400
                        )
                        
                        background_image = gr.Image(
                            label="背景图片（可选，默认透明背景）",
                            type="pil",
                            height=200
                        )
                        ##半透明切换按钮###
                        # —— 半透明扣除：开关/滑块/模式 + 折叠说明（复用一套） ——
                        semi_enable_img, semi_strength_img, semi_mode_img = build_semi_controls()

                        # 去白边开关（自动消除 1–2 px 白色毛边）
                        defringe_img = gr.Checkbox(
                            label="去白边（自动消除 1–2 px 白色毛边）",
                            value=False,
                            info="轻微收缩边缘并回灌前景色，减少白色毛边。"
                        )
                        # —— 去白边力度滑杆（默认隐藏；勾选后显示）——
                        with gr.Group(visible=False) as defringe_opts_img:
                            defringe_strength_img = gr.Slider(
                                label="去白边力度",
                                minimum=0.0, maximum=1.0, step=0.05, value=0.7,
                                info="推荐：人像 0.6–0.85；白底可到 0.9–1.0（更强收边）。高分辨率下会自适应放大侵蚀核。"
                            )

                        # 勾选联动：显示/隐藏力度滑杆
                        defringe_img.change(
                            fn=lambda on: gr.update(visible=on),
                            inputs=defringe_img,
                            outputs=defringe_opts_img
                        )
#############################绘画涂抹#####################################
                        # === ROI 画板 UI（新版） ===
                        roi_enable = gr.Checkbox(
                            label="🎯 指定区域（在进入模型前裁剪并对齐回原图）",
                            value=False,
                            info="开启后只对你圈定/涂抹的区域做抠图，其他区域保持背景"
                        )

                        with gr.Group(visible=False) as roi_group:
                            # 默认收起的高级选项
                            with gr.Accordion("高级选项", open=False):
                                with gr.Row():
                                    roi_thumb_side = gr.Slider(
                                        label="缩略图长边 (px)",
                                        minimum=256, maximum=1200, step=64, value=640,
                                        info="只影响画板显示与交互，不影响最终分辨率"
                                    )
                                    roi_pad_px = gr.Slider(
                                        label="ROI 外扩 padding (px)",
                                        minimum=0, maximum=128, step=2, value=16,
                                        info="先裁剪再分割时的安全边，越大越保守、速度稍慢"
                                    )
                                    roi_crop_before = gr.Checkbox(
                                        label="在模型前裁剪（更快/更准）",
                                        value=True
                                    )

                            # 半透明画笔（默认 45% 不透明度），颜色固定为一组半透明色
                            roi_canvas = gr.ImageEditor(
                                label="在缩略图上用画笔涂抹 ROI（半透明预览，不影响结果）",
                                type="numpy", image_mode="RGBA", height=380, sources=None, layers=True,
                                brush=gr.Brush(
                                    default_size="auto",
                                    colors=["#ff9800", "#1e88e5", "#43a047", "#e53935", "#ffffff"],
                                    default_color="#ff9800",
                                    color_mode="fixed"
                                ),
                            )

                            with gr.Row():
                                roi_clear = gr.Button("清空涂抹", variant="secondary")
                                roi_tips = gr.Markdown(
                                    "提示：选择画笔后在图上**半透明**涂抹要保留的前景区域；无需涂满，适当涂抹 + padding 即可。"
                                )

                        roi_meta_state = gr.State(value=None)   # 记录缩略图/原图尺寸

                        ####
                        # === 工具：初始化画板（返回 numpy RGBA 背景，匹配 type="numpy"） ===
                        def _init_roi_editor(img: Image.Image | None, long_side: int, overlay_color=(255, 152, 0), overlay_alpha=0.45):
                            if img is None:
                                return gr.update(), None
                            ev, meta = _make_editor_thumbnail(img, int(long_side))
                            thumb = ev["background"].convert("RGBA") if hasattr(ev["background"], "convert") else ev["background"]
                            bg_np = np.array(thumb, dtype=np.uint8)

                            # 生成半透明预览（此时还没图层，先把 composite = 背景）
                            editor_value = {"background": bg_np, "layers": [], "composite": bg_np}
                            return editor_value, meta

                        # 清空：仅清图层，保留背景，避免变成白底看不到原图
                        def _clear_roi_layers(editor_value):
                            bg = editor_value.get("background") if isinstance(editor_value, dict) else None
                            return {"background": bg, "layers": [], "composite": bg}

                        # 开关勾选 → 自动显示/隐藏 + 自动初始化画板（相当于“默认点击启动”）
                        def _on_roi_toggle(enabled, img, long_side):
                            if enabled and img is not None:
                                ev, meta = _init_roi_editor(img, int(long_side))
                                return gr.update(visible=True), ev, meta
                            else:
                                # 关掉时隐藏并清空
                                return gr.update(visible=False), None, None

                        roi_enable.change(
                            _on_roi_toggle,
                            inputs=[roi_enable, input_image, roi_thumb_side],
                            outputs=[roi_group, roi_canvas, roi_meta_state],
                            show_progress=False
                        )

                        # 改缩略图长边 → 自动刷新（仅在已启用时）
                        def _maybe_refresh_editor(enabled, img, long_side):
                            if not enabled or img is None:
                                return gr.update(), None
                            return _init_roi_editor(img, int(long_side))

                        roi_thumb_side.change(
                            _maybe_refresh_editor,
                            inputs=[roi_enable, input_image, roi_thumb_side],
                            outputs=[roi_canvas, roi_meta_state],
                            show_progress=False
                        )

                        # 更换输入图 → 自动刷新（仅在已启用时）
                        input_image.change(
                            _maybe_refresh_editor,
                            inputs=[roi_enable, input_image, roi_thumb_side],
                            outputs=[roi_canvas, roi_meta_state],
                            show_progress=False
                        )

                        # 清空涂抹（保留背景）
                        roi_clear.click(_clear_roi_layers, inputs=[roi_canvas], outputs=[roi_canvas])

#################################绘画涂抹########################################
                        process_btn = gr.Button(
                            "🚀 开始处理",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            label="处理结果",
                            height=400,
                            format="png"  # 强制使用 PNG 格式
                        )
                        
                        mask_preview = gr.Image(
                            label="遮罩预览",
                            height=200,
                            format="png"  # 强制使用 PNG 格式
                        )
                        
                        status_text = gr.Textbox(
                            label="处理状态",
                            interactive=False
                        )
                
                # 绑定处理函数
                process_btn.click(
                    fn=process_image_with_settings,
                    inputs=[input_image, background_image,
                            semi_enable_img, semi_strength_img, semi_mode_img,
                            defringe_img, defringe_strength_img,
                            roi_enable, roi_canvas, roi_meta_state, roi_crop_before, roi_pad_px,resolution
                            ],
                    outputs=[output_image, mask_preview]
                )

            # 批量图片处理标签页
            with gr.Tab("📁 批量图片处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_images = gr.File(
                            label="上传多张图片",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        
                        batch_bg_image = gr.Image(
                            label="背景图片（可选，默认绿色背景）",
                            type="pil",
                            height=200
                        )
                        # ===== 批量图片处理 Tab =====
                        semi_enable_bi, semi_strength_bi, semi_mode_bi = build_semi_controls()

                        defringe_bi = gr.Checkbox(
                            label="去白边（自动）",
                            value=False,
                            info="批量图片去白边。"
                        )
                        with gr.Group(visible=False) as defringe_opts_bi:
                            defringe_strength_bi = gr.Slider(
                                label="去白边力度（批量）",
                                minimum=0.0, maximum=1.0, step=0.05, value=0.65,
                                info="推荐：0.55–0.8 兼顾速度与质量；>0.9 为激进模式（更强收边）。高分辨率自适应放大。"
                            )

                        defringe_bi.change(
                            fn=lambda on: gr.update(visible=on),
                            inputs=defringe_bi,
                            outputs=defringe_opts_bi
                        )
                        batch_process_btn = gr.Button(
                            "🚀 批量处理",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        batch_output = gr.File(
                            label="下载处理结果（ZIP文件）"
                        )
                        
                        batch_status = gr.Textbox(
                            label="处理状态",
                            interactive=False
                        )
                
                # 绑定批量处理函数
                batch_process_btn.click(
                    fn=process_batch_images,
                    inputs=[batch_images, batch_bg_image,
                            semi_enable_bi, semi_strength_bi, semi_mode_bi,
                            defringe_bi, defringe_strength_bi],
                    outputs=[batch_output, batch_status]
                )

            # 单个视频处理标签页
            with gr.Tab("🎬 单个视频处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_video = gr.Video(
                            label="上传视频",
                            height=300
                        )
                        
                        video_bg_image = gr.Image(
                            label="背景图片（可选，默认绿色背景）",
                            type="pil",
                            height=200
                        )
                        video_bg_color = gr.ColorPicker(
                            label="背景颜色（未上传图片时生效）",
                            value="#00FF00"
                        )
                        # ===== 单个视频处理 Tab =====

                        semi_enable_v, semi_strength_v, semi_mode_v = build_semi_controls()

                        video_process_btn = gr.Button(
                            "🚀 开始处理",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        output_video = gr.Video(
                            label="处理结果",
                            height=300
                        )
                        
                        video_status = gr.Textbox(
                            label="处理状态",
                            interactive=False
                        )
                
                # 绑定视频处理函数（✅ 多传两个新参数）
                video_process_btn.click(
                    fn=process_video,
                    inputs=[input_video, video_bg_image, video_bg_color,
                            semi_enable_v, semi_strength_v, semi_mode_v],
                    outputs=[output_video, video_status]
                )

            # 批量视频处理标签页
            with gr.Tab("📹 批量视频处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_videos = gr.File(
                            label="上传多个视频",
                            file_count="multiple",
                            file_types=["video"]
                        )
                        
                        batch_video_bg_image = gr.Image(
                            label="背景图片（可选，默认绿色背景）",
                            type="pil",
                            height=200
                        )

                        batch_video_bg_color = gr.ColorPicker(
                            label="背景颜色（未上传图片时生效）",
                            value="#00FF00"
                        )
                        # ===== 批量视频处理 Tab =====

                        semi_enable_bv, semi_strength_bv, semi_mode_bv = build_semi_controls()

                        batch_video_process_btn = gr.Button(
                            "🚀 批量处理",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        batch_video_output = gr.File(
                            label="下载处理结果（ZIP文件）"
                        )
                        
                        batch_video_status = gr.Textbox(
                            label="处理状态",
                            interactive=False
                        )
                
                # 绑定批量视频处理函数
                batch_video_process_btn.click(
                    fn=process_batch_videos,
                    inputs=[batch_videos, batch_video_bg_image, resolution, batch_video_bg_color,
                            semi_enable_bv, semi_strength_bv, semi_mode_bv],
                    outputs=[batch_video_output, batch_video_status]
                )
            # （已移除：模型训练标签页）
            # （已移除：配置调整标签页）
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown(
                """
                ### 🔧 功能说明
                
                1. **单张图片处理**：上传一张图片，可选择背景图片，系统自动移除背景
                2. **批量图片处理**：同时上传多张图片进行批量处理，结果打包为ZIP文件
                3. **视频处理**：支持单个和批量视频处理，逐帧移除背景
                4. **背景选择**：可上传自定义背景图片，或使用默认绿色背景
                
                
                
                ### ⚡ 性能优化
                
                - 使用GPU加速推理（如果可用）
                - 支持半精度计算提升速度
                - 批量处理自动优化内存使用
                
                ### 📝 注意事项
                
                - 支持常见图片格式：JPG, PNG, WEBP等
                - 支持常见视频格式：MP4, AVI, MOV等
                - 视频处理需要较长时间，请耐心等待
                - 批量处理结果会自动打包为ZIP文件供下载
                - 训练功能需要准备好的数据集
                - 配置修改会自动创建备份文件
                """
            )
        with gr.Accordion("📂 打开缓存与结果目录", open=False):
            gr.Markdown(
                "你可以打开或清理缓存与输出文件夹。"
                "\n💡 建议使用“安全清理”保留离线模型，避免断网后无法加载模型。"
            )

            # === 打开目录按钮 ===
            open_preds = gr.Button("🖼️ 打开抠图结果目录 (preds-BiRefNet)")
            open_weights = gr.Button("🧱 打开离线模型目录 (models_local)")
            output_text = gr.Textbox(label="操作结果", interactive=False)

            def open_folder(path):
                import subprocess, platform, os
                abs_path = os.path.abspath(path)
                os.makedirs(abs_path, exist_ok=True)
                try:
                    if platform.system() == "Windows":
                        subprocess.Popen(f'explorer "{abs_path}"')
                    elif platform.system() == "Darwin":
                        subprocess.Popen(["open", abs_path])
                    else:
                        subprocess.Popen(["xdg-open", abs_path])
                    return f"📂 已打开：{abs_path}"
                except Exception as e:
                    return f"⚠️ 无法打开目录：{e}"

            open_preds.click(fn=lambda: open_folder("preds-BiRefNet"), outputs=[output_text])
            open_weights.click(fn=lambda: open_folder("models_local"), outputs=[output_text])

            # === 清理缓存按钮 ===
            gr.Markdown("### 🧹 缓存清理选项")

            clear_safe_btn = gr.Button("🧹 安全清理 (保留离线模型)", variant="secondary")
            clear_full_btn = gr.Button("🔥 完全清理 (包含模型缓存)", variant="stop")

            def clear_cache_safe():
                """安全清理：保留离线模型，仅删除缓存和结果"""
                import shutil, os
                cleared = []

                # 1️⃣ 清理推理结果和临时缓存
                for path in ["weights", "preds-BiRefNet", "__pycache__"]:
                    if os.path.exists(path):
                        try:
                            shutil.rmtree(path)
                            cleared.append(path)
                        except Exception as e:
                            print(f"⚠️ 删除失败 {path}: {e}")

                # 2️⃣ 清理 HuggingFace 缓存目录但保留离线模型
                models_local = "models_local"
                if os.path.exists(models_local):
                    subdirs = os.listdir(models_local)
                    deletable = []
                    for d in subdirs:
                        full_path = os.path.join(models_local, d)
                        # 删除 huggingface 缓存目录（models-- 开头）
                        if d.startswith("models--"):
                            deletable.append(full_path)
                    for path in deletable:
                        try:
                            shutil.rmtree(path)
                            cleared.append(path)
                        except Exception as e:
                            print(f"⚠️ 删除失败 {path}: {e}")

                if cleared:
                    return "✅ 已清理以下目录（保留离线模型）:\n" + "\n".join(cleared)
                else:
                    return "ℹ️ 未发现可清理缓存。"

            def clear_cache_full():
                """完全清理：包括模型缓存"""
                import shutil, os
                cleared = []
                for path in ["weights", "preds-BiRefNet", "models_local", "__pycache__"]:
                    if os.path.exists(path):
                        try:
                            shutil.rmtree(path)
                            cleared.append(path)
                        except Exception as e:
                            print(f"⚠️ 删除失败 {path}: {e}")
                if cleared:
                    return "🧨 已彻底清理以下目录（模型缓存已删除）:\n" + "\n".join(cleared)
                else:
                    return "ℹ️ 未发现可清理缓存。"

            clear_safe_btn.click(fn=clear_cache_safe, outputs=[output_text])
            clear_full_btn.click(fn=clear_cache_full, outputs=[output_text])

    return interface

# （已移除：训练相关函数）
import webbrowser
import threading
import time

def update_available_models():
    """自动获取 HuggingFace 上 zhengpeng7 的最新模型列表"""
    import requests
    try:
        url = "https://huggingface.co/api/models?author=ZhengPeng7"
        resp = requests.get(url, timeout=5)
        repos = [m["modelId"].split("/")[-1] for m in resp.json()]
        return sorted(repos)
    except Exception as e:
        print(f"⚠️ 无法获取模型列表: {e}")
        return list(usage_to_weights_file.values())

import time, webbrowser

import time

def run_app():
    """非阻塞启动 + 自动打开浏览器（Gradio 自处理端口与打开）"""
    normalize_model_structure("models_local")
    print("正在启动图片/视频背景更换工具...")
    print("⚙️ 等待用户选择模型后再加载...")

    demo = create_interface()

    # Gradio 5.4：queue() 不再接收并发等参数
    demo.queue()   # 或者直接删掉这行，保留也没问题

    demo.launch(
        server_name="0.0.0.0",
        server_port=None,        # 让系统挑端口
        share=False,
        show_error=True,
        quiet=False,
        inbrowser=True,          # ✅ 自动打开浏览器（真实端口）
        prevent_thread_lock=True # ✅ 非阻塞
    )

    # 阻塞主线程，等同你以前的 thread.join()
    demo.block_thread()

def normalize_model_structure(base_dir="models_local"):
    """
    🔧 自动迁移旧结构 -> HuggingFace 标准缓存结构
    例如: models_local/BiRefNet_lite-2K → models_local/models--zhengpeng7--BiRefNet_lite-2K
    """
    import shutil
    if not os.path.exists(base_dir):
        return
    for d in os.listdir(base_dir):
        full_path = os.path.join(base_dir, d)
        if os.path.isdir(full_path) and not d.startswith("models--"):
            target = os.path.join(base_dir, f"models--zhengpeng7--{d}")
            if not os.path.exists(target):
                print(f"🔧 迁移旧结构 {d} → {target}")
                try:
                    shutil.move(full_path, target)
                except Exception as e:
                    print(f"⚠️ 迁移失败: {e}")

if __name__ == "__main__":
    run_app()

# （已移除：配置管理相关函数）

