import numpy as np
import cv2
from PIL import Image
from src.utils.logger import logger
from .utils import hex_to_rgb
from .post_process import _map_defringe_strength, _color_decontam_edge, _scale_px_for_image, _hair_protect_weight, _remove_white_halo_rgba

def estimate_background_color(image_array: np.ndarray, mask: np.ndarray):
    """
    估算背景颜色（用于色差半透明抠图）
    
    Args:
        image_array: RGB图像 (H,W,3)
        mask: 2D掩码 (H,W)，0为背景，255为前景
    
    Returns:
        (r, g, b, is_valid): 背景RGB均值和是否有效
    """
    try:
        # 1. 确定背景区域 (根据Mask确定背景)
        # 放宽阈值 (10 -> 40)，以便在模型 Mask 稍微偏大时也能取到背景
        bg_mask = (mask < 40).astype(np.uint8)
        
        # 如果背景区域太小 (<2%)，也没法估算，视为全屏物体
        if bg_mask.mean() < 0.02:
            return (0, 0, 0, False)
        
        # 腐蚀背景掩码，深处取样，绝对避免物体边缘干扰
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bg_mask_safe = cv2.erode(bg_mask, ker, iterations=3)
        
        if bg_mask_safe.sum() < 100: # 像素太少，回退到未腐蚀
            bg_mask_safe = bg_mask
            
        # 2. 采样颜色
        # 我们使用中位数 (Median) 来抵抗噪点和异常值
        # 注意：OpenCV 的 medianBlur 太慢，我们直接取像素点求中位数
        pixels = image_array[bg_mask_safe > 0]
        
        # 为了速度，如果像素太多，随机采样 10000 个点
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
            
        median_color = np.median(pixels, axis=0)
        return (int(median_color[0]), int(median_color[1]), int(median_color[2]), True)
        
    except Exception as e:
        logger.error(f"背景颜色估算失败: {e}")
        return (0, 0, 0, False)

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

def replace_background_with_mask(
    image_array,
    background_array,
    mask,
    remove_white_halo: bool = False,
    defringe_strength: float | None = None,
    is_transparent_mode: bool = False,
    *,
    band_px: int = 2,
    strength: float = 0.7,
    erode_px: int = 1
):
    """
    [修复版] 将前景按 mask 融合到背景
    增加了：Mask闭运算预处理（防空洞）、限制侵蚀核大小。
    """
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

    # 修复核心 1: 预先闭运算 (填补 Mask 内部的微小孔洞)
    # [BUG FIX]: 在半透明模式下，必须禁用闭运算，否则抠出来的“透明孔洞”会被重新填满！
    if is_transparent_mode:
        # 半透明模式下不进行闭运算，保留所有算法扣出的细节
        pass 
    else:
        # 普通硬边模式，进行轻微闭运算清理
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
