import numpy as np
import cv2
from .alpha import _hair_protect_weight

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


def _bleed_foreground_color(rgb_roi: np.ndarray, mask_roi: np.ndarray, iterations: int = 3) -> np.ndarray:
    """
    [Option A] 局部颜色扩散 (Local Color Bleeding)
    通过迭代模糊并归一化的方式，将前景颜色“扩张”到边缘和背景区域。
    这比估算背景色更鲁棒，因为它是从内部向外“借”颜色。
    """
    f = rgb_roi.astype(np.float32)
    # 取 Alpha > 200 的区域作为可靠前景中心
    m = (mask_roi > 200).astype(np.float32) / 255.0
    
    # 第一次：只保留可靠前景的颜色
    curr_f = f * m[..., None]
    curr_w = m
    
    # 迭代扩散：模糊颜色，模糊权重，然后相除，使颜色向外溢出
    for _ in range(iterations):
        # 这里的核大小随迭代可调，或者固定一个小核
        curr_f = cv2.GaussianBlur(curr_f, (7, 7), 0)
        curr_w = cv2.GaussianBlur(curr_w, (7, 7), 0)
        # 归一化，得到扩散后的纯净前景色彩层
        valid = curr_w > 1e-4
        curr_f[valid] /= curr_w[valid][..., None]
        curr_w[valid] = 1.0 # 重置权重为 1 方便下一轮
        
    return np.clip(curr_f, 0, 255).astype(np.uint8)

def _color_decontam_edge(rgb_u8: np.ndarray, mask: np.ndarray, band_px: int = 2, strength: float = 0.7):
    """
    仅在边带ROI做颜色去污染；采用 Option A 颜色扩散方案。
    """
    H, W = rgb_u8.shape[:2]
    rgb = rgb_u8 if rgb_u8.dtype == np.uint8 else np.clip(rgb_u8, 0, 255).astype(np.uint8)

    m = mask
    if m.dtype != np.uint8:
        m = (np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8)

    # 1. 确定边带 (Band)
    band_px = int(max(1, band_px))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    fg = (m > 0).astype(np.uint8)
    dil = cv2.dilate(fg, ker, iterations=1)
    ero = cv2.erode(fg, ker, iterations=1)
    band = cv2.subtract(dil, ero)

    ys, xs = np.where(band > 0)
    if ys.size == 0:
        return rgb

    # 2. 局部 ROI 裁剪 (提速)
    pad = 12 # 扩散需要更大的采样区
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    
    rgb_roi = rgb[y0:y1, x0:x1]
    m_roi = m[y0:y1, x0:x1]
    band_roi = band[y0:y1, x0:x1]

    # 3. [核心改变] 颜色扩散生成纯净前景层
    # 迭代次数根据强度调整，强度越大，扩散越远
    iter_cnt = 2 + int(strength * 3) 
    F_roi = _bleed_foreground_color(rgb_roi, m_roi, iterations=iter_cnt)

    # 4. 合成：***关键点*** 仅在边带区域进行颜色中和
    # 以前的版本是全局混合，导致内部模糊。现在我们通过 band_roi 作为遮罩，
    # 确保物体内部（band=0）的像素 100% 保留原始高频细节。
    w_hair = _hair_protect_weight(rgb_roi, m_roi, band_roi) 
    # 发丝保护：在细节丰富的地方稍微保留一点原色
    final_S = float(strength) * (1.0 - 0.5 * w_hair)
    
    # 将 band_roi 转为 float 遮罩，并做极轻微模糊（2px）以平滑过渡
    mix_mask = (band_roi.astype(np.float32) / 255.0)
    mix_mask = cv2.GaussianBlur(mix_mask, (3, 3), 0)
    
    # 最终混合因子 = 强度 * 边带遮罩
    blend_factor = final_S[..., None] * mix_mask[..., None]
    
    out_roi = (1.0 - blend_factor) * (rgb_roi.astype(np.float32)) + blend_factor * F_roi.astype(np.float32)
    out_roi = np.clip(out_roi + 0.5, 0, 255).astype(np.uint8)

    out = rgb.copy()
    out[y0:y1, x0:x1] = out_roi
    return out


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
    assert rgba.ndim == 3 and rgba.shape[2] >= 4, "expect HxWx4 RGBA"
    H, W = rgba.shape[:2]

    def _scale_px(px: int, base: int = 1024, cap: int = 10) -> int:
        if px <= 0: return 0
        scale = max(H, W) / float(base)
        return max(1, min(cap, int(round(px * max(1.0, scale)))))

    if mask.dtype != np.uint8:
        m = (np.clip(mask, 0, 1) * 255.0 + 0.5).astype(np.uint8)
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

        fg = (m > 0).astype(np.uint8)
        dil = cv2.dilate(fg, ker_b, iterations=1)
        ero = cv2.erode(fg, ker_b,  iterations=1)
        band = cv2.subtract(dil, ero)

        m_er = cv2.erode(m, ker_e, iterations=1)

        w_hair = _hair_protect_weight(rgb, m, band)
        
        # 增加保护力度
        erode_eff = (1.0 - 0.9 * w_hair).astype(np.float32)
        m_blend = (erode_eff * m_er.astype(np.float32) + (1.0 - erode_eff) * m.astype(np.float32))
        a_u8 = np.where(band > 0, m_blend, m).astype(np.uint8)
    else:
        a_u8 = a_u8_base.copy()

    # 轻羽化
    a_out = cv2.GaussianBlur(a_u8, (0, 0), sigmaX=0.6, sigmaY=0.6)

    out = np.dstack([rgb_fixed, a_out]).astype(np.uint8)
    return out

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
