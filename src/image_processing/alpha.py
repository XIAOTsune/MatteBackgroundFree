import numpy as np
import cv2
from PIL import Image

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

def _hair_protect_weight(rgb_u8: np.ndarray, m_u8: np.ndarray, band_u8: np.ndarray) -> np.ndarray:
    """
    计算发丝保护权重 w_hair ∈ [0,1]
    修复问题：原版偏向保护暗色物体，导致浅色皮肤/白衣边缘被侵蚀。
    新版策略：仅依赖 梯度(细节) + 距离(薄度)，去除亮度偏见。
    """
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

def compute_alpha_unified(
    image,
    model_mask: np.ndarray,
    strength: float,
    bg_color: tuple
) -> np.ndarray:
    """
    [Unified V4.1] 强化版色差估算算法
    使用 Linear Ramp (Tolerance + Range) 逻辑进行稳健抠图
    """
    # 1. 准备数据 (RGB Float [0, 255])
    I_f = image.astype(np.float32)
    bg_color_f = np.array(bg_color, dtype=np.float32)
    
    # 2. 计算色差 (欧氏距离 [0, ~441])
    diff = np.linalg.norm(I_f - bg_color_f, axis=2)
    
    # 3. 线性映射 logic: Alpha = (diff - Tol) / Range
    # strength 0.0 -> Tol=5,  Range=150 (极度保守，几乎不扣)
    # strength 0.5 -> Tol=40, Range=100 (标准，黑纱效果)
    # strength 1.0 -> Tol=120, Range=60  (强力，玻璃效果)
    
    tol = 5.0 + (strength * 115.0)    # [5.0, 120.0]
    range_val = 150.0 - (strength * 90.0) # [150.0, 60.0]
    
    # 计算初步 Alpha
    alpha_est = (diff - tol) / (range_val + 1e-6)
    alpha_est = np.clip(alpha_est, 0.0, 1.0)
    
    # 4. 非线性修正 (Gamma)
    # strength 越大，越希望半透明区域更“透”，即 Alpha 越小 -> 指数增加
    gamma = 1.0 + (strength * 1.5) # [1.0, 2.5]
    alpha_est = np.power(alpha_est, gamma)
    
    # 5. 融合模型 Mask (作为基准边界)
    # 如果模型认为这里完全是背景 (mask=0)，则强制为0
    # 如果模型认为这里完全是前景 (mask=1)，我们允许色差算法将其变透
    
    # 策略：Final = ModelMask * AlphaEst
    final_alpha = model_mask * alpha_est
    
    return (np.clip(final_alpha, 0.0, 1.0) * 255).astype(np.uint8)


# 保留为了兼容性，但内部逻辑可以复用或弃用
def estimate_soft_alpha_inside_mask(
    image_or_array,
    base_mask: np.ndarray | float,
    *,
    strength: float = 0.5,        # 0~1
    mode: str = "auto"            # Deprecated, kept for interface compatibility
) -> np.ndarray:
    """
    [Legacy V3.0] 保留此函数用于回退或兼容，但推荐使用 V4.0 unified。
    这里我们做一个简单的桥接，如果需要完全复用旧逻辑可保留原代码。
    为了不破坏已有逻辑，我们这里保留原 V3 代码不变。
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
    gamma_base = 1.0 + (strength * 2.5)
    
    if mode in ("暗部优先", "dark"):
        final_gamma = gamma_base * 0.8 
    else:
        final_gamma = gamma_base

    alpha_processed = np.power(alpha, final_gamma)

    # ---- 3. 核心逻辑：亮度加权 (Luma Masking) ----
    if mode in ("透色优先", "bleed", "light"):
        luma = 0.299 * I_f[:,:,0] + 0.587 * I_f[:,:,1] + 0.114 * I_f[:,:,2]
        luma_weight = luma * (0.8 * strength) + (1.0 - (0.8 * strength))
        alpha_processed = alpha_processed * luma_weight

    return (np.clip(alpha_processed, 0.0, 1.0) * 255).astype(np.uint8)

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

