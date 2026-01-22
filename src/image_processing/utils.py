import os
import gradio as gr
from PIL import Image
from src.utils.logger import logger

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

def _force_png_path(path: str) -> str:
    """把任意路径的扩展名改成 .png"""
    root, _ = os.path.splitext(path)
    return root + ".png"

def _save_image_safe(img, save_path: str):
    """
    安全保存图像：
    - RGBA/LA 或包含 transparency → 强制 PNG
    - JPEG 目标但图像非 RGB → 转 RGB 再存
    """
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
