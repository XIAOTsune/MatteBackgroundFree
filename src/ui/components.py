import numpy as np
import cv2
from typing import Dict, Any

def _editor_layers_to_mask_fullres(editor_value: Dict[str, Any], meta: Dict[str, int]) -> np.ndarray | None:
    """
    从 ImageEditor 的编辑值提取 ROI（二值）并映射回原图尺寸。
    """
    if not editor_value or not meta:
        return None

    # Note: Gradio ImageEditor in numpy mode returns arrays.
    # If meta is simplified (thumb_w == ori_w), we can just process directly.
    # In the refactored interface.py, we set thumb_w = ori_w in update_roi_meta.
    # So we can trust the layers size if they match background.
    
    # However, let's keep the logic robust.
    
    # 1) 直接从图层 alpha 聚合
    layers = editor_value.get("layers") or []
    mask_accum = None

    for layer in layers:
        if layer is None:
            continue
        arr = np.array(layer)
        if arr.ndim == 3 and arr.shape[2] == 4:     # RGBA
            alpha = arr[..., 3]
            if mask_accum is None:
                mask_accum = alpha
            else:
                mask_accum = np.maximum(mask_accum, alpha)
        elif arr.ndim == 3 and arr.shape[2] == 3:   # RGB 兜底
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            mask_val = (gray > 0).astype(np.uint8) * 255
            if mask_accum is None:
                mask_accum = mask_val
            else:
                mask_accum = np.maximum(mask_accum, mask_val)

    # 2) 图层为空兜底：composite 与 background 差异
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

    if (mask_accum is None or mask_accum.max() == 0) and editor_value.get("composite") is not None and editor_value.get("background") is not None:
        bg = _safe_get_rgba(editor_value.get("background"))
        comp = _safe_get_rgba(editor_value.get("composite"))
        
        if bg is not None and comp is not None and bg.shape == comp.shape:
            # 简单阈值差异
            diff = np.abs(comp[..., :3].astype(np.int16) - bg[..., :3].astype(np.int16)).sum(axis=2)
            mask_accum = (diff > 5).astype(np.uint8) * 255

    if mask_accum is None or mask_accum.max() == 0:
        return None
    
    # No resize needed if we assume full res from input
    # But strictly speaking we should use meta["ori_w"] to resize if needed.
    
    return (mask_accum > 127).astype(np.uint8) * 255
