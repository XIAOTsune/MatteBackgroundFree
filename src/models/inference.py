import os
import torch
import numpy as np
import cv2
import traceback
from PIL import Image
from torchvision import transforms

from src.utils.logger import logger
from .loader import model_manager
from src.image_processing.background import replace_background_with_mask

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

def segment_image(image, model_name='General', input_size=(1024, 1024)):
    """使用指定模型和分辨率进行图像分割"""
    
    # ---- 容错：防止 model_name 被误传为 int ----
    if isinstance(model_name, (int, float)):
        sz = int(model_name)
        input_size = (sz, sz)
        model_name = (
            model_manager.current_loaded_model_name
            if model_manager.current_loaded_model_name
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
    if (model_manager.model is None) or (model_manager.current_loaded_model_name != model_name):
        if not model_manager.load_model(model_name, input_size):
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
        input_tensor = transform_pipeline(resized_image).unsqueeze(0).to(model_manager.device)
        
        # 如果模型是 FP16，输入也要转 FP16
        if torch.cuda.is_available():
            input_tensor = input_tensor.half()

        with torch.no_grad():
            outputs = model_manager.model(input_tensor)
            pred = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            
            # === 修复核心 BUG ===
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

def process_single_frame(frame, background_image, model_name='General', input_size=(1024, 1024)):
    """处理单帧视频"""
    try:
        # Note: apply_background_replacement logic needs to be accessed here or duplicated.
        # Ideally, higher level logic calls segment_image directly or we import from image_processor (circular?)
        # For simple frame processing, we often just want segmentation + replace.
        # But this function was in app_gradio_new.py invoking apply_background_replacement.
        # For now, let's keep it simple and just do segmentation + quick replacement if needed, 
        # or we will define it in processors to avoid circular dependency if inference imports processors.
        # Actually inference shouldn't import processors. Processors depend on inference.
        pass 
    except Exception as e:
        logger.error(f"处理帧失败: {e}")
        return frame
