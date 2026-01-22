import os
import torch
import requests
import traceback
try:
    from transformers import AutoModelForImageSegmentation
    from torchvision import transforms
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers未安装，模型加载功能将不可用")

from PIL import Image

from src.utils.logger import logger
from src.config import USAGE_TO_WEIGHTS_FILE, MODELS_LOCAL_DIR


class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = None
        self.current_loaded_model_name = None
        self.current_loaded_resolution = None
        
        # 统一模型存储目录（绝对路径）
        self.models_dir = os.path.abspath(os.path.join(os.getcwd(), MODELS_LOCAL_DIR))
        os.makedirs(self.models_dir, exist_ok=True)
        
        self._initialized = True

    def _get_repo_name(self, model_name):
        """获取模型对应的仓库名称"""
        if model_name in USAGE_TO_WEIGHTS_FILE:
            return USAGE_TO_WEIGHTS_FILE[model_name]
        return model_name

    def _get_model_path(self, model_name):
        """获取模型本地存储路径"""
        repo_name = self._get_repo_name(model_name)
        return os.path.join(self.models_dir, repo_name)

    def check_model_exists(self, model_name):
        """
        检测本地模型是否存在且完整
        
        Returns:
            "ready" - 模型存在且完整，可直接加载
            "incomplete" - 目录存在但文件不完整（缺少代码文件）
            "not_found" - 模型不存在
        """
        model_path = self._get_model_path(model_name)
        
        if not os.path.exists(model_path):
            return "not_found"
        
        # 检查必要文件
        config_file = os.path.join(model_path, "config.json")
        safetensors_file = os.path.join(model_path, "model.safetensors")
        bin_file = os.path.join(model_path, "pytorch_model.bin")
        # BiRefNet 自定义模型需要代码文件
        code_file = os.path.join(model_path, "birefnet.py")
        
        has_config = os.path.exists(config_file)
        has_weights = os.path.exists(safetensors_file) or os.path.exists(bin_file)
        has_code = os.path.exists(code_file)
        
        if has_config and has_weights and has_code:
            return "ready"
        elif has_config or has_weights:
            return "incomplete"  # 有部分文件但不完整
        else:
            return "not_found"

    def get_all_model_status(self):
        """
        获取所有模型的状态
        
        Returns:
            dict: {model_name: status} 
            status: "ready" / "incomplete" / "not_found"
        """
        status_map = {}
        for model_name in USAGE_TO_WEIGHTS_FILE.keys():
            status_map[model_name] = self.check_model_exists(model_name)
        return status_map

    def check_hf_access(self):
        """检测 HuggingFace 网络是否可访问"""
        try:
            # 增加超时时间到15秒，适应VPN连接延迟
            r = requests.get("https://huggingface.co", timeout=15)
            return r.status_code == 200
        except Exception:
            return False

    def _load_from_local(self, model_path):
        """从本地路径加载模型"""
        print(f"📂 正在从本地加载模型：{model_path}")
        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            self._setup_model()
            return True
        except Exception as e:
            print(f"⚠️ 本地加载失败: {e}")
            return False

    def _download_and_load(self, model_name):
        """从 HuggingFace 下载并加载模型"""
        repo_name = self._get_repo_name(model_name)
        hf_repo = f"zhengpeng7/{repo_name}"
        model_path = self._get_model_path(model_name)
        
        print(f"🌐 正在从 HuggingFace 下载模型：{hf_repo}")
        print(f"📦 下载目标目录：{model_path}")
        
        # 使用 local_dir 参数直接下载到目标路径（避免冗余缓存）
        try:
            from huggingface_hub import snapshot_download
            
            # 先下载到目标目录
            snapshot_download(
                repo_id=hf_repo,
                local_dir=model_path,
                local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
                ignore_patterns=["*.md", "*.txt", ".gitattributes"]  # 忽略非必要文件
            )
            print(f"✅ 模型下载完成：{model_path}")
            
        except ImportError:
            # 如果没有 huggingface_hub，回退到 from_pretrained
            print("⚠️ huggingface_hub 未安装，使用 from_pretrained 下载...")
            self.model = AutoModelForImageSegmentation.from_pretrained(
                hf_repo,
                trust_remote_code=True,
                cache_dir=self.models_dir  # 临时使用，后续会自动保存
            )
            # 保存到目标路径
            self.model.save_pretrained(model_path)
            self._setup_model()
            return True
        
        # 从下载的本地路径加载
        return self._load_from_local(model_path)

    def _setup_model(self):
        """配置模型（设备、精度、评估模式）"""
        if torch.cuda.is_available():
            self.model.to(self.device)
            try:
                self.model.half()
                print("⚡ 已启用 FP16 半精度推理")
            except Exception:
                self.model.float()
                print("⚠️ FP16 转换失败，回退到 FP32")
        else:
            self.model.float()
        
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_name='General', input_size=(1024, 1024)):
        """
        加载 BiRefNet 模型（本地优先策略）
        
        加载逻辑：
        1. 检测本地模型是否存在且完整 → 直接加载
        2. 本地不存在 → 检测网络 → 下载后加载
        3. 本地不存在且无网络 → 返回失败
        """
        # --- 参数安全检查 ---
        if isinstance(model_name, (int, float)):
            raise TypeError(f"model_name 必须为字符串，收到: {model_name}({type(model_name)})")
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        elif not (isinstance(input_size, tuple) and len(input_size) == 2):
            raise ValueError(f"input_size 必须为 (H, W) tuple，收到: {input_size}")

        if not TRANSFORMERS_AVAILABLE:
            print("❌ transformers 库未安装，无法加载模型")
            return False

        # --- 缓存命中检测 ---
        if self.model is not None and self.current_loaded_model_name == model_name:
            self.current_loaded_resolution = input_size
            print(f"✅ 已加载模型：{model_name}（无需重复加载），当前分辨率参数 {input_size}")
            return True

        # 更新加载记录
        self.current_loaded_model_name = model_name
        self.current_loaded_resolution = input_size

        try:
            # === Step 1: 检测本地模型 ===
            local_status = self.check_model_exists(model_name)
            model_path = self._get_model_path(model_name)
            
            if local_status == "ready":
                # 本地模型完整，尝试加载（无需网络）
                print(f"📦 检测到本地模型：{model_path}")
                if self._load_from_local(model_path):
                    print(f"✅ 模型加载完成：{model_name}，输入尺寸 {input_size}")
                else:
                    # 本地加载失败，尝试重新下载
                    print(f"⚠️ 本地模型加载失败，尝试重新下载...")
                    if self.check_hf_access():
                        # 清理不完整的本地文件
                        import shutil
                        if os.path.exists(model_path):
                            shutil.rmtree(model_path)
                        self._download_and_load(model_name)
                        print(f"✅ 模型重新下载并加载完成：{model_name}")
                    else:
                        raise ConnectionError(
                            f"❌ 本地模型 {model_name} 加载失败，且无法访问 HuggingFace。\n"
                            "请检查网络连接（可能需要 VPN），或手动下载完整模型到：\n"
                            f"{model_path}"
                        )
                
            elif local_status == "incomplete":
                # 本地模型不完整
                print(f"⚠️ 本地模型不完整，需要重新下载：{model_path}")
                if self.check_hf_access():
                    # 清理不完整的本地文件
                    import shutil
                    if os.path.exists(model_path):
                        shutil.rmtree(model_path)
                    self._download_and_load(model_name)
                    print(f"✅ 模型下载并加载完成：{model_name}")
                else:
                    raise ConnectionError(
                        f"❌ 模型 {model_name} 本地文件不完整，且无法访问 HuggingFace。\n"
                        "请检查网络连接（可能需要 VPN），或手动下载模型到：\n"
                        f"{model_path}"
                    )
                    
            else:  # not_found
                # 本地不存在，需要下载
                print(f"📭 本地未找到模型 {model_name}，尝试联网下载...")
                if self.check_hf_access():
                    self._download_and_load(model_name)
                    print(f"✅ 模型下载并加载完成：{model_name}")
                else:
                    raise ConnectionError(
                        f"❌ 本地未找到模型 {model_name}，且无法访问 HuggingFace。\n"
                        "请检查网络连接（可能需要 VPN），或手动下载模型到：\n"
                        f"{model_path}"
                    )

            # === 更新 Transform ===
            def resize_keep_ratio(img, target_size):
                w, h = img.size
                scale = target_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                return img.resize((new_w, new_h), Image.BILINEAR)

            self.transform = transforms.Compose([
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

    def unload_model(self):
        """
        卸载当前模型，释放显存和内存
        """
        if self.model is not None:
            del self.model
            self.model = None
            self.current_loaded_model_name = None
            self.transform = None
            
            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 触发 Python 垃圾回收
            import gc
            gc.collect()
            
            print("✅ 模型已卸载，显存/内存已释放")
            return True
        else:
            print("ℹ️ 当前没有加载的模型")
            return False

    def get_downloaded_models(self):
        """
        获取已下载的模型列表
        
        Returns:
            list: 已下载模型名称列表 (使用友好名称如 'General' 而非 'BiRefNet')
        """
        downloaded = []
        for model_name in USAGE_TO_WEIGHTS_FILE.keys():
            if self.check_model_exists(model_name) == "ready":
                downloaded.append(model_name)
        return downloaded

    def delete_model(self, model_name):
        """
        删除指定的本地模型
        
        Args:
            model_name: 模型名称（如 'General', 'Matting' 等）
        
        Returns:
            bool: 删除是否成功
        """
        import shutil
        
        model_path = self._get_model_path(model_name)
        
        if not os.path.exists(model_path):
            print(f"⚠️ 模型 {model_name} 不存在")
            return False
        
        # 如果要删除的是当前加载的模型，先卸载
        if self.current_loaded_model_name == model_name:
            self.unload_model()
        
        try:
            shutil.rmtree(model_path)
            print(f"✅ 模型 {model_name} 已删除")
            return True
        except Exception as e:
            print(f"❌ 删除模型失败：{e}")
            return False


# Global instance
model_manager = ModelManager()
