# MatteBackgroundFree (小T的抠图工具箱)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Gradio](https://img.shields.io/badge/Gradio-UI-green)
![License](https://img.shields.io/badge/License-MIT-blue)

**MatteBackgroundFree** 是一款基于 SOTA 模型 **BiRefNet** 开发的高精度 AI 抠图工具。它不仅支持超高分辨率图像的快速背景移除，还集成了专业的**发丝级精修**、**半透明物体处理**（如婚纱、玻璃）以及**视频抠图**功能。

本项目旨在提供一个“开箱即用”且具备专业后期处理能力的本地化抠图解决方案。

---

## ✨ 核心功能 (Features)

### 1. 多模型支持与自动切换
内置支持多种 **BiRefNet** 变体，满足不同场景需求：
- **General (通用版)**: 平衡精度与速度，适合大多数日常图片。
- **General-Lite (轻量版)**: 极速推理，适合低配电脑或批量处理。
- **Portrait (人像版)**: 针对人像优化的权重，发丝细节更精准。
- **Matting (抠图专用)**: 专注于处理复杂的半透明边缘。
- **DIS / COD / HRSOD**: 针对特定学术数据集（如伪装目标检测）的某种变体。

> **智能加载**: 首次使用时会自动从 HuggingFace 下载模型。支持断网后的**离线模式**（模型自动缓存在 `models_local` 目录）。

### 2. 专业的边缘精修 (Advanced Matting)
不同于简单的二值分割，本工具引入了类似 Photoshop 的通道抠图与蒙版优化算法：
- **发丝保护**: 有效防止细微发丝被误删。
- **透明度恢复**: 针对婚纱、烟雾、玻璃杯等半透明物体，支持“透色优先”模式，保留真实的透明质感。
- **边缘去黑/去白**: 自动修复抠图边缘的杂色溢出。

### 3. 高清支持
- 无论原图分辨率多大（2K/4K/8K），工具都会自动进行分块或缩放处理，并在输出时还原到**原始分辨率**，保证细节不丢失。

### 4. 视频抠图 (Video Matting)
- 支持导入 `.mp4`, `.avi` 等常见视频格式。
- 逐帧处理并合成透明通道视频，或替换绿幕背景。

---

## 🛠️ 环境安装 (Installation)

建议使用 Anaconda 创建独立的 Python 环境，以避免依赖冲突。

### 1. 准备环境
确保安装了 Python 3.10 或更高版本。
```bash
conda create -n matting python=3.10
conda activate matting
```

### 2. 安装 CUDA (推荐)
为了获得最佳速度，请确保安装了支持 GPU 加速的 PyTorch。
访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取适合你显卡的安装命令。例如（CUDA 11.8）：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
*如果不使用 GPU，也可以安装 CPU 版本，但处理高清图会较慢。*

### 3. 安装依赖
克隆本项目后，在项目根目录运行：
```bash
pip install -r requirements.txt
```

**依赖列表 (`requirements.txt`) 重点：**
- `gradio`: Web UI 界面
- `opencv-python`: 图像处理
- `timm`, `kornia`: 模型依赖库
- `moviepy`: 视频处理支持

---

## 🚀 快速开始 (无脑运行指南)

### 方法一：Windows 一键脚本 (推荐小白使用) ✅

我们在项目根目录提供了自动化的脚本，只需两步即可运行。

**第一步：检查/安装 Python**
您的电脑需要安装 Python 3.10 或更高版本。
- 打开命令提示符输入 `python --version` 检查。
- 如果没有，请去 [Python 官网](https://www.python.org/downloads/) 下载安装（**注意：安装时务必勾选 "Add Python to PATH"**）。

**第二步：一键运行**
1. 双击运行 **`一键安装环境.bat`** (或者 `install.bat`)。
   - 它会自动创建环境并配置国内镜像源加速下载依赖。
   - 等待出现 "安装完成" 提示。
   
2. 双击运行 **`一键启动程序.bat`** (或者 `run.bat`)。
   - 稍等片刻，浏览器会自动弹出一个网页，即可开始抠图！

---

### 方法二：极客/开发者模式 (命令行)

如果您熟悉命令行，或者使用 Linux/Mac 系统，请使用标准方式：

1. **创建环境 (建议)**
   ```bash
   conda create -n matting python=3.10
   conda activate matting
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **启动程序**
   ```bash
   python app_gradio_new.py
   ```

---

## 📂 目录结构说明

```
MatteBackgroundFree/
├── app_gradio_new.py    # 主程序入口 (Gradio UI)
├── requirements.txt     # 依赖列表
├── models_local/        # [自动生成] 模型权重下载目录 (支持离线加载)
├── inputs/              # 示例输入文件夹
├── preds-BiRefNet/      # 默认输出保存文件夹
└── src/                 # 核心处理逻辑源码
```

## 🧩 高级用法 (Advanced Usage)

### 模型手动管理
如果你的网络无法连接 HuggingFace，可以手动下载模型文件：
1. 下载模型文件（通常包含 `config.json` 和 `model.safetensors`/`.pth`）。
2. 将其放入 `models_local/BiRefNet` (或者对应的模型文件夹名) 中。
3. 重新启动软件，程序会自动识别本地模型。

### 显存优化
- 程序默认会自动尝试使用 **FP16 (半精度)** 推理以节省显存。
- 如果遇到 NaN 错误或显存不足，可以尝试在代码中强制切换回 FP32（修改 `app_gradio_new.py` 中的 `model.half()` 部分）。

---

## 🙏 致谢 (Credits)

本项目核心算法基于 **BiRefNet**。感谢作者的杰出工作！
- **BiRefNet Repo**: [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- **HuggingFace**: [zhengpeng7/BiRefNet](https://huggingface.co/zhengpeng7)

## 📄 开源协议 (License)
本项目遵循 MIT 协议开源。
BiRefNet 模型权重遵循其原作者的许可协议。
