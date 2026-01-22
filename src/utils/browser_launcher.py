import webbrowser
import subprocess
import shutil
import platform
import os
from src.utils.logger import logger

def open_app_window(url):
    """
    尝试以“应用模式”打开 URL (无地址栏、无书签栏)，提供更像原生应用体验。
    按顺序尝试:
    1. Microsoft Edge (Windows 默认)
    2. Google Chrome
    3. 默认浏览器 (回退)
    """
    system = platform.system()
    
    if system == "Windows":
        # 尝试 Edge
        edge_paths = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
        ]
        for path in edge_paths:
            if os.path.exists(path):
                try:
                    logger.info(f"Using Edge App Mode: {path}")
                    return subprocess.Popen([path, f"--app={url}"])
                except Exception as e:
                    logger.warning(f"Failed to launch Edge: {e}")

    # 尝试 Chrome (跨平台通常命令类似)
    # Windows
    if system == "Windows":
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe")
        ]
        for path in chrome_paths:
            if os.path.exists(path):
                try:
                    logger.info(f"Using Chrome App Mode: {path}")
                    return subprocess.Popen([path, f"--app={url}"])
                    return
                except Exception as e:
                    logger.warning(f"Failed to launch Chrome: {e}")

    # Fallback to standard browser tag
    logger.info("Browser app mode not found, falling back to default browser.")
    webbrowser.open(url)
