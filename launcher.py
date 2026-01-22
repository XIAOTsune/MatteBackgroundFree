"""
小T的抠图工具箱 - 启动器
此脚本用于启动主应用，编译为 EXE 后可作为入口
"""
import os
import sys
import subprocess
import threading
import time

def show_loading_window():
    """显示启动加载窗口"""
    try:
        import tkinter as tk
        from tkinter import ttk
        
        # 创建窗口
        root = tk.Tk()
        root.title("小T的抠图工具箱")
        root.geometry("400x200")
        root.resizable(False, False)
        
        # 居中显示
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - 400) // 2
        y = (screen_height - 200) // 2
        root.geometry(f"400x200+{x}+{y}")
        
        # 设置图标（如果存在）
        try:
            icon_path = os.path.join(os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__), "logo.ico")
            if os.path.exists(icon_path):
                root.iconbitmap(icon_path)
        except:
            pass
        
        # 标题
        title_label = tk.Label(root, text="🎯 小T的抠图工具箱", font=("微软雅黑", 16, "bold"))
        title_label.pack(pady=20)
        
        # 状态文本
        status_label = tk.Label(root, text="正在启动程序，请稍候...", font=("微软雅黑", 10))
        status_label.pack(pady=10)
        
        # 进度条
        progress = ttk.Progressbar(root, mode='indeterminate', length=300)
        progress.pack(pady=10)
        progress.start(10)
        
        # 提示文本
        tip_label = tk.Label(root, text="首次启动可能需要下载模型，请耐心等待", font=("微软雅黑", 8), fg="gray")
        tip_label.pack(pady=10)
        
        # 存储窗口引用，供外部关闭
        return root
        
    except ImportError:
        # 如果tkinter不可用，返回None
        return None

def main():
    # 获取当前 exe 所在目录
    if getattr(sys, 'frozen', False):
        # 如果是打包后的 exe
        app_dir = os.path.dirname(sys.executable)
    else:
        # 如果是直接运行 py 文件
        app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 切换到应用目录
    os.chdir(app_dir)
    
    # 构建 Python 解释器和 app.py 的路径
    python_exe = os.path.join(app_dir, "python", "python.exe")
    app_py = os.path.join(app_dir, "app.py")
    
    # 检查文件是否存在
    if not os.path.exists(python_exe):
        print(f"错误：找不到 Python 解释器: {python_exe}")
        input("按 Enter 键退出...")
        return 1
    
    if not os.path.exists(app_py):
        print(f"错误：找不到主程序: {app_py}")
        input("按 Enter 键退出...")
        return 1
    
    # 显示加载窗口
    loading_window = show_loading_window()
    
    # 启动主程序（后台）
    try:
        process = subprocess.Popen(
            [python_exe, app_py],
            cwd=app_dir,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 等待主程序启动（检查是否有浏览器窗口打开）
        if loading_window:
            # 最多等待30秒
            wait_count = 0
            while wait_count < 30:
                loading_window.update()
                time.sleep(0.5)
                wait_count += 0.5
                
                # 检查进程是否还在运行（如果崩溃会立即退出）
                if process.poll() is not None:
                    # 进程已退出，可能是错误
                    stdout, stderr = process.communicate()
                    if process.returncode != 0:
                        loading_window.destroy()
                        print(f"启动失败，错误码: {process.returncode}")
                        if stderr:
                            print(stderr.decode('utf-8', errors='ignore'))
                        input("按 Enter 键退出...")
                        return process.returncode
                    break
                
                # 检查是否已经有浏览器窗口打开（简单检测：等待15秒后关闭加载窗口）
                if wait_count >= 15:
                    break
            
            # 关闭加载窗口
            loading_window.destroy()
        
        # 等待主程序结束
        return process.wait()
        
    except Exception as e:
        if loading_window:
            loading_window.destroy()
        print(f"启动失败: {e}")
        input("按 Enter 键退出...")
        return 1

if __name__ == "__main__":
    sys.exit(main())
