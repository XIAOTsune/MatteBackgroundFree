import os
import sys
import atexit

# === STARTUP FIX: Redirect stdout/stderr for EXE compatibility ===
# This must be done BEFORE any other imports to prevent libraries from
# caching the original broken streams (which are None in NO_WINDOW mode).

# === 全局变量：保存浏览器进程引用和Gradio实例 ===
browser_process = None
demo_instance = None

def cleanup_processes():
    """
    退出时清理所有子进程
    使用 psutil 递归终止所有子进程，防止进程残留
    """
    global browser_process
    try:
        import psutil
        # 获取当前进程
        current_process = psutil.Process()
        
        # 获取所有子进程（递归获取）
        children = current_process.children(recursive=True)
        
        if children:
            print(f"清理 {len(children)} 个子进程...")
            
            # 先尝试优雅终止
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # 等待进程终止（最多3秒）
            gone, alive = psutil.wait_procs(children, timeout=3)
            
            # 强制杀死仍然存活的进程
            for p in alive:
                try:
                    print(f"强制终止进程: {p.pid}")
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
        
        # 单独处理浏览器进程（如果有引用）
        if browser_process:
            try:
                browser_process.terminate()
                browser_process.wait(timeout=2)
            except:
                try:
                    browser_process.kill()
                except:
                    pass
                    
    except ImportError:
        # 如果 psutil 未安装，使用基础方法
        print("警告: psutil 未安装，使用基础清理方法")
        if browser_process:
            try:
                browser_process.terminate()
            except:
                try:
                    browser_process.kill()
                except:
                    pass
    except Exception as e:
        print(f"清理进程时出错: {e}")

# 注册退出清理函数
atexit.register(cleanup_processes)

def shutdown_application():
    """
    统一的应用退出函数
    由浏览器监控或退出按钮调用，优雅地关闭整个应用
    """
    global demo_instance
    
    try:
        logger.info("正在关闭Gradio服务器...")
        if demo_instance:
            try:
                demo_instance.close()
                logger.info("Gradio服务器已关闭")
            except Exception as e:
                logger.warning(f"关闭Gradio时出错: {e}")
        
        # 清理子进程
        cleanup_processes()
        
        logger.info("程序已安全退出")
        
        # 强制退出主进程
        os._exit(0)
        
    except Exception as e:
        logger.error(f"退出时出错: {e}")
        os._exit(1)

def monitor_browser_process():
    """
    监控浏览器进程，当浏览器窗口关闭时自动退出程序
    
    解决方案：
    Edge/Chrome 的 --app 模式启动器进程会立即退出，但会创建实际的浏览器窗口子进程。
    我们使用 psutil 查找名为 msedge.exe 或 chrome.exe 的进程，这些才是真正的浏览器窗口。
    """
    global browser_process, demo_instance
    
    import time
    
    # 等待浏览器进程启动
    time.sleep(3)
    
    try:
        import psutil
        
        # 记录启动时存在的浏览器进程PID
        browser_pids_before = set()
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                name = proc.info['name'].lower()
                if 'msedge.exe' in name or 'chrome.exe' in name:
                    browser_pids_before.add(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        logger.info(f"浏览器进程监控已启动 (启动前有 {len(browser_pids_before)} 个浏览器进程)")
        
        # 等待新的浏览器进程出现
        time.sleep(2)
        
        # 查找新启动的浏览器进程（应用窗口）
        app_browser_pids = set()
        for proc in psutil.process_iter(['pid', 'name', 'create_time']):
            try:
                name = proc.info['name'].lower()
                pid = proc.info['pid']
                if ('msedge.exe' in name or 'chrome.exe' in name) and pid not in browser_pids_before:
                    app_browser_pids.add(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if not app_browser_pids:
            logger.warning("未检测到新的浏览器进程，监控功能使用全部浏览器进程")
            app_browser_pids = browser_pids_before
        
        logger.info(f"监控 {len(app_browser_pids)} 个浏览器进程: {app_browser_pids}")
        
        # 持续监控这些进程
        check_count = 0
        while True:
            time.sleep(2)
            check_count += 1
            
            # 检查是否还有我们的浏览器进程在运行
            still_running = []
            for pid in app_browser_pids:
                try:
                    proc = psutil.Process(pid)
                    if proc.is_running():
                        still_running.append(pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # 如果所有监控的浏览器进程都退出了
            if not still_running:
                logger.info("检测到浏览器窗口已关闭，准备退出程序...")
                time.sleep(1)  # 稍微延迟，确保不是误判
                shutdown_application()
                break
            
            # 每30秒输出一次状态
            if check_count % 15 == 0:
                logger.debug(f"监控中，当前运行 {len(still_running)} 个浏览器进程")
                
    except ImportError:
        logger.error("psutil 未安装，无法使用浏览器监控功能。请使用右上角退出按钮。")
        # 保持线程运行但不监控
        while True:
            time.sleep(60)
    except Exception as e:
        logger.error(f"浏览器监控出错: {e}")
        logger.info("监控功能已停止，请使用右上角退出按钮。")
        while True:
            time.sleep(60)

class SafeLogger:
    def __init__(self, filename, stream):
        self.filename = filename
        self.stream = stream

    def write(self, message):
        # Write to file
        try:
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(message)
        except: pass
        
        # Write to original stream if it exists
        if self.stream:
            try: 
                self.stream.write(message)
                self.stream.flush()
            except: pass

    def flush(self):
        if self.stream:
            try: self.stream.flush()
            except: pass
    
    def isatty(self):
        # This is key! Uvicorn/Gradio checks this.
        # Return False to indicate this is not an interactive terminal.
        return False

# 1. Setup Logging File
try:
    log_file = os.path.join(os.getcwd(), "startup_log.txt")
    # Clear log file on new run
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"=== Application Startup: {os.getcwd()} ===\n")
except Exception as e:
    # If we can't write to CWD, try temp or just ignore file logging
    log_file = "startup_log.txt" 

# 2. Patch sys.stdout and sys.stderr
# We wrap them even if they exist, to ensure .isatty() behaves correctly
# and to capture all logs to file.
sys.stdout = SafeLogger(log_file, sys.stdout)
sys.stderr = SafeLogger(log_file, sys.stderr)

# =================================================================

# Ensure the current directory is in the path so that src can be imported
sys.path.append(os.getcwd())

from src.ui import create_interface
from src.utils.logger import logger

if __name__ == "__main__":
    try:
        # === PROXY CONFIGURATION (Smart) ===
        # 1. Force localhost to bypass proxy so the UI can launch locally without 502 errors.
        # 2. KEEP existing proxy settings (VPN) so users can download models from HuggingFace.
        
        # Get existing NO_PROXY or empty string
        current_no_proxy = os.environ.get("NO_PROXY", "") or os.environ.get("no_proxy", "")
        no_proxy_list = [x.strip() for x in current_no_proxy.split(",") if x.strip()]
        
        # Ensure essential local addresses are ignored by proxy
        essentials = ["localhost", "127.0.0.1", "::1", "0.0.0.0"]
        for item in essentials:
            if item not in no_proxy_list:
                no_proxy_list.append(item)
        
        # Apply updated NO_PROXY
        os.environ["NO_PROXY"] = ",".join(no_proxy_list)
        # For some libraries case might matter, set both
        os.environ["no_proxy"] = os.environ["NO_PROXY"]

        logger.info("Starting 小T的抠图工具箱...")
        logger.info(f"Network Config: Proxy={os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy', 'None')}")
        logger.info(f"Network Config: NO_PROXY={os.environ['NO_PROXY']}")
        
        # Find a free port starting from 7860
        import socket
        def find_free_port(start_port=7860):
            port = start_port
            while True:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    if sock.connect_ex(('127.0.0.1', port)) != 0:
                        return port
                    port += 1
        
        server_port = find_free_port()
        logger.info(f"Selected port: {server_port}")
        
        import threading
        import time
        from src.utils.browser_launcher import open_app_window

        def launch_browser():
            """启动浏览器窗口"""
            global browser_process
            time.sleep(1.5)  # Wait a bit for server to start
            logger.info(f"Opening browser at http://127.0.0.1:{server_port}")
            process = open_app_window(f"http://127.0.0.1:{server_port}")
            browser_process = process  # 保存引用供cleanup_processes使用

        threading.Thread(target=launch_browser, daemon=True).start()
        
        # 启动浏览器进程监控线程
        threading.Thread(target=monitor_browser_process, daemon=True).start()
        # 注意：浏览器自动监控已禁用，请使用界面右上角的"退出程序"按钮
        
        demo = create_interface()
        demo_instance = demo  # 保存Gradio实例引用，供退出函数使用
        
        # 端口绑定重试机制（防止竞态条件）
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"尝试启动Gradio服务器 (尝试 {attempt + 1}/{max_retries})...")
                demo.queue().launch(
                    server_name="127.0.0.1",
                    server_port=server_port,
                    inbrowser=False, 
                    show_error=True, 
                    favicon_path="favicon.png"
                )
                break  # 启动成功，退出重试循环
                
            except OSError as e:
                last_error = e
                if "address already in use" in str(e).lower() or "被占用" in str(e):
                    logger.warning(f"端口 {server_port} 被占用，重新查找空闲端口...")
                    server_port = find_free_port(server_port + 1)
                    logger.info(f"找到新端口: {server_port}")
                    
                    if attempt < max_retries - 1:
                        continue  # 重试
                    else:
                        raise  # 最后一次尝试失败，抛出异常
                else:
                    # 其他OSError，直接抛出
                    raise

    except Exception as e:
        logger.error(f"Failed to start app: {e}")
        import traceback
        traceback.print_exc()
        # In EXE mode, input() might fail if stdin is closed, but we try anyway.
        try:
            input("Press Enter to exit...")
        except:
            pass