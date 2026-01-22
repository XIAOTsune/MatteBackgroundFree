import logging
import io
from collections import deque
from threading import Lock

class LogBuffer:
    """线程安全的日志缓冲区，用于在 UI 中显示日志"""
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LogBuffer, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._buffer = deque(maxlen=500)  # 保留最近500条日志
        self._lock = Lock()
        self._initialized = True
    
    def append(self, message):
        with self._lock:
            self._buffer.append(message)
    
    def get_logs(self):
        with self._lock:
            return "\n".join(self._buffer)
    
    def clear(self):
        with self._lock:
            self._buffer.clear()


class UILogHandler(logging.Handler):
    """将日志同时发送到 UI 缓冲区的 Handler"""
    def __init__(self, log_buffer):
        super().__init__()
        self.log_buffer = log_buffer
        
    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_buffer.append(msg)
        except Exception:
            self.handleError(record)


# 全局日志缓冲区
log_buffer = LogBuffer()

def setup_logger(name=__name__):
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(name)
    
    # 添加 UI 日志 handler
    ui_handler = UILogHandler(log_buffer)
    ui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ui_handler)
    
    return logger

logger = setup_logger()
