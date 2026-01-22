@echo off
chcp 65001 >nul
setlocal

echo ========================================================
echo       正在启动 MatteBackgroundFree...
echo ========================================================

REM 检测虚拟环境是否存在
if not exist venv (
    echo [错误] 未找到虚拟环境！
    echo 请先运行 "install.bat" (或 "一键安装.bat") 来安装环境。
    pause
    exit /b
)

REM 激活虚拟环境
call venv\Scripts\activate

REM 启动主程序
python app_gradio_new.py

echo.
echo 程序已退出。
pause
