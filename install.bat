@echo off
setlocal
chcp 65001 >nul

echo ========================================================
echo       MatteBackgroundFree 一键环境安装脚本
echo ========================================================
echo.

REM 1. 检测 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10 或更高版本。
    echo 推荐下载：https://www.python.org/downloads/
    echo 注意：安装时请务必勾选 "Add Python to PATH"
    pause
    exit /b
)

echo [1/4] 检测到 Python，准备创建虚拟环境...
if not exist venv (
    
    REM 尝试创建虚拟环境
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [错误] 创建虚拟环境失败。请检查 Python 是否安装正确。
        pause
        exit /b
    )
    echo 虚拟环境创建成功！
) else (
    echo 虚拟环境已存在，跳过创建。
)

echo.
echo [2/4] 激活虚拟环境...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [错误] 无法激活虚拟环境。
    pause
    exit /b
)

echo.
echo [3/4] 升级 pip 并配置国内镜像源...
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo [4/4] 正在安装依赖库 (这可能需要几分钟)...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

if %errorlevel% neq 0 (
    echo.
    echo [错误] 依赖安装失败！请检查网络连接。
    pause
    exit /b
)

echo.
echo ========================================================
echo              安装完成！
echo   请双击 "run.bat" 或 "一键启动.bat" 开始使用
echo ========================================================
pause
