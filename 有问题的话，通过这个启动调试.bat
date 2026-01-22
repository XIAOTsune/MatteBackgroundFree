@echo off
chcp 65001 >nul
cd /d %~dp0
title BiRefNet - 启动器
color 07

:: 设置环境变量

set PYTHONUSERBASE=.\python\Lib\site-packages
set HF_HOME=%CD%\models_local
set TRANSFORMERS_CACHE=%CD%\models_local
set PATH=%PATH%;.\python\Scripts
set GRADIO_ANALYTICS_ENABLED=FALSE

:: 强制 localhost 和本地 IP 不走代理
set NO_PROXY=localhost,127.0.0.1,0.0.0.0,::1

:: 如果需要通过 VPN 下载模型，请取消下面两行的注释(::)，并修改端口号为你 VPN 的端口
:: set HTTP_PROXY=http://127.0.0.1:7897
:: set HTTPS_PROXY=http://127.0.0.1:7897
:: ==========================================

cls
echo.
echo ================================================================
echo                 【✨ BiRefNet 图片/视频背景替换工具 ✨】
echo ================================================================
echo.
echo     感谢原作者: 郑鹏 (ZhengPeng7)  github.com/ZhengPeng7/BiRefNet
echo     代码修改与功能添加: 小T_sune
echo.
echo     ⚠ 本软件仅供学习与研究使用，严禁商用！
echo ================================================================
echo.
echo     启动时间: %date% %time%
echo     操作系统: %OS%
echo ================================================================
echo.
echo 🚀 正在启动 BiRefNet，请稍候加载模型中...
echo ================================================================
echo.

:: 后台启动主程序（不阻塞显示）
.\python\python.exe app.py

:: 显示加载动画（Python 正在后台加载）
setlocal enabledelayedexpansion
for /l %%i in (1,1,12) do (
    set /a dots=%%i %% 4
    set "progress=加载中"
    for /l %%j in (1,1,!dots!) do set "progress=!progress!."
    <nul set /p=!progress!
    timeout /t 1 >nul
    <nul set /p=                  
)
endlocal

echo.
echo ================================================================
echo 🌟 模型已启动，请在浏览器中使用 BiRefNet！
echo ================================================================
pause
