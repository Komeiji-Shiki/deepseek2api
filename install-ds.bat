@echo off
:: 创建 conda 环境，Python 3.11，环境名为 dsapi
echo 正在创建 conda 环境 dsapi...
call conda create -n dsapi python=3.11 -y

:: 激活 conda 环境
echo 正在激活环境 dsapi...
call conda activate dsapi

:: 安装 requirements.txt 中的依赖（使用 pip）
echo 正在安装 requirements.txt 中的依赖...
pip install -r requirements.txt

echo 环境 dsapi 已创建并安装好依赖。
pause