@echo off
echo ShopSmart API 서버 시작...
echo.
cd /d "%~dp0"
call venv\Scripts\activate
python api_server.py
