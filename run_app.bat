@echo off
echo ShopSmart Streamlit 앱 시작...
echo.
cd /d "%~dp0"
call venv\Scripts\activate
streamlit run app.py
