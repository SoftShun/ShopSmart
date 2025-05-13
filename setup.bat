@echo off
echo ShopSmart 시스템 환경 설정 시작...
echo.

cd /d "%~dp0"

:: 가상환경 생성
echo 가상환경 생성 중...
python -m venv venv

:: 가상환경 활성화
echo 가상환경 활성화 중...
call venv\Scripts\activate

:: 필요한 라이브러리 설치
echo 필요한 라이브러리 설치 중...
pip install -r requirements.txt

:: 환경 변수 파일 생성 (템플릿 복사)
if not exist .env (
  echo .env 파일 생성 중...
  copy .env.template .env
  echo.
  echo .env 파일이 생성되었습니다. 이 파일을 편집하여 OpenAI API 키를 설정하세요.
) else (
  echo .env 파일이 이미 존재합니다.
)

:: 디렉토리 생성
if not exist logs (
  echo 로그 디렉토리 생성 중...
  mkdir logs
)

if not exist chroma_db (
  echo Chroma DB 디렉토리 생성 중...
  mkdir chroma_db
)

echo.
echo 환경 설정이 완료되었습니다!
echo OpenAI API 키를 .env 파일에 설정한 후 다음 명령어로 시스템을 실행할 수 있습니다:
echo.
echo   run_api.bat    - API 서버 실행
echo   run_app.bat    - Streamlit 앱 실행
echo   run_chatbot.bat - 대화형 챗봇 실행
echo.
echo 주의: API 서버가 실행되어야 Streamlit 앱이 제대로 작동합니다.
echo.

pause
