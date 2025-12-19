@echo off
echo ========================================
echo PhotoBlog Django Server 시작
echo ========================================
echo.

REM 가상환경 활성화
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo [OK] 가상환경 활성화됨
) else (
    echo [경고] 가상환경이 없습니다. 생성 중...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo [OK] 가상환경 생성 및 활성화 완료
    echo.
    echo [정보] 의존성 설치 중...
    pip install -r requirements.txt
)

echo.
echo [정보] Django 서버 시작 중...
echo [정보] 브라우저에서 http://127.0.0.1:8000 접속 가능
echo [정보] 종료하려면 Ctrl+C를 누르세요
echo.

python manage.py runserver

pause

