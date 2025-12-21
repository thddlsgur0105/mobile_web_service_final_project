@echo off
echo ========================================
echo Django 서버 시작 (에뮬레이터용)
echo ========================================
echo.
echo 서버가 0.0.0.0:8000에서 실행됩니다.
echo 에뮬레이터에서 http://10.0.2.2:8000 으로 접근 가능합니다.
echo.
echo 서버를 중지하려면 Ctrl+C를 누르세요.
echo.

call venv\Scripts\activate.bat
python manage.py runserver 0.0.0.0:8000

pause

