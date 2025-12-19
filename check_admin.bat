@echo off
echo ========================================
echo Django 관리자 계정 확인
echo ========================================
echo.

REM 가상환경 활성화
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo [OK] 가상환경 활성화됨
) else (
    echo [오류] 가상환경이 없습니다.
    echo 먼저 setup.bat을 실행하세요.
    pause
    exit /b 1
)

echo.
echo [정보] 관리자 계정 확인 중...
echo.

python check_admin.py

echo.
echo ========================================
echo 비밀번호 재설정 방법:
echo   python manage.py changepassword admin
echo 또는
echo   python check_admin.py admin newpassword
echo ========================================
echo.
pause

