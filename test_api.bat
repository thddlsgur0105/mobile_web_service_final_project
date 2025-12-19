@echo off
echo ========================================
echo RESTful API 테스트
echo ========================================
echo.
echo [중요] Django 서버가 실행 중이어야 합니다!
echo.
pause

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
echo [정보] API 테스트 시작...
echo.

python test_api.py

echo.
pause

