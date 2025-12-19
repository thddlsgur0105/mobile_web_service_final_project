@echo off
echo ========================================
echo Wellbeing Analyzer 시작
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

echo [정보] Wellbeing Analyzer 시작 중...
echo [정보] 웹캠이 연결되어 있어야 합니다.
echo [정보] 종료하려면 ESC 키를 누르세요
echo.

python wellbeing_analyzer.py

pause

