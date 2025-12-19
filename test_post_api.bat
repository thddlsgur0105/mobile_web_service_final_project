@echo off
echo ========================================
echo Post API 테스트 가이드
echo ========================================
echo.
echo [중요] 이 스크립트를 실행하기 전에:
echo   1. Django 서버가 실행 중이어야 합니다
echo   2. 다른 터미널에서 start_server.bat을 실행하세요
echo.
echo ========================================
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
echo [정보] Wellbeing Analyzer 시작 중...
echo [정보] 웹캠이 연결되어 있어야 합니다.
echo [정보] 새 객체가 검출되면 자동으로 Post가 게시됩니다.
echo [정보] 종료하려면 ESC 키를 누르세요
echo.
echo ========================================
echo.

python wellbeing_analyzer.py

echo.
echo ========================================
echo 테스트 완료!
echo.
echo 게시된 Post 확인:
echo   http://127.0.0.1:8000/api_root/Post/
echo ========================================
echo.
pause

