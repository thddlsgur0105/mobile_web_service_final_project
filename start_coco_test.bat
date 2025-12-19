@echo off
echo ========================================
echo COCO 80가지 객체 검출 테스트
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
echo [1] 클래스 목록 확인
python list_coco_classes.py

echo.
echo [2] 웹캠 실시간 검출 테스트 시작
echo ESC 키를 누르면 종료됩니다.
echo.
python test_coco_detection.py

pause

