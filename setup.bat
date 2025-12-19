@echo off
echo ========================================
echo PhotoBlog 프로젝트 초기 설정
echo ========================================
echo.

REM 가상환경 생성
if exist venv (
    echo [경고] 가상환경이 이미 존재합니다.
    echo 기존 가상환경을 사용합니다.
) else (
    echo [1/4] 가상환경 생성 중...
    python -m venv venv
    echo [OK] 가상환경 생성 완료
)

REM 가상환경 활성화
echo.
echo [2/4] 가상환경 활성화 중...
call venv\Scripts\activate.bat
echo [OK] 가상환경 활성화 완료

REM 의존성 설치
echo.
echo [3/4] Python 패키지 설치 중...
pip install --upgrade pip
pip install -r requirements.txt
echo [OK] 패키지 설치 완료

REM 데이터베이스 마이그레이션
echo.
echo [4/4] 데이터베이스 마이그레이션 실행 중...
python manage.py migrate
echo [OK] 마이그레이션 완료

echo.
echo ========================================
echo 설정 완료!
echo ========================================
echo.
echo 다음 명령어로 서버를 시작할 수 있습니다:
echo   start_server.bat
echo 또는
echo   python manage.py runserver
echo.
pause

