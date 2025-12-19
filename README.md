# PhotoBlog 프로젝트 실행 가이드

이 프로젝트는 Django 기반의 사진 블로그 서버와 웰빙 분석 시스템을 포함합니다.

## 📋 프로젝트 구성

- **Django 서버** (`PhotoBlogServer/`): REST API를 제공하는 백엔드 서버
- **Blog 앱** (`blog/`): 포스트 및 웰빙 로그 관리
- **Wellbeing Analyzer** (`wellbeing_analyzer.py`): YOLOv5 + DeepFace를 사용한 실시간 감정/활동 분석
- **Socket Server** (`socket_server/`): 바이너리 데이터 수신 서버
- **Android 앱** (`PhotoViewer/`): 모바일 클라이언트

---

## 🚀 빠른 시작 가이드

### 1단계: Python 환경 설정

#### 가상환경 생성 및 활성화

**Windows PowerShell:**
```powershell
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
.\venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
venv\Scripts\activate.bat
```

#### 의존성 설치

```bash
# Django 프로젝트 기본 의존성
pip install -r requirements.txt

# Wellbeing Analyzer 실행을 위한 추가 의존성 (선택사항)
# YOLOv5는 torch.hub에서 자동 다운로드됩니다
pip install deepface
```

---

### 2단계: Django 서버 실행

#### 데이터베이스 마이그레이션

```bash
# 마이그레이션 파일 생성 (필요시)
python manage.py makemigrations

# 데이터베이스 마이그레이션 실행
python manage.py migrate

# 관리자 계정 생성 (선택사항)
python manage.py createsuperuser
```

#### 서버 실행

```bash
# 개발 서버 시작 (기본 포트: 8000)
python manage.py runserver

# 특정 포트 지정
python manage.py runserver 8000

# 모든 호스트에서 접근 가능하도록 실행
python manage.py runserver 0.0.0.0:8000
```

서버가 실행되면 브라우저에서 다음 주소로 접속할 수 있습니다:
- 메인 페이지: http://127.0.0.1:8000/
- 관리자 페이지: http://127.0.0.1:8000/admin/
- API 루트: http://127.0.0.1:8000/api_root/

---

### 3단계: Wellbeing Analyzer 실행 (선택사항)

웹캠을 사용한 실시간 감정 및 활동 분석을 실행합니다.

**주의사항:**
- 웹캠이 연결되어 있어야 합니다
- 첫 실행 시 YOLOv5 모델이 자동으로 다운로드됩니다 (시간 소요)
- DeepFace 모델도 첫 실행 시 다운로드됩니다

```bash
# 가상환경 활성화 상태에서 실행
python wellbeing_analyzer.py
```

**기능:**
- 웹캠에서 사람 감지 (YOLOv5)
- 감정 분석 (DeepFace)
- 활동성 측정 (움직임 추적)
- 60초마다 Django 서버로 데이터 전송

**종료:** ESC 키를 누르면 종료됩니다.

---

### 4단계: Socket Server 실행 (선택사항)

바이너리 데이터를 수신하는 소켓 서버를 실행합니다.

```bash
python socket_server/socket_server.py
```

**기능:**
- 포트 9000에서 클라이언트 연결 대기
- 수신한 데이터를 `socket_server/request/` 폴더에 저장

**종료:** Ctrl+C로 종료합니다.

---

### 5단계: Android 앱 실행 (선택사항)

자세한 내용은 `PhotoViewer/README.md`를 참조하세요.

**요약:**
1. Android Studio에서 `PhotoViewer` 프로젝트 열기
2. Gradle 동기화 대기
3. 에뮬레이터 또는 실제 기기 연결
4. Run 버튼 클릭

---

## 📡 API 엔드포인트

### Post API
- **GET** `/api_root/Post/` - 모든 포스트 조회
- **POST** `/api_root/Post/` - 새 포스트 생성
- **GET** `/api_root/Post/{id}/` - 특정 포스트 조회
- **PUT/PATCH** `/api_root/Post/{id}/` - 포스트 수정
- **DELETE** `/api_root/Post/{id}/` - 포스트 삭제

### WellbeingLog API
- **GET** `/api_root/WellbeingLog/` - 모든 웰빙 로그 조회
- **POST** `/api_root/WellbeingLog/` - 새 웰빙 로그 생성
- **GET** `/api_root/WellbeingLog/summary/` - 웰빙 로그 요약 (오늘/최근 7일)

### 인증
- **POST** `/api-token-auth/` - 토큰 인증 (username, password 필요)

---

## 🔧 설정 파일

### Django 설정 (`PhotoBlogServer/settings.py`)

주요 설정:
- `DEBUG = True` - 개발 모드 (프로덕션에서는 False로 변경)
- `ALLOWED_HOSTS` - 접근 허용 호스트 목록
- `TIME_ZONE = 'Asia/Seoul'` - 시간대 설정
- `MEDIA_ROOT` - 업로드된 미디어 파일 저장 경로
- `STATIC_ROOT` - 정적 파일 저장 경로

### Wellbeing Analyzer 설정 (`wellbeing_analyzer.py`)

주요 설정:
- `DJANGO_BASE_URL` - Django 서버 주소 (기본: `http://127.0.0.1:8000`)
- `SUMMARY_INTERVAL` - 서버 전송 주기 (초 단위, 기본: 60초)
- `WINDOW_SECONDS` - 데이터 집계 윈도우 크기 (기본: 60초)

---

## 📁 프로젝트 구조

```
PhotoBlog/
├── PhotoBlogServer/          # Django 프로젝트 설정
│   ├── settings.py           # 프로젝트 설정
│   ├── urls.py               # URL 라우팅
│   └── wsgi.py               # WSGI 설정
├── blog/                     # Blog 앱
│   ├── models.py             # 데이터 모델 (Post, WellbeingLog)
│   ├── views.py              # API 뷰셋
│   ├── serializers.py        # DRF 시리얼라이저
│   ├── urls.py               # 앱 URL 설정
│   └── migrations/           # 데이터베이스 마이그레이션
├── socket_server/            # 소켓 서버
│   └── socket_server.py      # 소켓 서버 스크립트
├── wellbeing_analyzer.py     # 웰빙 분석 스크립트
├── manage.py                 # Django 관리 스크립트
├── requirements.txt          # Python 의존성
├── db.sqlite3                # SQLite 데이터베이스
├── media/                    # 업로드된 미디어 파일
└── static/                   # 정적 파일
```

---

## 🐛 문제 해결

### Django 서버가 시작되지 않음
- 가상환경이 활성화되어 있는지 확인
- `pip install -r requirements.txt` 실행
- 포트 8000이 이미 사용 중인지 확인 (`netstat -ano | findstr :8000`)

### Wellbeing Analyzer 실행 오류
- 웹캠이 연결되어 있는지 확인
- `pip install deepface` 실행 확인
- 첫 실행 시 모델 다운로드 시간이 필요합니다 (인터넷 연결 필요)

### 데이터베이스 오류
- `python manage.py migrate` 실행
- `db.sqlite3` 파일 삭제 후 다시 마이그레이션

### 이미지가 표시되지 않음
- `MEDIA_ROOT` 경로 확인
- Django 서버가 `DEBUG=True` 모드로 실행 중인지 확인
- `media/` 폴더 권한 확인

---

## 🔐 보안 주의사항

⚠️ **프로덕션 배포 전 필수 사항:**

1. `SECRET_KEY` 변경 (환경 변수로 관리 권장)
2. `DEBUG = False` 설정
3. `ALLOWED_HOSTS`에 실제 도메인만 추가
4. HTTPS 설정
5. 데이터베이스 백업 설정
6. 정적 파일 서빙 설정 (Nginx 등)

---

## 📝 추가 정보

- **Django 버전:** 5.2.7
- **Python 버전:** 3.8+ 권장
- **데이터베이스:** SQLite3 (개발용)
- **API 프레임워크:** Django REST Framework

---

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. 가상환경 활성화 여부
2. 모든 의존성 설치 여부
3. 데이터베이스 마이그레이션 실행 여부
4. 포트 충돌 여부

