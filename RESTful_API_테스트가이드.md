# RESTful API 사용 현황 및 테스트 가이드

## 📊 현재 사용 중인 RESTful API

### ✅ 사용 중인 API

1. **Post API** (`/api_root/Post/`)
   - **사용 위치**: `wellbeing_analyzer.py`의 `send_post_to_server()` 함수
   - **HTTP 메서드**: `POST`
   - **용도**: 검출된 객체 정보를 이미지와 함께 게시

2. **WellbeingLog API** (`/api_root/WellbeingLog/`)
   - **사용 위치**: `wellbeing_analyzer.py`의 `send_summary_to_server()` 함수
   - **HTTP 메서드**: `POST`
   - **용도**: 감정/활동성 데이터를 주기적으로 전송

---

## 🔍 사용 가능한 모든 RESTful API 엔드포인트

### 1. Post API

#### GET - 모든 포스트 조회
```http
GET http://127.0.0.1:8000/api_root/Post/
```

**테스트 방법:**
- **브라우저**: 주소창에 입력
- **curl**: `curl http://127.0.0.1:8000/api_root/Post/`
- **PowerShell**: `Invoke-WebRequest -Uri http://127.0.0.1:8000/api_root/Post/`

#### POST - 새 포스트 생성
```http
POST http://127.0.0.1:8000/api_root/Post/
Content-Type: multipart/form-data

title: "테스트 포스트"
text: "이것은 테스트입니다"
image: [이미지 파일]
```

**테스트 방법:**
```bash
# curl (이미지 파일 필요)
curl -X POST http://127.0.0.1:8000/api_root/Post/ \
  -F "title=테스트 포스트" \
  -F "text=이것은 테스트입니다" \
  -F "image=@test_image.jpg"
```

#### GET - 특정 포스트 조회
```http
GET http://127.0.0.1:8000/api_root/Post/{id}/
```

**테스트 방법:**
- 브라우저: `http://127.0.0.1:8000/api_root/Post/1/` (1은 포스트 ID)

#### PUT/PATCH - 포스트 수정
```http
PUT http://127.0.0.1:8000/api_root/Post/{id}/
PATCH http://127.0.0.1:8000/api_root/Post/{id}/
```

#### DELETE - 포스트 삭제
```http
DELETE http://127.0.0.1:8000/api_root/Post/{id}/
```

---

### 2. WellbeingLog API

#### GET - 모든 웰빙 로그 조회
```http
GET http://127.0.0.1:8000/api_root/WellbeingLog/
```

**테스트 방법:**
- 브라우저: `http://127.0.0.1:8000/api_root/WellbeingLog/`

#### POST - 새 웰빙 로그 생성
```http
POST http://127.0.0.1:8000/api_root/WellbeingLog/
Content-Type: application/json

{
  "dominant_emotion": "happy",
  "dominant_emotion_ratio": 0.75,
  "emotion_counts": {"happy": 10, "sad": 2},
  "avg_movement": 15.5
}
```

**테스트 방법:**
```bash
# curl
curl -X POST http://127.0.0.1:8000/api_root/WellbeingLog/ \
  -H "Content-Type: application/json" \
  -d '{
    "dominant_emotion": "happy",
    "dominant_emotion_ratio": 0.75,
    "emotion_counts": {"happy": 10, "sad": 2},
    "avg_movement": 15.5
  }'
```

#### GET - 웰빙 로그 요약
```http
GET http://127.0.0.1:8000/api_root/WellbeingLog/summary/
```

**테스트 방법:**
- 브라우저: `http://127.0.0.1:8000/api_root/WellbeingLog/summary/`

---

### 3. 인증 API

#### POST - 토큰 발급
```http
POST http://127.0.0.1:8000/api-token-auth/
Content-Type: application/json

{
  "username": "admin",
  "password": "password"
}
```

---

## 🧪 각 API별 동작 확인 방법

### 방법 1: 브라우저로 확인 (가장 쉬움)

1. **Django 서버 실행**
   ```bash
   python manage.py runserver
   ```

2. **브라우저에서 접속**
   - Post 목록: `http://127.0.0.1:8000/api_root/Post/`
   - WellbeingLog 목록: `http://127.0.0.1:8000/api_root/WellbeingLog/`
   - WellbeingLog 요약: `http://127.0.0.1:8000/api_root/WellbeingLog/summary/`

---

### 방법 2: PowerShell로 테스트

**Post 목록 조회:**
```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/api_root/Post/ | Select-Object -ExpandProperty Content
```

**WellbeingLog 생성:**
```powershell
$body = @{
    dominant_emotion = "happy"
    dominant_emotion_ratio = 0.75
    emotion_counts = '{"happy": 10, "sad": 2}'
    avg_movement = 15.5
} | ConvertTo-Json

Invoke-WebRequest -Uri http://127.0.0.1:8000/api_root/WellbeingLog/ `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

---

### 방법 3: Python 스크립트로 테스트

**테스트 스크립트 생성:**
```python
import requests

BASE_URL = "http://127.0.0.1:8000"

# 1. Post 목록 조회
response = requests.get(f"{BASE_URL}/api_root/Post/")
print("Post 목록:", response.json())

# 2. WellbeingLog 생성
data = {
    "dominant_emotion": "happy",
    "dominant_emotion_ratio": 0.75,
    "emotion_counts": {"happy": 10, "sad": 2},
    "avg_movement": 15.5
}
response = requests.post(f"{BASE_URL}/api_root/WellbeingLog/", json=data)
print("WellbeingLog 생성:", response.status_code)

# 3. WellbeingLog 요약 조회
response = requests.get(f"{BASE_URL}/api_root/WellbeingLog/summary/")
print("WellbeingLog 요약:", response.json())
```

---

### 방법 4: Postman 사용 (고급)

1. Postman 설치: https://www.postman.com/downloads/
2. 새 Request 생성
3. HTTP 메서드 선택 (GET, POST, PUT, DELETE 등)
4. URL 입력: `http://127.0.0.1:8000/api_root/Post/`
5. Body에 데이터 입력 (POST/PUT 시)
6. Send 클릭

---

## 📋 API 동작 확인 체크리스트

### Post API 확인
- [ ] `GET /api_root/Post/` - 목록 조회 성공
- [ ] `POST /api_root/Post/` - 포스트 생성 성공
- [ ] `GET /api_root/Post/{id}/` - 특정 포스트 조회 성공
- [ ] Wellbeing Analyzer 실행 시 자동 게시 확인

### WellbeingLog API 확인
- [ ] `GET /api_root/WellbeingLog/` - 목록 조회 성공
- [ ] `POST /api_root/WellbeingLog/` - 로그 생성 성공
- [ ] `GET /api_root/WellbeingLog/summary/` - 요약 조회 성공
- [ ] Wellbeing Analyzer 실행 시 자동 전송 확인

---

## 🔧 실제 사용 확인 방법

### 1. Wellbeing Analyzer 실행으로 자동 확인

```bash
# 터미널 1: Django 서버
python manage.py runserver

# 터미널 2: Wellbeing Analyzer
python wellbeing_analyzer.py
```

**확인 사항:**
- 콘솔에 `✅ Post 게시 성공` 메시지 확인
- 브라우저에서 `http://127.0.0.1:8000/api_root/Post/` 접속하여 게시된 포스트 확인
- 브라우저에서 `http://127.0.0.1:8000/api_root/WellbeingLog/` 접속하여 생성된 로그 확인

### 2. 수동 API 호출 테스트

**간단한 테스트 스크립트 실행:**
```bash
python test_api.py  # 아래에 스크립트 제공
```

---

## 🐛 문제 해결

### API가 응답하지 않을 때
1. Django 서버가 실행 중인지 확인
2. URL이 정확한지 확인 (`/api_root/Post/` 끝에 `/` 필수)
3. 브라우저 콘솔에서 에러 확인 (F12)

### POST 요청이 실패할 때
1. Content-Type 헤더 확인 (`application/json` 또는 `multipart/form-data`)
2. 필수 필드가 모두 포함되었는지 확인
3. 서버 로그 확인 (터미널에서 에러 메시지 확인)

---

## 📚 참고

- Django REST Framework 문서: https://www.django-rest-framework.org/
- HTTP 메서드 설명: https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods

