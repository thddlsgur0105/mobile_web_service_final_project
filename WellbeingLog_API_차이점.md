# WellbeingLog API 엔드포인트 차이점 설명

## 📊 두 엔드포인트 비교

### 1. `/api_root/WellbeingLog/` (기본 RESTful API)

**타입**: Django REST Framework의 **ModelViewSet** (표준 REST API)

**기능**:
- **GET**: 모든 WellbeingLog 목록 조회 (개별 로그 데이터)
- **POST**: 새 WellbeingLog 생성
- **GET /{id}/**: 특정 로그 상세 조회
- **PUT/PATCH /{id}/**: 로그 수정
- **DELETE /{id}/**: 로그 삭제

**반환 데이터 예시**:
```json
[
  {
    "id": 1,
    "created_at": "2025-01-19T10:30:00Z",
    "dominant_emotion": "happy",
    "dominant_emotion_ratio": 0.75,
    "emotion_counts": {"happy": 10, "sad": 2},
    "avg_movement": 15.5
  },
  {
    "id": 2,
    "created_at": "2025-01-19T10:31:00Z",
    "dominant_emotion": "neutral",
    "dominant_emotion_ratio": 0.60,
    "emotion_counts": {"neutral": 8, "happy": 3},
    "avg_movement": 12.3
  }
]
```

**특징**:
- ✅ 표준 RESTful API (CRUD 작업)
- ✅ 개별 로그 데이터를 그대로 반환
- ✅ 페이징 지원 (여러 로그가 있을 경우)

---

### 2. `/api_root/WellbeingLog/summary/` (커스텀 RESTful API)

**타입**: Django REST Framework의 **@action 데코레이터** (커스텀 REST API)

**기능**:
- **GET만 가능**: 데이터를 집계하여 요약 정보 반환
- 오늘 0시 이후 데이터 집계
- 최근 7일 데이터 집계

**반환 데이터 예시**:
```json
{
  "today": {
    "count": 5,
    "dominant_emotion": "happy",
    "emotion_counts": {
      "happy": 25,
      "neutral": 10,
      "sad": 3
    },
    "avg_movement": 14.2
  },
  "last_7_days": {
    "count": 42,
    "dominant_emotion": "happy",
    "emotion_counts": {
      "happy": 180,
      "neutral": 95,
      "sad": 25
    },
    "avg_movement": 13.8
  }
}
```

**특징**:
- ✅ 커스텀 RESTful API 엔드포인트
- ✅ 데이터 집계 및 통계 계산
- ✅ 여러 로그를 하나의 요약으로 반환
- ✅ GET 메서드만 지원

---

## 🔍 주요 차이점 요약

| 항목 | `/api_root/WellbeingLog/` | `/api_root/WellbeingLog/summary/` |
|------|---------------------------|-----------------------------------|
| **타입** | 표준 ModelViewSet | 커스텀 @action |
| **HTTP 메서드** | GET, POST, PUT, PATCH, DELETE | GET만 |
| **반환 데이터** | 개별 로그 목록 | 집계된 요약 정보 |
| **데이터 형태** | 배열 (여러 로그) | 객체 (요약 통계) |
| **용도** | 로그 조회/생성/수정/삭제 | 통계 및 요약 정보 조회 |
| **RESTful** | ✅ 표준 RESTful | ✅ 커스텀 RESTful |

---

## 💡 사용 예시

### `/api_root/WellbeingLog/` 사용 시나리오

**언제 사용?**
- 모든 로그를 확인하고 싶을 때
- 특정 로그를 조회/수정/삭제하고 싶을 때
- 새로운 로그를 생성할 때

**예시:**
```python
# 모든 로그 조회
GET http://127.0.0.1:8000/api_root/WellbeingLog/

# 특정 로그 조회
GET http://127.0.0.1:8000/api_root/WellbeingLog/1/

# 새 로그 생성
POST http://127.0.0.1:8000/api_root/WellbeingLog/
{
  "dominant_emotion": "happy",
  "dominant_emotion_ratio": 0.75,
  "emotion_counts": {"happy": 10},
  "avg_movement": 15.5
}
```

---

### `/api_root/WellbeingLog/summary/` 사용 시나리오

**언제 사용?**
- 오늘의 웰빙 통계를 보고 싶을 때
- 최근 7일간의 웰빙 요약을 보고 싶을 때
- 대시보드나 차트에 표시할 데이터가 필요할 때

**예시:**
```python
# 요약 정보 조회
GET http://127.0.0.1:8000/api_root/WellbeingLog/summary/

# 결과:
# {
#   "today": { "count": 5, "dominant_emotion": "happy", ... },
#   "last_7_days": { "count": 42, "dominant_emotion": "happy", ... }
# }
```

---

## ✅ 둘 다 RESTful API인가요?

**네, 둘 다 RESTful API입니다!**

1. **`/api_root/WellbeingLog/`**
   - 표준 RESTful API (REST 원칙 준수)
   - HTTP 메서드로 리소스 조작 (GET, POST, PUT, DELETE)
   - 리소스 중심 설계

2. **`/api_root/WellbeingLog/summary/`**
   - 커스텀 RESTful API (REST 원칙 준수)
   - HTTP GET 메서드 사용
   - 리소스의 특정 액션을 나타내는 엔드포인트
   - Django REST Framework의 `@action` 데코레이터 사용

---

## 🧪 실제 테스트

### 브라우저에서 확인

1. **기본 엔드포인트**
   ```
   http://127.0.0.1:8000/api_root/WellbeingLog/
   ```
   → 개별 로그 목록이 JSON 배열로 표시됨

2. **요약 엔드포인트**
   ```
   http://127.0.0.1:8000/api_root/WellbeingLog/summary/
   ```
   → 집계된 요약 정보가 JSON 객체로 표시됨

---

## 📝 코드에서의 구현

### 기본 엔드포인트 (ModelViewSet)
```python
class WellbeingLogViewSet(viewsets.ModelViewSet):
    queryset = WellbeingLog.objects.all()
    serializer_class = WellbeingLogSerializer
    # 자동으로 CRUD 엔드포인트 생성
```

### 요약 엔드포인트 (@action)
```python
@action(detail=False, methods=['get'])
def summary(self, request):
    # 커스텀 로직으로 데이터 집계
    # 오늘/최근 7일 데이터 계산
    return Response(data)
```

---

## 🎯 결론

- **둘 다 RESTful API**입니다
- **첫 번째**는 표준 REST API (CRUD 작업)
- **두 번째**는 커스텀 REST API (집계/통계 작업)
- **용도가 다르므로** 상황에 맞게 사용하면 됩니다!

