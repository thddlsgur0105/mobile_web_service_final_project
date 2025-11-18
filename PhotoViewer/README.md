# PhotoViewer 안드로이드 앱 실행 가이드

## 📋 사전 준비

1. **Android Studio 설치**
   - Android Studio Hedgehog (2023.1.1) 이상 버전 설치
   - [다운로드 링크](https://developer.android.com/studio)

2. **JDK 설치**
   - Android Studio 설치 시 자동 포함되지만, JDK 17 이상 권장

## 🚀 실행 방법

### 1단계: 프로젝트 열기

1. Android Studio 실행
2. **File → Open** 선택
3. `C:\Users\thddl\PhotoBlog\PhotoViewer` 폴더 선택
4. **OK** 클릭하여 프로젝트 열기

### 2단계: Gradle 동기화

- 프로젝트를 열면 자동으로 Gradle 동기화가 시작됩니다
- 상단에 "Gradle Sync" 진행 상황 표시
- 완료될 때까지 대기 (첫 실행 시 5-10분 소요 가능)

### 3단계: 토큰 설정 (선택사항)

인증이 필요한 경우:

1. `app/src/main/java/com/example/photoviewer/MainActivity.java` 파일 열기
2. 34번째 줄의 `token` 변수 수정:
   ```java
   String token = "YOUR_ACTUAL_TOKEN_HERE";
   ```
3. 인증이 필요 없는 경우 이 줄을 주석 처리하거나 빈 문자열로 설정

### 4단계: 에뮬레이터 또는 실제 기기 연결

#### 방법 A: Android 에뮬레이터 사용 (권장)

1. **Tools → Device Manager** 클릭
2. **Create Device** 클릭
3. 기기 선택 (예: Pixel 5)
4. 시스템 이미지 선택 (API 30 이상 권장)
5. **Finish** 클릭
6. 에뮬레이터 실행

#### 방법 B: 실제 안드로이드 기기 사용

1. 기기의 **개발자 옵션** 활성화:
   - 설정 → 휴대전화 정보 → 빌드 번호 7번 탭
2. **USB 디버깅** 활성화:
   - 설정 → 개발자 옵션 → USB 디버깅 ON
3. USB 케이블로 PC에 연결
4. 기기에 "USB 디버깅 허용" 팝업이 나타나면 **허용** 클릭

### 5단계: 앱 실행

1. 상단 툴바에서 실행할 기기 선택 (에뮬레이터 또는 연결된 기기)
2. **Run** 버튼 (▶️) 클릭 또는 `Shift + F10`
3. 앱이 빌드되고 설치됩니다 (첫 실행 시 1-2분 소요)

### 6단계: 서버 연결 확인

1. **Django 서버가 실행 중인지 확인**:
   ```bash
   cd C:\Users\thddl\PhotoBlog
   python manage.py runserver
   ```

2. 앱에서 **"동기화"** 버튼 클릭
3. 이미지가 로드되면 성공!

## ⚙️ 네트워크 설정

### 에뮬레이터 사용 시
- 서버 주소: `http://10.0.2.2:8000` (기본값)
- `10.0.2.2`는 안드로이드 에뮬레이터에서 호스트 PC의 `127.0.0.1`을 가리킵니다

### 실제 기기 사용 시
1. PC와 안드로이드 기기가 **같은 Wi-Fi 네트워크**에 연결되어 있어야 합니다
2. PC의 로컬 IP 주소 확인:
   - Windows: `ipconfig` 명령어 실행
   - `IPv4 주소` 확인 (예: `192.168.0.100`)
3. `MainActivity.java`의 `siteUrl` 수정:
   ```java
   String siteUrl = "http://192.168.0.100:8000";  // 실제 IP로 변경
   ```

## 🐛 문제 해결

### Gradle 동기화 실패
- **File → Invalidate Caches → Invalidate and Restart**
- 인터넷 연결 확인
- Android Studio 업데이트

### 앱이 실행되지 않음
- 에뮬레이터가 완전히 부팅되었는지 확인
- **Build → Clean Project** 후 **Build → Rebuild Project**
- **Run → Run 'app'** 다시 시도

### 이미지가 로드되지 않음
1. Django 서버가 실행 중인지 확인
2. `http://127.0.0.1:8000/api_root/Post/` 브라우저에서 접속 테스트
3. 네트워크 보안 설정 확인 (`network_security_config.xml`)
4. 로그 확인: **View → Tool Windows → Logcat**

### 권한 오류
- Android 10 이상에서는 일부 권한이 런타임에 요청됩니다
- 앱 실행 시 권한 요청 팝업에서 **허용** 클릭

## 📱 주요 기능

- **동기화 버튼**: 서버에서 이미지 목록 다운로드
- **새로운 이미지 게시 버튼**: 업로드 기능 (현재 구현 예정)

## 🔗 관련 파일

- `MainActivity.java`: 메인 액티비티 및 네트워크 통신
- `ImageAdapter.java`: RecyclerView 어댑터
- `activity_main.xml`: 메인 레이아웃
- `item_image.xml`: 이미지 아이템 레이아웃
- `AndroidManifest.xml`: 권한 및 앱 설정
- `network_security_config.xml`: 네트워크 보안 설정


