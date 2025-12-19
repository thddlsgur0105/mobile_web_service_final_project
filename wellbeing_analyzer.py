import time
import os
import json
from collections import Counter, deque
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import requests
import torch
from deepface import DeepFace


# =========================
# 설정
# =========================

# YOLOv5 모델 이름 또는 경로
# - 인터넷이 가능하면: "yolov5s" (torch.hub에서 다운로드)
# - 로컬 yolov5 레포를 쓰고 싶으면: torch.hub.load("yolov5", "custom", path="yolov5s.pt")
YOLO_MODEL_NAME = "yolov5s"

# Django 서버 엔드포인트 (로컬 개발 기준)
DJANGO_BASE_URL = "http://127.0.0.1:8000"
WELLBEING_API_URL = f"{DJANGO_BASE_URL}/api_root/WellbeingLog/"
POST_API_URL = f"{DJANGO_BASE_URL}/api_root/Post/"
REGISTER_API_URL = f"{DJANGO_BASE_URL}/api_root/register/"
LOGIN_API_URL = f"{DJANGO_BASE_URL}/api-token-auth/"

# 로그인 정보 저장 경로 (localhost)
AUTH_INFO_FILE = os.path.join(os.path.dirname(__file__), "auth_info.json")

# Post 게시 설정
ENABLE_POST_API = True  # Post API 사용 여부
POST_INTERVAL = 300  # Post 게시 주기 (초, 기본 5분)
POST_ON_NEW_OBJECT = True  # 새 객체가 검출되면 즉시 게시

# 요약 전송 주기 (초)
SUMMARY_INTERVAL = 60  # 60초마다 한 번 서버로 요약 전송

# 윈도우 크기 (최근 N초 데이터만 집계)
WINDOW_SECONDS = 60
FRAME_FPS_ASSUMPTION = 2  # 초당 2프레임 정도로 본다고 가정


def load_yolo_model():
    """
    YOLOv5 모델 로드 (MS COCO 80가지 객체 검출).
    torch.hub을 사용하며, 최초 1회는 인터넷에서 모델을 내려받습니다.
    """
    print("YOLOv5 모델 로딩 중... (MS COCO 80가지 객체 검출 가능)")
    model = torch.hub.load("ultralytics/yolov5", YOLO_MODEL_NAME, pretrained=True)
    model.conf = 0.4  # confidence threshold
    # model.classes = [0]  # 주석 처리: 모든 80가지 객체 검출 가능
    # COCO 클래스: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, 
    # traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, 
    # sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, 
    # suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, 
    # skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, 
    # bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, 
    # cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, 
    # remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, 
    # clock, vase, scissors, teddy bear, hair drier, toothbrush
    return model


def estimate_movement(prev_boxes, curr_boxes):
    """
    간단한 활동성 지표:
    이전 프레임 대비 사람 bbox 중심 이동 평균 거리를 계산합니다.
    """
    if not prev_boxes or not curr_boxes:
        return 0.0

    prev_centers = np.array(
        [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for x1, y1, x2, y2 in prev_boxes]
    )
    curr_centers = np.array(
        [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for x1, y1, x2, y2 in curr_boxes]
    )

    n = min(len(prev_centers), len(curr_centers))
    if n == 0:
        return 0.0

    prev_centers = prev_centers[:n]
    curr_centers = curr_centers[:n]

    diffs = np.linalg.norm(curr_centers - prev_centers, axis=1)
    return float(np.mean(diffs))


def analyze_emotion(face_img):
    """
    DeepFace를 사용해 얼굴 감정을 분석합니다.
    실패 시 None을 반환합니다.
    """
    try:
        result = DeepFace.analyze(
            face_img, actions=["emotion"], enforce_detection=False
        )
        if isinstance(result, list):
            result = result[0]
        return result.get("dominant_emotion")
    except Exception as e:
        print("Emotion analyze error:", e)
        return None


def save_auth_info(username, token):
    """
    로그인 정보를 localhost 파일에 저장합니다.
    
    Args:
        username: 사용자명
        token: 인증 토큰
    """
    auth_data = {
        "username": username,
        "token": token,
        "saved_at": datetime.now().isoformat()
    }
    try:
        with open(AUTH_INFO_FILE, 'w', encoding='utf-8') as f:
            json.dump(auth_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 로그인 정보가 저장되었습니다: {AUTH_INFO_FILE}")
        return True
    except Exception as e:
        print(f"❌ 로그인 정보 저장 실패: {e}")
        return False


def load_auth_info():
    """
    저장된 로그인 정보를 localhost 파일에서 불러옵니다.
    
    Returns:
        tuple: (username, token) 또는 (None, None)
    """
    if not os.path.exists(AUTH_INFO_FILE):
        return None, None
    
    try:
        with open(AUTH_INFO_FILE, 'r', encoding='utf-8') as f:
            auth_data = json.load(f)
        return auth_data.get("username"), auth_data.get("token")
    except Exception as e:
        print(f"❌ 로그인 정보 불러오기 실패: {e}")
        return None, None


def register_user(username, email, password, password_confirm):
    """
    회원가입 API 호출
    
    Args:
        username: 사용자명
        email: 이메일
        password: 비밀번호
        password_confirm: 비밀번호 확인
    
    Returns:
        tuple: (success, token) 또는 (False, error_message)
    """
    try:
        data = {
            "username": username,
            "email": email,
            "password": password,
            "password_confirm": password_confirm
        }
        response = requests.post(REGISTER_API_URL, json=data, timeout=10)
        
        if response.status_code == 201:
            result = response.json()
            token = result.get("token")
            if token:
                save_auth_info(username, token)
                return True, token
            return True, None
        else:
            error_msg = response.json()
            if isinstance(error_msg, dict):
                # 에러 메시지 추출
                errors = []
                for key, value in error_msg.items():
                    if isinstance(value, list):
                        errors.extend(value)
                    else:
                        errors.append(str(value))
                error_msg = ", ".join(errors)
            return False, error_msg
    except Exception as e:
        return False, str(e)


def login_user(username, password):
    """
    로그인 API 호출 및 토큰 저장
    
    Args:
        username: 사용자명
        password: 비밀번호
    
    Returns:
        tuple: (success, token) 또는 (False, error_message)
    """
    try:
        data = {
            "username": username,
            "password": password
        }
        response = requests.post(LOGIN_API_URL, json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            token = result.get("token")
            if token:
                save_auth_info(username, token)
                return True, token
            return False, "토큰을 받지 못했습니다."
        else:
            error_msg = response.json()
            if isinstance(error_msg, dict):
                non_field_errors = error_msg.get("non_field_errors", [])
                if non_field_errors:
                    error_msg = non_field_errors[0]
                else:
                    error_msg = str(error_msg)
            return False, error_msg
    except Exception as e:
        return False, str(e)


def verify_token(token):
    """
    토큰이 유효한지 확인합니다.
    
    Args:
        token: 인증 토큰
    
    Returns:
        bool: 토큰이 유효하면 True
    """
    if not token:
        return False
    
    try:
        headers = {"Authorization": f"Token {token}"}
        # 간단한 API 호출로 토큰 검증 (WellbeingLog 목록 조회)
        response = requests.get(WELLBEING_API_URL, headers=headers, timeout=5)
        return response.status_code in [200, 201]
    except:
        return False


def get_auth_token():
    """
    저장된 토큰을 불러오고, 유효성을 확인합니다.
    토큰이 없거나 만료되었으면 None을 반환합니다.
    
    Returns:
        str: 인증 토큰 또는 None
    """
    username, token = load_auth_info()
    
    if token:
        # 토큰 유효성 확인
        if verify_token(token):
            return token
        else:
            print("⚠️ 저장된 토큰이 만료되었습니다. 다시 로그인해주세요.")
            # 만료된 토큰 정보 삭제
            if os.path.exists(AUTH_INFO_FILE):
                os.remove(AUTH_INFO_FILE)
            return None
    else:
        return None


def send_summary_to_server(emotion_window, movement_window):
    """
    최근 윈도우의 감정/활동성을 집계해 Django 서버로 전송합니다.
    """
    if not emotion_window:
        return

    emo_counts = Counter(emotion_window)
    total = sum(emo_counts.values())
    dominant_emotion, dominant_count = emo_counts.most_common(1)[0]
    dominant_ratio = dominant_count / total if total > 0 else 0.0

    avg_movement = float(np.mean(movement_window)) if movement_window else 0.0

    payload = {
        "dominant_emotion": dominant_emotion,
        "dominant_emotion_ratio": dominant_ratio,
        "emotion_counts": dict(emo_counts),
        "avg_movement": avg_movement,
        "timestamp": time.time(),
    }

    headers = {"Content-Type": "application/json"}
    
    # 저장된 토큰 사용
    token = get_auth_token()
    if token:
        headers["Authorization"] = f"Token {token}"
    else:
        print("⚠️ 인증 토큰이 없어 요청이 실패할 수 있습니다.")

    try:
        resp = requests.post(
            WELLBEING_API_URL, json=payload, headers=headers, timeout=5
        )
        if resp.status_code in [200, 201]:
            print("✅ WellbeingLog 전송 성공")
        else:
            print(f"⚠️ WellbeingLog 전송 실패 ({resp.status_code}): {resp.text[:200]}")
    except Exception as e:
        print("서버 전송 오류:", e)


def send_post_to_server(frame, detected_objects, title=None, text=None):
    """
    검출된 객체 정보를 포함한 이미지를 Post API로 Django 서버에 게시합니다.
    
    Args:
        frame: OpenCV 이미지 (numpy array)
        detected_objects: 검출된 객체 딕셔너리 {object_name: count}
        title: 포스트 제목 (None이면 자동 생성)
        text: 포스트 내용 (None이면 자동 생성)
    """
    if not ENABLE_POST_API:
        return
    
    if not detected_objects:
        return
    
    try:
        # 제목과 내용 자동 생성
        if title is None:
            obj_list = ", ".join([f"{name}({count})" for name, count in list(detected_objects.items())[:5]])
            title = f"Detected Objects: {obj_list}"
        
        if text is None:
            text = f"Detected objects at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            text += "Objects detected:\n"
            for obj_name, count in sorted(detected_objects.items(), key=lambda x: x[1], reverse=True):
                text += f"- {obj_name}: {count}\n"
        
        # 이미지를 JPEG 형식으로 인코딩
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        
        # 파일 업로드를 위한 multipart/form-data 준비
        files = {
            'image': ('detection.jpg', BytesIO(img_bytes), 'image/jpeg')
        }
        
        data = {
            'title': title,
            'text': text,
        }
        
        headers = {}
        
        # 저장된 토큰 사용
        token = get_auth_token()
        if token:
            headers['Authorization'] = f"Token {token}"
        else:
            print("⚠️ 인증 토큰이 없어 요청이 실패할 수 있습니다.")
        
        # Post API로 전송
        resp = requests.post(
            POST_API_URL,
            data=data,
            files=files,
            headers=headers,
            timeout=10
        )
        
        if resp.status_code in [200, 201]:
            print(f"✅ Post 게시 성공: {title}")
        else:
            print(f"⚠️ Post 게시 실패 ({resp.status_code}): {resp.text[:200]}")
            
    except Exception as e:
        print(f"❌ Post 게시 오류: {e}")


def main():
    # 로그인 확인
    print("="*60)
    print("  사용자 인증 확인")
    print("="*60)
    
    username, token = load_auth_info()
    
    # 토큰이 없거나 만료된 경우
    if not token or not verify_token(token):
        if token:
            print("⚠️ 저장된 토큰이 만료되었습니다.")
            if os.path.exists(AUTH_INFO_FILE):
                os.remove(AUTH_INFO_FILE)
        
        print("\n로그인이 필요합니다.")
        print("\n[1] 회원가입")
        print("[2] 로그인")
        choice = input("\n선택하세요 (1 또는 2): ").strip()
        
        if choice == "1":
            print("\n=== 회원가입 ===")
            username = input("사용자명: ").strip()
            email = input("이메일 (선택사항): ").strip() or ""
            password = input("비밀번호: ").strip()
            password_confirm = input("비밀번호 확인: ").strip()
            
            success, result = register_user(username, email, password, password_confirm)
            if success:
                print(f"✅ 회원가입 성공! 토큰이 저장되었습니다.")
                token = result
            else:
                print(f"❌ 회원가입 실패: {result}")
                return
        
        elif choice == "2":
            print("\n=== 로그인 ===")
            username = input("사용자명: ").strip()
            password = input("비밀번호: ").strip()
            
            success, result = login_user(username, password)
            if success:
                print(f"✅ 로그인 성공! 토큰이 저장되었습니다.")
                token = result
            else:
                print(f"❌ 로그인 실패: {result}")
                return
        else:
            print("❌ 잘못된 선택입니다.")
            return
    else:
        print(f"✅ 저장된 로그인 정보 사용: {username}")
    
    print("="*60)
    print()
    
    yolo = load_yolo_model()

    print("웹캠 열기...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    emotion_window = deque(maxlen=WINDOW_SECONDS * FRAME_FPS_ASSUMPTION)
    movement_window = deque(maxlen=WINDOW_SECONDS * FRAME_FPS_ASSUMPTION)
    prev_person_boxes = []
    last_summary_time = time.time()
    last_post_time = time.time()
    prev_detected_objects = set()  # 이전 프레임에서 검출된 객체 추적

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다. 종료합니다.")
            break

        results = yolo(frame)
        det = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]
        
        # COCO 클래스 이름 가져오기
        class_names = results.names  # {0: 'person', 1: 'bicycle', ...}

        person_boxes = []
        detected_objects = {}  # 검출된 객체 카운트

        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, xyxy)
            cls_id = int(cls)
            cls_name = class_names.get(cls_id, f"class_{cls_id}")
            conf_score = float(conf)
            
            # 검출된 객체 카운트
            detected_objects[cls_name] = detected_objects.get(cls_name, 0) + 1
            
            # 사람(class 0)인 경우에만 감정 분석 수행
            if cls_id == 0:  # person
                person_boxes.append((x1, y1, x2, y2))

                # 상반신 상단 부분을 얼굴로 간주한 간단한 crop
                h = y2 - y1
                face_y2 = y1 + int(h * 0.6)
                face_img = frame[y1:face_y2, x1:x2]
                if face_img.size > 0:
                    emo = analyze_emotion(face_img)
                    if emo:
                        emotion_window.append(emo)
                    
                    # 사람 박스는 초록색, 감정 표시
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {conf_score:.2f}"
                    if emo:
                        label += f" [{emo}]"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
            else:
                # 사람이 아닌 다른 객체들은 파란색으로 표시
                color = (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf_score:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

        movement = estimate_movement(prev_person_boxes, person_boxes)
        movement_window.append(movement)
        prev_person_boxes = person_boxes

        # 화면에 정보 표시
        y_offset = 30
        cv2.putText(
            frame,
            f"movement: {movement:.1f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        
        # 검출된 객체 목록 표시 (최대 5개)
        y_offset += 30
        detected_list = list(detected_objects.items())[:5]
        for i, (obj_name, count) in enumerate(detected_list):
            cv2.putText(
                frame,
                f"{obj_name}: {count}",
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        cv2.imshow("Wellbeing Analyzer", frame)

        # ESC키로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

        now = time.time()
        
        # WellbeingLog 전송 (주기적)
        if now - last_summary_time > SUMMARY_INTERVAL:
            send_summary_to_server(emotion_window, movement_window)
            last_summary_time = now
        
        # Post 게시 처리
        if ENABLE_POST_API:
            current_objects = set(detected_objects.keys())
            
            # 새 객체가 검출되었을 때 즉시 게시
            if POST_ON_NEW_OBJECT:
                new_objects = current_objects - prev_detected_objects
                if new_objects:
                    print(f"🆕 새 객체 검출: {', '.join(new_objects)}")
                    send_post_to_server(frame.copy(), detected_objects)
                    prev_detected_objects = current_objects.copy()
            
            # 주기적으로 게시 (검출된 객체가 있을 때만)
            if now - last_post_time > POST_INTERVAL and detected_objects:
                send_post_to_server(frame.copy(), detected_objects)
                last_post_time = now
            
            # prev_detected_objects 업데이트
            prev_detected_objects = current_objects.copy()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


