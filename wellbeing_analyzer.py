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

# MediaPipe import (선택적)
MEDIAPIPE_AVAILABLE = False
mp = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ MediaPipe를 import할 수 없습니다: {e}")
    print("   설치 방법: pip install mediapipe")
except Exception as e:
    print(f"⚠️ MediaPipe 초기화 오류: {e}")


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


def estimate_pose(pose_landmarks, image_height, image_width):
    """
    MediaPipe Pose를 사용하여 자세를 추정합니다.
    MediaPipe 0.10+ tasks API 사용
    Returns: 'sitting', 'standing', 'bending', or None
    """
    if not pose_landmarks or len(pose_landmarks) == 0:
        return None
    
    try:
        # 주요 랜드마크 인덱스
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        
        # 랜드마크 좌표 추출 (tasks API는 리스트 형태)
        landmarks = pose_landmarks
        
        def get_landmark(idx):
            if idx >= len(landmarks):
                return None
            lm = landmarks[idx]
            # tasks API는 x, y, z 속성을 가짐
            return (lm.x * image_width, lm.y * image_height)
        
        # 어깨와 엉덩이 중심점
        left_shoulder = get_landmark(LEFT_SHOULDER)
        right_shoulder = get_landmark(RIGHT_SHOULDER)
        left_hip = get_landmark(LEFT_HIP)
        right_hip = get_landmark(RIGHT_HIP)
        left_knee = get_landmark(LEFT_KNEE)
        right_knee = get_landmark(RIGHT_KNEE)
        left_ankle = get_landmark(LEFT_ANKLE)
        right_ankle = get_landmark(RIGHT_ANKLE)
        
        # 필수 랜드마크 확인
        if not all([left_shoulder, right_shoulder, left_hip, right_hip, 
                   left_knee, right_knee, left_ankle, right_ankle]):
            return None
        
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        knee_center_y = (left_knee[1] + right_knee[1]) / 2
        ankle_center_y = (left_ankle[1] + right_ankle[1]) / 2
        
        # 어깨와 엉덩이 사이 거리
        torso_height = abs(hip_center_y - shoulder_center_y)
        
        # 발목이 무릎보다 아래에 있는지 확인 (서 있는지)
        is_standing = ankle_center_y > knee_center_y
        
        # 무릎과 발목의 높이 차이 (다리가 펴져 있는지)
        leg_extended = (ankle_center_y - knee_center_y) > (knee_center_y - hip_center_y) * 0.5
        
        # 상체가 앞으로 기울어져 있는지 (숙여 있는지)
        # 어깨와 엉덩이의 수직 거리와 수평 거리를 비교
        shoulder_hip_vertical = abs(hip_center_y - shoulder_center_y)
        shoulder_hip_horizontal = abs((left_hip[0] + right_hip[0]) / 2 - (left_shoulder[0] + right_shoulder[0]) / 2)
        
        # 각도 계산 (수평 거리가 너무 작으면 나누기 오류 방지)
        if shoulder_hip_horizontal < 1:
            shoulder_hip_horizontal = 1
        
        # 상체 기울기 각도 (0도 = 완전히 똑바름, 90도 = 완전히 수평)
        tilt_angle = np.arctan2(shoulder_hip_vertical, shoulder_hip_horizontal) * 180 / np.pi
        
        # 상체가 앞으로 기울어졌는지 (어깨가 엉덩이보다 앞에 있는지)
        shoulder_forward = (left_shoulder[0] + right_shoulder[0]) / 2 < (left_hip[0] + right_hip[0]) / 2
        
        # 자세 판단
        if not is_standing or not leg_extended:
            # 다리가 펴져 있지 않거나 발목이 무릎 위에 있으면 앉아 있음
            return 'sitting'
        elif tilt_angle < 60 and shoulder_forward:
            # 상체가 기울어지고 앞으로 숙여졌으면 bending
            # (60도 미만이면 상체가 많이 기울어짐)
            return 'bending'
        else:
            # 그 외는 서 있음
            return 'standing'
    except Exception as e:
        print(f"Pose estimation error: {e}")
        return None


def estimate_head_pose(face_landmarks, image_width, image_height):
    """
    MediaPipe Face Mesh를 사용하여 Head Pose를 추정합니다.
    MediaPipe 0.10+ tasks API 사용
    Returns: {"pitch": float, "yaw": float, "roll": float} or None
    """
    if not face_landmarks or len(face_landmarks) == 0:
        return None
    
    try:
        # 얼굴 랜드마크 인덱스 (MediaPipe Face Mesh)
        # 코 끝, 턱, 왼쪽 귀, 오른쪽 귀, 왼쪽 눈, 오른쪽 눈
        NOSE_TIP = 1
        CHIN = 175
        LEFT_EAR = 234
        RIGHT_EAR = 454
        LEFT_EYE = 33
        RIGHT_EYE = 263
        
        # tasks API는 리스트 형태
        landmarks = face_landmarks
        
        def get_landmark_3d(idx):
            if idx >= len(landmarks):
                return None
            lm = landmarks[idx]
            # tasks API는 x, y, z 속성을 가짐
            return np.array([lm.x * image_width, lm.y * image_height, lm.z * image_width])
        
        # 3D 좌표 추출
        nose_tip = get_landmark_3d(NOSE_TIP)
        chin = get_landmark_3d(CHIN)
        left_ear = get_landmark_3d(LEFT_EAR)
        right_ear = get_landmark_3d(RIGHT_EAR)
        left_eye = get_landmark_3d(LEFT_EYE)
        right_eye = get_landmark_3d(RIGHT_EYE)
        
        # 얼굴 중심선 (코-턱)
        face_center = (nose_tip + chin) / 2
        
        # 좌우 귀 중심
        ear_center = (left_ear + right_ear) / 2
        
        # 눈 중심
        eye_center = (left_eye + right_eye) / 2
        
        # Pitch (상하 움직임): 얼굴 중심선과 수직선의 각도
        face_vector = chin - nose_tip
        pitch = np.arctan2(face_vector[1], face_vector[2]) * 180 / np.pi
        
        # Yaw (좌우 움직임): 귀 중심선과 수평선의 각도
        ear_vector = right_ear - left_ear
        yaw = np.arctan2(ear_vector[0], ear_vector[2]) * 180 / np.pi
        
        # Roll (회전): 눈 중심선과 수평선의 각도
        eye_vector = right_eye - left_eye
        roll = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
        
        return {
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll": float(roll)
        }
    except Exception as e:
        print(f"Head pose estimation error: {e}")
        return None


def analyze_eye_blink(face_landmarks, prev_eye_state, frame_count):
    """
    MediaPipe Face Mesh를 사용하여 눈 깜빡임을 감지하고 집중도/피로도를 계산합니다.
    MediaPipe 0.10+ tasks API 사용
    Returns: (blink_count, focus_level, fatigue_level, new_eye_state)
    """
    if not face_landmarks or len(face_landmarks) == 0:
        return 0, 0.0, 0.0, prev_eye_state
    
    try:
        # 눈 랜드마크 인덱스 (MediaPipe Face Mesh)
        # 왼쪽 눈: 상단 159, 하단 145, 좌측 33, 우측 133
        # 오른쪽 눈: 상단 386, 하단 374, 좌측 362, 우측 263
        LEFT_EYE_TOP = 159
        LEFT_EYE_BOTTOM = 145
        LEFT_EYE_LEFT = 33
        LEFT_EYE_RIGHT = 133
        
        RIGHT_EYE_TOP = 386
        RIGHT_EYE_BOTTOM = 374
        RIGHT_EYE_LEFT = 362
        RIGHT_EYE_RIGHT = 263
        
        # tasks API는 리스트 형태
        landmarks = face_landmarks
        
        def get_landmark(idx):
            if idx >= len(landmarks):
                return None
            lm = landmarks[idx]
            # tasks API는 x, y 속성을 가짐
            return (lm.x, lm.y)
        
        # 왼쪽 눈 좌표
        left_eye_top = get_landmark(LEFT_EYE_TOP)
        left_eye_bottom = get_landmark(LEFT_EYE_BOTTOM)
        left_eye_left = get_landmark(LEFT_EYE_LEFT)
        left_eye_right = get_landmark(LEFT_EYE_RIGHT)
        
        # 오른쪽 눈 좌표
        right_eye_top = get_landmark(RIGHT_EYE_TOP)
        right_eye_bottom = get_landmark(RIGHT_EYE_BOTTOM)
        right_eye_left = get_landmark(RIGHT_EYE_LEFT)
        right_eye_right = get_landmark(RIGHT_EYE_RIGHT)
        
        # 눈의 높이와 너비 계산
        left_eye_height = abs(left_eye_top[1] - left_eye_bottom[1])
        left_eye_width = abs(left_eye_right[0] - left_eye_left[0])
        right_eye_height = abs(right_eye_top[1] - right_eye_bottom[1])
        right_eye_width = abs(right_eye_right[0] - right_eye_left[0])
        
        # 눈 종횡비 (EAR: Eye Aspect Ratio)
        left_ear = left_eye_height / (left_eye_width + 1e-6)
        right_ear = right_eye_height / (right_eye_width + 1e-6)
        avg_ear = (left_ear + right_ear) / 2
        
        # 눈 깜빡임 감지 (EAR이 임계값 이하일 때)
        EAR_THRESHOLD = 0.25
        is_blinking = avg_ear < EAR_THRESHOLD
        
        # 이전 상태와 비교하여 깜빡임 카운트
        blink_count = 0
        if prev_eye_state is not None:
            if not prev_eye_state['was_blinking'] and is_blinking:
                blink_count = 1
        
        # 집중도 계산 (눈이 열려있고 안정적일 때 높음)
        # EAR이 정상 범위(0.25~0.35)에 있고 안정적이면 집중도 높음
        if 0.25 <= avg_ear <= 0.35:
            focus_level = min(1.0, avg_ear / 0.3)
        else:
            focus_level = max(0.0, 1.0 - abs(avg_ear - 0.3) * 2)
        
        # 피로도 계산 (눈 깜빡임 빈도가 낮거나 눈이 자주 감기면 피로도 높음)
        # 프레임당 깜빡임 빈도가 낮으면 피로도 증가
        if prev_eye_state:
            time_since_last_blink = frame_count - prev_eye_state.get('last_blink_frame', 0)
            # 3초 이상 깜빡임이 없으면 피로도 증가
            if time_since_last_blink > 90:  # 약 3초 (30fps 기준)
                fatigue_level = min(1.0, (time_since_last_blink - 90) / 180)
            else:
                fatigue_level = max(0.0, 1.0 - time_since_last_blink / 90)
        else:
            fatigue_level = 0.0
        
        # 현재 상태 저장
        current_state = {
            'was_blinking': is_blinking,
            'last_blink_frame': frame_count if is_blinking else prev_eye_state.get('last_blink_frame', 0) if prev_eye_state else 0,
            'avg_ear': avg_ear
        }
        
        return blink_count, focus_level, fatigue_level, current_state
    except Exception as e:
        print(f"Eye blink analysis error: {e}")
        return 0, 0.0, 0.0, None


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


def send_summary_to_server(emotion_window, movement_window, pose_window, head_pose_window, 
                           blink_count_window, focus_window, fatigue_window, frame=None):
    """
    최근 윈도우의 감정/활동성/자세/고개/눈 상태를 집계해 Django 서버로 전송합니다.
    이미지도 함께 전송하여 데이터베이스에 저장합니다.
    
    Args:
        frame: OpenCV 이미지 (numpy array), None이면 이미지 없이 전송
    """
    if not emotion_window:
        return

    emo_counts = Counter(emotion_window)
    total = sum(emo_counts.values())
    dominant_emotion, dominant_count = emo_counts.most_common(1)[0]
    dominant_ratio = dominant_count / total if total > 0 else 0.0

    avg_movement = float(np.mean(movement_window)) if movement_window else 0.0
    
    # 자세 집계 (가장 빈도 높은 자세)
    dominant_pose = None
    if pose_window:
        pose_counts = Counter(pose_window)
        dominant_pose = pose_counts.most_common(1)[0][0] if pose_counts else None
    
    # Head Pose 평균 계산
    avg_head_pose = None
    if head_pose_window:
        valid_poses = [p for p in head_pose_window if p is not None]
        if valid_poses:
            avg_pitch = np.mean([p.get('pitch', 0) for p in valid_poses])
            avg_yaw = np.mean([p.get('yaw', 0) for p in valid_poses])
            avg_roll = np.mean([p.get('roll', 0) for p in valid_poses])
            avg_head_pose = {
                "pitch": float(avg_pitch),
                "yaw": float(avg_yaw),
                "roll": float(avg_roll)
            }
    
    # 눈 깜빡임 총합
    total_blinks = sum(blink_count_window) if blink_count_window else 0
    
    # 집중도와 피로도 평균
    avg_focus = float(np.mean(focus_window)) if focus_window else 0.0
    avg_fatigue = float(np.mean(fatigue_window)) if fatigue_window else 0.0

    # 저장된 토큰 사용
    token = get_auth_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Token {token}"
    else:
        print("⚠️ 인증 토큰이 없어 요청이 실패할 수 있습니다.")

    try:
        # 이미지가 있으면 multipart/form-data로 전송
        if frame is not None:
            # 이미지를 JPEG 형식으로 인코딩
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_bytes = img_encoded.tobytes()
            
            # multipart/form-data로 전송
            files = {
                'image': ('wellbeing_image.jpg', BytesIO(img_bytes), 'image/jpeg')
            }
            
            data = {
                "dominant_emotion": dominant_emotion,
                "dominant_emotion_ratio": str(dominant_ratio),
                "emotion_counts": json.dumps(dict(emo_counts), ensure_ascii=False),
                "avg_movement": str(avg_movement),
                "pose": dominant_pose or "",
                "head_pose": json.dumps(avg_head_pose, ensure_ascii=False) if avg_head_pose else "{}",
                "eye_blink_count": str(total_blinks),
                "focus_level": str(avg_focus),
                "fatigue_level": str(avg_fatigue),
            }
            
            resp = requests.post(
                WELLBEING_API_URL, data=data, files=files, headers=headers, timeout=10
            )
        else:
            # 이미지가 없으면 JSON으로 전송 (기존 방식)
            payload = {
                "dominant_emotion": dominant_emotion,
                "dominant_emotion_ratio": dominant_ratio,
                "emotion_counts": dict(emo_counts),
                "avg_movement": avg_movement,
                "pose": dominant_pose,
                "head_pose": avg_head_pose,
                "eye_blink_count": total_blinks,
                "focus_level": avg_focus,
                "fatigue_level": avg_fatigue,
                "timestamp": time.time(),
            }
            headers["Content-Type"] = "application/json"
            resp = requests.post(
                WELLBEING_API_URL, json=payload, headers=headers, timeout=5
            )
        
        if resp.status_code in [200, 201]:
            print("✅ WellbeingLog 전송 성공 (이미지 포함)" if frame is not None else "✅ WellbeingLog 전송 성공")
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
    global MEDIAPIPE_AVAILABLE
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
    
    # MediaPipe 초기화 (tasks API)
    pose_landmarker = None
    face_landmarker = None
    
    if MEDIAPIPE_AVAILABLE:
        try:
            print("MediaPipe 모델 로딩 중...")
            from mediapipe.tasks.python import vision
            import urllib.request
            import tempfile
            
            # 모델 파일 다운로드 경로
            model_dir = os.path.join(os.path.dirname(__file__), "mediapipe_models")
            os.makedirs(model_dir, exist_ok=True)
            
            # Pose Landmarker 모델 파일
            pose_model_path = os.path.join(model_dir, "pose_landmarker.task")
            if not os.path.exists(pose_model_path):
                print("   Pose 모델 다운로드 중...")
                pose_model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
                urllib.request.urlretrieve(pose_model_url, pose_model_path)
            
            # Face Landmarker 모델 파일
            face_model_path = os.path.join(model_dir, "face_landmarker.task")
            if not os.path.exists(face_model_path):
                print("   Face 모델 다운로드 중...")
                face_model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                urllib.request.urlretrieve(face_model_url, face_model_path)
            
            # Pose Landmarker 초기화
            base_options = python.BaseOptions(model_asset_path=pose_model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            pose_landmarker = vision.PoseLandmarker.create_from_options(options)
            
            # Face Landmarker 초기화
            face_base_options = python.BaseOptions(model_asset_path=face_model_path)
            face_options = vision.FaceLandmarkerOptions(
                base_options=face_base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
            
            print("✅ MediaPipe 모델 로딩 완료")
        except Exception as e:
            print(f"⚠️ MediaPipe 초기화 실패: {e}")
            print(f"   오류 상세: {type(e).__name__}: {str(e)}")
            MEDIAPIPE_AVAILABLE = False
            pose_landmarker = None
            face_landmarker = None
    else:
        print("⚠️ MediaPipe를 사용할 수 없어 자세/Head Pose/눈 분석이 비활성화됩니다.")

    print("웹캠 열기...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    emotion_window = deque(maxlen=WINDOW_SECONDS * FRAME_FPS_ASSUMPTION)
    movement_window = deque(maxlen=WINDOW_SECONDS * FRAME_FPS_ASSUMPTION)
    pose_window = deque(maxlen=WINDOW_SECONDS * FRAME_FPS_ASSUMPTION)
    head_pose_window = deque(maxlen=WINDOW_SECONDS * FRAME_FPS_ASSUMPTION)
    blink_count_window = deque(maxlen=WINDOW_SECONDS * FRAME_FPS_ASSUMPTION)
    focus_window = deque(maxlen=WINDOW_SECONDS * FRAME_FPS_ASSUMPTION)
    fatigue_window = deque(maxlen=WINDOW_SECONDS * FRAME_FPS_ASSUMPTION)
    
    prev_person_boxes = []
    prev_eye_state = None
    frame_count = 0
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
        
        # MediaPipe 분석 (사람이 검출된 경우)
        current_pose = None
        current_head_pose = None
        blink_count = 0
        focus_level = 0.0
        fatigue_level = 0.0
        
        if person_boxes and MEDIAPIPE_AVAILABLE and pose_landmarker and face_landmarker:
            try:
                # RGB로 변환 (MediaPipe는 RGB를 사용)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]
                
                # MediaPipe Image 생성
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Pose 추정
                pose_detection_result = pose_landmarker.detect(mp_image)
                if pose_detection_result.pose_landmarks and len(pose_detection_result.pose_landmarks) > 0:
                    # 첫 번째 사람의 랜드마크 사용
                    pose_landmarks = pose_detection_result.pose_landmarks[0]
                    current_pose = estimate_pose(pose_landmarks, h, w)
                    if current_pose:
                        pose_window.append(current_pose)
                
                # Face Mesh (Head Pose & Eye Blink)
                face_detection_result = face_landmarker.detect(mp_image)
                if face_detection_result.face_landmarks and len(face_detection_result.face_landmarks) > 0:
                    # 첫 번째 얼굴의 랜드마크 사용
                    face_landmarks = face_detection_result.face_landmarks[0]
                    
                    # Head Pose 추정
                    current_head_pose = estimate_head_pose(face_landmarks, w, h)
                    if current_head_pose:
                        head_pose_window.append(current_head_pose)
                    
                    # 눈 깜빡임 및 집중도/피로도 분석
                    blink_count, focus_level, fatigue_level, new_eye_state = analyze_eye_blink(
                        face_landmarks, prev_eye_state, frame_count
                    )
                    if new_eye_state:
                        prev_eye_state = new_eye_state
                    if blink_count > 0:
                        blink_count_window.append(blink_count)
                    focus_window.append(focus_level)
                    fatigue_window.append(fatigue_level)
            except Exception as e:
                print(f"MediaPipe 분석 오류: {e}")
        
        frame_count += 1

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
        
        # 자세 정보 표시
        if current_pose:
            y_offset += 30
            cv2.putText(
                frame,
                f"pose: {current_pose}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
        
        # 집중도/피로도 표시
        if focus_level > 0 or fatigue_level > 0:
            y_offset += 30
            cv2.putText(
                frame,
                f"focus: {focus_level:.2f} | fatigue: {fatigue_level:.2f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 200, 0),
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
            send_summary_to_server(
                emotion_window, movement_window, pose_window, head_pose_window,
                blink_count_window, focus_window, fatigue_window, frame=frame.copy()
            )
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


