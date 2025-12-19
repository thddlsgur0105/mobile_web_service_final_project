"""
MS COCO 80가지 객체 검출 테스트 스크립트
YOLOv5 모델이 모든 COCO 클래스를 검출할 수 있는지 확인합니다.
"""

import sys
import io

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import cv2

def get_coco_classes():
    """MS COCO 80가지 클래스 목록 반환"""
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    return coco_classes

def load_yolo_model():
    """YOLOv5 모델 로드 (wellbeing_analyzer.py와 동일한 방식)"""
    print("YOLOv5 모델 로딩 중...")
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.conf = 0.25  # confidence threshold
    return model

def print_coco_classes():
    """COCO 80가지 클래스 목록 출력"""
    classes = get_coco_classes()
    print("\n" + "="*60)
    print("MS COCO 80가지 객체 클래스 목록")
    print("="*60)
    for i, cls_name in enumerate(classes, 1):
        print(f"{i:2d}. {cls_name}")
    print("="*60)
    print(f"총 {len(classes)}가지 클래스\n")

def test_model_classes(model):
    """모델이 지원하는 클래스 확인"""
    print("\n" + "="*60)
    print("YOLOv5 모델이 지원하는 클래스 확인")
    print("="*60)
    
    # 모델의 클래스 이름 가져오기
    class_names = model.names
    
    print(f"모델이 지원하는 클래스 수: {len(class_names)}")
    print("\n클래스 ID와 이름:")
    for cls_id, cls_name in class_names.items():
        print(f"  ID {cls_id:2d}: {cls_name}")
    
    # COCO 클래스와 비교
    coco_classes = get_coco_classes()
    print(f"\nCOCO 표준 클래스 수: {len(coco_classes)}")
    
    # 일치 여부 확인
    if len(class_names) == len(coco_classes):
        print("✅ 모델이 모든 COCO 80가지 클래스를 지원합니다!")
    else:
        print(f"⚠️  클래스 수가 다릅니다. (모델: {len(class_names)}, COCO: {len(coco_classes)})")
    
    print("="*60 + "\n")

def test_webcam_detection():
    """웹캠을 사용한 실시간 객체 검출 테스트"""
    print("\n" + "="*60)
    print("웹캠 실시간 객체 검출 테스트")
    print("="*60)
    print("ESC 키를 누르면 종료됩니다.\n")
    
    model = load_yolo_model()
    test_model_classes(model)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        return
    
    print("웹캠 연결 성공! 검출 시작...\n")
    
    detected_history = {}  # 검출된 모든 객체 기록
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 객체 검출 (wellbeing_analyzer.py와 동일한 방식)
        results = model(frame)
        det = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]
        
        # COCO 클래스 이름 가져오기
        class_names = results.names  # {0: 'person', 1: 'bicycle', ...}
        
        # 현재 프레임에서 검출된 객체
        current_detections = {}
        
        for *xyxy, conf, cls in det:
            cls_id = int(cls)
            cls_name = class_names.get(cls_id, f"class_{cls_id}")
            conf_score = float(conf)
            
            # 검출 기록
            if cls_name not in detected_history:
                detected_history[cls_name] = []
            detected_history[cls_name].append(conf_score)
            current_detections[cls_name] = current_detections.get(cls_name, 0) + 1
            
            # 바운딩 박스 그리기
            x1, y1, x2, y2 = map(int, xyxy)
            
            # 사람은 초록색, 나머지는 파란색
            color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 라벨 표시
            label = f"{cls_name} {conf_score:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        
        # 화면 좌측에 검출 정보 표시
        y_offset = 30
        cv2.putText(
            frame,
            f"Detected Objects: {len(current_detections)}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        
        y_offset += 30
        for i, (obj_name, count) in enumerate(list(current_detections.items())[:10]):
            cv2.putText(
                frame,
                f"{obj_name}: {count}",
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        
        # 우측 상단에 총 검출된 클래스 수 표시
        total_classes_detected = len(detected_history)
        cv2.putText(
            frame,
            f"Total Classes Detected: {total_classes_detected}/80",
            (frame.shape[1] - 350, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        
        cv2.imshow("COCO 80 Classes Detection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 검출 결과 요약
    print("\n" + "="*60)
    print("검출 결과 요약")
    print("="*60)
    print(f"총 {len(detected_history)}가지 서로 다른 클래스가 검출되었습니다:\n")
    
    for cls_name in sorted(detected_history.keys()):
        confidences = detected_history[cls_name]
        avg_conf = sum(confidences) / len(confidences)
        print(f"  - {cls_name}: {len(confidences)}회 검출 (평균 신뢰도: {avg_conf:.2f})")
    
    print("\n" + "="*60)

def test_image_detection(image_path):
    """이미지 파일을 사용한 객체 검출 테스트"""
    print(f"\n이미지 파일 검출 테스트: {image_path}")
    
    model = load_yolo_model()
    test_model_classes(model)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 이미지를 불러올 수 없습니다: {image_path}")
        return
    
    results = model(img)
    det = results.xyxy[0].cpu().numpy()
    class_names = results.names
    
    detected_objects = {}
    for *xyxy, conf, cls in det:
        cls_id = int(cls)
        cls_name = class_names.get(cls_id, f"class_{cls_id}")
        conf_score = float(conf)
        
        detected_objects[cls_name] = detected_objects.get(cls_name, 0) + 1
        
        # 바운딩 박스 그리기
        x1, y1, x2, y2 = map(int, xyxy)
        color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = f"{cls_name} {conf_score:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 5, 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    print(f"\n검출된 객체: {len(detected_objects)}가지")
    for cls_name, count in detected_objects.items():
        print(f"  - {cls_name}: {count}개")
    
    cv2.imshow("Detection Result", img)
    print("\n아무 키나 누르면 종료됩니다...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("MS COCO 80가지 객체 검출 테스트")
    print("="*60)
    
    # COCO 클래스 목록 출력
    print_coco_classes()
    
    if len(sys.argv) > 1:
        # 이미지 파일이 제공된 경우
        image_path = sys.argv[1]
        test_image_detection(image_path)
    else:
        # 웹캠 테스트
        print("웹캠 테스트를 시작합니다...")
        print("(이미지 파일을 테스트하려면: python test_coco_detection.py <이미지경로>)\n")
        test_webcam_detection()

