"""
MS COCO 80가지 클래스 목록을 출력하고 YOLOv5 모델과 비교합니다.
"""

import sys
import io

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch

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

def main():
    print("\n" + "="*70)
    print("MS COCO 80가지 객체 클래스 목록")
    print("="*70)
    
    coco_classes = get_coco_classes()
    
    # 클래스 목록 출력
    print("\n[클래스 목록]\n")
    for i, cls_name in enumerate(coco_classes, 1):
        print(f"{i:2d}. {cls_name:20s}", end="")
        if i % 4 == 0:
            print()
    
    if len(coco_classes) % 4 != 0:
        print()
    
    print(f"\n총 {len(coco_classes)}가지 클래스\n")
    
    # YOLOv5 모델 로드 및 확인
    print("="*70)
    print("YOLOv5 모델 클래스 확인")
    print("="*70)
    
    try:
        print("\n모델 로딩 중...")
        # wellbeing_analyzer.py와 동일한 방식으로 로드
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        model_class_names = model.names
        
        print(f"\n✅ 모델이 지원하는 클래스 수: {len(model_class_names)}")
        
        if len(model_class_names) == len(coco_classes):
            print("✅ 모델이 모든 COCO 80가지 클래스를 지원합니다!\n")
        else:
            print(f"⚠️  클래스 수가 다릅니다. (모델: {len(model_class_names)}, COCO: {len(coco_classes)})\n")
        
        # 모델 클래스와 COCO 클래스 비교
        print("[모델 클래스 ID와 이름]\n")
        for cls_id in sorted(model_class_names.keys()):
            cls_name = model_class_names[cls_id]
            coco_idx = coco_classes.index(cls_name) + 1 if cls_name in coco_classes else None
            marker = "✓" if cls_name in coco_classes else "✗"
            print(f"  ID {cls_id:2d}: {cls_name:20s} {marker} (COCO #{coco_idx if coco_idx else 'N/A'})")
        
        print("\n" + "="*70)
        print("\n검출 테스트를 실행하려면:")
        print("  python test_coco_detection.py")
        print("\n웹캠으로 실시간 검출 테스트:")
        print("  python test_coco_detection.py")
        print("\n이미지 파일로 검출 테스트:")
        print("  python test_coco_detection.py <이미지경로>")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Model load error: {e}")
        print("Please check your internet connection or ensure torch is installed.\n")

if __name__ == "__main__":
    main()

