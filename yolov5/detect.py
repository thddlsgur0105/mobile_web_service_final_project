# """
# Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.
# """

# import argparse
# import csv
# import os
# import platform
# import sys
# from pathlib import Path

# import torch

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# from ultralytics.utils.plotting import Annotator, colors, save_one_box

# from models.common import DetectMultiBackend
# from changedetection import ChangeDetection
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# from utils.general import (
#     LOGGER,
#     Profile,
#     check_file,
#     check_img_size,
#     check_imshow,
#     check_requirements,
#     colorstr,
#     cv2,
#     increment_path,
#     non_max_suppression,
#     print_args,
#     scale_boxes,
#     strip_optimizer,
#     xyxy2xywh,
# )
# from utils.torch_utils import select_device, smart_inference_mode


# @smart_inference_mode()
# def run(
#     weights=ROOT / "yolov5s.pt",
#     source=ROOT / "data/images",
#     data=ROOT / "data/coco128.yaml",
#     imgsz=(640, 640),
#     conf_thres=0.25,
#     iou_thres=0.45,
#     max_det=1000,
#     device="",
#     view_img=False,
#     save_txt=False,
#     save_format=0,
#     save_csv=False,
#     save_conf=False,
#     save_crop=False,
#     nosave=False,
#     classes=None,
#     agnostic_nms=False,
#     augment=False,
#     visualize=False,
#     update=False,
#     project=ROOT / "runs/detect",
#     name="exp",
#     exist_ok=False,
#     line_thickness=3,
#     hide_labels=False,
#     hide_conf=False,
#     half=False,
#     dnn=False,
#     vid_stride=1,
# ):

#     source = str(source)
#     save_img = not nosave and not source.endswith(".txt")
#     is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
#     is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
#     webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
#     screenshot = source.lower().startswith("screen")
#     if is_url and is_file:
#         source = check_file(source)

#     save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
#     (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
#     stride, names, pt = model.stride, model.names, model.pt
#     imgsz = check_img_size(imgsz, s=stride)

#     bs = 1
#     if webcam:
#         view_img = check_imshow(warn=True)
#         dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
#         bs = len(dataset)
#     elif screenshot:
#         dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
#     else:
#         dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
#     vid_path, vid_writer = [None] * bs, [None] * bs

#     cd = ChangeDetection(names)

#     model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
#     seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

#     for path, im, im0s, vid_cap, s in dataset:
#         with dt[0]:
#             im = torch.from_numpy(im).to(model.device)
#             im = im.half() if model.fp16 else im.float()
#             im /= 255
#             if len(im.shape) == 3:
#                 im = im[None]
#             if model.xml and im.shape[0] > 1:
#                 ims = torch.chunk(im, im.shape[0], 0)

#         with dt[1]:
#             visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
#             if model.xml and im.shape[0] > 1:
#                 pred = None
#                 for image in ims:
#                     if pred is None:
#                         pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
#                     else:
#                         pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), 0)
#                 pred = [pred, None]
#             else:
#                 pred = model(im, augment=augment, visualize=visualize)

#         with dt[2]:
#             pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

#         csv_path = save_dir / "predictions.csv"

#         def write_to_csv(image_name, prediction, confidence):
#             data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
#             file_exists = os.path.isfile(csv_path)
#             with open(csv_path, mode="a", newline="") as f:
#                 writer = csv.DictWriter(f, fieldnames=data.keys())
#                 if not file_exists:
#                     writer.writeheader()
#                 writer.writerow(data)

#         for i, det in enumerate(pred):
#             seen += 1
#             if webcam:
#                 p, im0, frame = path[i], im0s[i].copy(), dataset.count
#                 s += f"{i}: "
#             else:
#                 p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

#             p = Path(p)
#             save_path = str(save_dir / p.name)
#             txt_path = str(save_dir / "labels" / p.stem) + (
#                 "" if dataset.mode == "image" else f"_{frame}"
#             )
#             s += "{:g}x{:g} ".format(*im.shape[2:])
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
#             imc = im0.copy() if save_crop else im0

#             annotator = Annotator(im0, line_width=line_thickness, example=str(names))

#             detected = [0 for _ in range(len(names))]

#             # =====================================================
#             # === ADDED FOR COUNTING: 객체 카운트 딕셔너리 생성 ===
#             # =====================================================
#             counts = {}
#             # =====================================================

#             if len(det):
#                 det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

#                 for c in det[:, 5].unique():
#                     n = (det[:, 5] == c).sum()
#                     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

#                 for *xyxy, conf, cls in reversed(det):
#                     c = int(cls)
#                     detected[int(cls)] = 1
#                     label = names[c] if hide_conf else f"{names[c]}"
#                     confidence = float(conf)
#                     confidence_str = f"{confidence:.2f}"

#                     if save_csv:
#                         write_to_csv(p.name, label, confidence_str)

#                     if save_txt:
#                         if save_format == 0:
#                             coords = (
#                                 (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
#                             )
#                         else:
#                             coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()

#                         line = (cls, *coords, conf) if save_conf else (cls, *coords)
#                         with open(f"{txt_path}.txt", "a") as f:
#                             f.write(("%g " * len(line)).rstrip() % line + "\n")

#                     # ============================================================
#                     # === ADDED FOR COUNTING: 클래스별 카운트 증가 로직 삽입 ===
#                     # ============================================================
#                     class_name = names[int(cls)]
#                     counts[class_name] = counts.get(class_name, 0) + 1
#                     # ============================================================

#                     if save_img or save_crop or view_img:
#                         c = int(cls)
#                         label = (
#                             None
#                             if hide_labels
#                             else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
#                         )
#                         annotator.box_label(xyxy, label, color=colors(c, True))

#                     if save_crop:
#                         save_one_box(
#                             xyxy,
#                             imc,
#                             file=save_dir / "crops" / names[c] / f"{p.stem}.jpg",
#                             BGR=True,
#                         )

#             cd.add(names, detected, save_dir, im0)

#             im0 = annotator.result()

#             # ============================================================
#             # === ADDED FOR COUNTING: 화면 좌측 상단 카운트 표시 영역 ===
#             # ============================================================
#             y_offset = 30
#             for cls_name, cnt in counts.items():
#                 cv2.putText(
#                     im0,
#                     f"{cls_name}: {cnt}",
#                     (10, y_offset),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     (0, 255, 255),
#                     2,
#                 )
#                 y_offset += 30
#             # ============================================================

#             if view_img:
#                 if platform.system() == "Linux" and p not in windows:
#                     windows.append(p)
#                     cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
#                     cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
#                 cv2.imshow(str(p), im0)
#                 cv2.waitKey(1)

#             if save_img:
#                 if dataset.mode == "image":
#                     cv2.imwrite(save_path, im0)
#                 else:
#                     if vid_path[i] != save_path:
#                         vid_path[i] = save_path
#                         if isinstance(vid_writer[i], cv2.VideoWriter):
#                             vid_writer[i].release()
#                         if vid_cap:
#                             fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                         else:
#                             fps, w, h = 30, im0.shape[1], im0.shape[0]
#                         save_path = str(Path(save_path).with_suffix(".mp4"))
#                         vid_writer[i] = cv2.VideoWriter(
#                             save_path,
#                             cv2.VideoWriter_fourcc(*"mp4v"),
#                             fps,
#                             (w, h),
#                         )
#                     vid_writer[i].write(im0)

#         LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

#     t = tuple(x.t / seen * 1e3 for x in dt)
#     LOGGER.info(
#         f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t
#     )
#     if save_txt or save_img:
#         s = (
#             f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
#             if save_txt
#             else ""
#         )
#         LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
#     if update:
#         strip_optimizer(weights[0])


# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt")
#     parser.add_argument("--source", type=str, default=ROOT / "data/images")
#     parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml")
#     parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
#     parser.add_argument("--conf-thres", type=float, default=0.25)
#     parser.add_argument("--iou-thres", type=float, default=0.45)
#     parser.add_argument("--max-det", type=int, default=1000)
#     parser.add_argument("--device", default="")
#     parser.add_argument("--view-img", action="store_true")
#     parser.add_argument("--save-txt", action="store_true")
#     parser.add_argument("--save-format", type=int, default=0)
#     parser.add_argument("--save-csv", action="store_true")
#     parser.add_argument("--save-conf", action="store_true")
#     parser.add_argument("--save-crop", action="store_true")
#     parser.add_argument("--nosave", action="store_true")
#     parser.add_argument("--classes", nargs="+", type=int)
#     parser.add_argument("--agnostic-nms", action="store_true")
#     parser.add_argument("--augment", action="store_true")
#     parser.add_argument("--visualize", action="store_true")
#     parser.add_argument("--update", action="store_true")
#     parser.add_argument("--project", default=ROOT / "runs/detect")
#     parser.add_argument("--name", default="exp")
#     parser.add_argument("--exist-ok", action="store_true")
#     parser.add_argument("--line-thickness", default=3, type=int)
#     parser.add_argument("--hide-labels", action="store_true")
#     parser.add_argument("--hide-conf", action="store_true")
#     parser.add_argument("--half", action="store_true")
#     parser.add_argument("--dnn", action="store_true")
#     parser.add_argument("--vid-stride", type=int, default=1)
#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
#     print_args(vars(opt))
#     return opt


# def main(opt):
#     check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
#     run(**vars(opt))


# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)

"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch
from deepface import DeepFace  # ✅ 감정 분석용 DeepFace 추가

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from utils.plots import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from changedetection import ChangeDetection
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


# ============================
# ✅ DeepFace 감정 분석 함수 추가
# ============================
def analyze_emotion(face_image):
    """
    얼굴 이미지(Numpy 배열)를 입력받아 DeepFace로 감정 분석.
    분석 실패 시 'unknown' 반환.
    """
    try:
        # DeepFace는 BGR/RGB 모두 어느 정도 처리하지만, 안전하게 쓰고 싶으면 cvtColor로 RGB 변환 가능
        # face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        result = DeepFace.analyze(face_image, actions=["emotion"], enforce_detection=False)

        # DeepFace 버전에 따라 list로 나올 수도 있고 dict로 나올 수도 있음
        if isinstance(result, list):
            result = result[0]

        dominant = result.get("dominant_emotion", "unknown")
        return dominant
    except Exception:
        return "unknown"


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source=ROOT / "data/images",
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    save_txt=False,
    save_format=0,
    save_csv=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
):

    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    vid_path, vid_writer = [None] * bs, [None] * bs

    cd = ChangeDetection(names)

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), 0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        csv_path = save_dir / "predictions.csv"

        def write_to_csv(image_name, prediction, confidence):
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)

        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / "labels" / p.stem) + (
                "" if dataset.mode == "image" else f"_{frame}"
            )
            s += "{:g}x{:g} ".format(*im.shape[2:])
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            detected = [0 for _ in range(len(names))]

            # =====================================================
            # ✅ 객체 카운트 딕셔너리 (프레임별)
            # =====================================================
            counts = {}
            # =====================================================

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    base_name = names[c]  # 기본 클래스 이름 (예: 'person', 'cell phone' 등)
                    detected[c] = 1

                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    # CSV에 저장할 때는 기본 클래스 이름만 사용
                    if save_csv:
                        write_to_csv(p.name, base_name, confidence_str)

                    # txt 저장
                    if save_txt:
                        if save_format == 0:
                            coords = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            )
                        else:
                            coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()

                        line = (cls, *coords, conf) if save_conf else (cls, *coords)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    # ✅ 클래스별 카운트
                    counts[base_name] = counts.get(base_name, 0) + 1

                    # ================================
                    # ✅ 감정 분석 + 라벨 생성 부분
                    # ================================
                    draw_label = None
                    if not hide_labels:
                        label_text = base_name

                        # person인 경우에만 감정 분석
                        if base_name == "person":
                            x1, y1, x2, y2 = map(int, xyxy)
                            # 이미지 범위 안으로 클리핑
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(im0.shape[1] - 1, x2)
                            y2 = min(im0.shape[0] - 1, y2)

                            face_crop = im0[y1:y2, x1:x2]

                            if face_crop is not None and face_crop.size > 0:
                                emotion = analyze_emotion(face_crop)
                                label_text = f"{base_name} ({emotion})"
                            else:
                                label_text = base_name

                        if hide_conf:
                            draw_label = label_text
                        else:
                            draw_label = f"{label_text} {conf:.2f}"

                    if save_img or save_crop or view_img:
                        annotator.box_label(xyxy, draw_label, color=colors(c, True))

                    if save_crop:
                        save_one_box(
                            xyxy,
                            imc,
                            file=save_dir / "crops" / base_name / f"{p.stem}.jpg",
                            BGR=True,
                        )

            cd.add(names, detected, save_dir, im0)

            im0 = annotator.result()

            # ============================================================
            # ✅ 화면 좌측 상단에 클래스별 카운트 표시
            # ============================================================
            y_offset = 30
            for cls_name, cnt in counts.items():
                cv2.putText(
                    im0,
                    f"{cls_name}: {cnt}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                y_offset += 30
            # ============================================================

            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))
                        vid_writer[i] = cv2.VideoWriter(
                            save_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (w, h),
                        )
                    vid_writer[i].write(im0)

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

    t = tuple(x.t / seen * 1e3 for x in dt)
    LOGGER.info(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t
    )
    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
            if save_txt
            else ""
        )
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt")
    parser.add_argument("--source", type=str, default=ROOT / "data/images")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default="")
    parser.add_argument("--view-img", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-format", type=int, default=0)
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    parser.add_argument("--save-crop", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    parser.add_argument("--classes", nargs="+", type=int)
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--project", default=ROOT / "runs/detect")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--line-thickness", default=3, type=int)
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--hide-conf", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dnn", action="store_true")
    parser.add_argument("--vid-stride", type=int, default=1)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
