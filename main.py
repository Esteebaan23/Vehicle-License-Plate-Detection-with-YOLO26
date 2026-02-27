import argparse
import os
import time
import cv2
import numpy as np
import fiftyone as fo
import fiftyone.core.labels as fol

from process.utils.geometry import (
    clamp_box, scale_box, normalize_bbox_xyxy, parse_roi, center_in_roi, Box
)
from process.computer_vision_models.vehicle_detection import VehicleDetection
from ultralytics import YOLO

# OCR
from process.ocr_extraction.text_extraction import TextExtraction

MODELS_DIR = "models"


def get_model_name(path_or_id: str) -> str:
    return os.path.basename(path_or_id)


def resolve_model_path(p: str) -> str:
    if os.path.exists(p):
        return p
    candidate = os.path.join(MODELS_DIR, p)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Model not found: {p}")


def yolo_result_to_boxes(res) -> list[Box]:
    if res.boxes is None or len(res.boxes) == 0:
        return []
    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    out = []
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        out.append(Box(int(x1), int(y1), int(x2), int(y2), float(c), int(k)))
    return out


def safe_first_line(text: str, max_len: int = 16) -> str:
    if not text:
        return ""
    line = text.splitlines()[0].strip()
    return line[:max_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--car-seg-model", default="yolo26n-seg.pt")
    ap.add_argument("--plate-det-model", required=True)
    ap.add_argument("--out", default="out.mp4")
    ap.add_argument("--conf-car", type=float, default=0.15)
    ap.add_argument("--conf-plate", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--plate-imgsz", type=int, default=416)
    ap.add_argument("--plates-in-cars-only", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--infer-scale", type=float, default=0.5)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-masks", action="store_true")

    # ROI (general)
    ap.add_argument("--use-roi", action="store_true")
    ap.add_argument("--roi", default="0.45,0.35,0.75,0.6")
    ap.add_argument("--roi-mode", default="plates", choices=["cars", "plates", "both"])

    # OCR
    ap.add_argument("--ocr", action="store_true", help="Apply OCR on detected plates")
    ap.add_argument("--ocr-min-conf", type=float, default=0.75,
                    help="Run OCR only if plate confidence >= this value")
    ap.add_argument("--ocr-max-len", type=int, default=16,
                    help="Max chars drawn on video for OCR text")

    # FiftyOne
    ap.add_argument("--fo", action="store_true")
    ap.add_argument("--fo-dataset", default="cars_plates_experiments")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--frames-dir", default="frames_runs")
    ap.add_argument("--fo-flush", type=int, default=50)
    ap.add_argument("--launch-fo", action="store_true")

    args = ap.parse_args()

    args.car_seg_model = resolve_model_path(args.car_seg_model)
    args.plate_det_model = resolve_model_path(args.plate_det_model)

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    if not args.run_id:
        args.run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_scale{args.infer_scale}_skip{args.frame_skip}"

    # Models
    car_det = VehicleDetection(args.car_seg_model, device=args.device)
    plate_model = YOLO(args.plate_det_model)

    # OCR engine (once)
    ocr_engine = TextExtraction() if args.ocr else None

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi = (0, 0, w, h)
    if args.use_roi:
        roi = parse_roi(args.roi, w, h)

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))

    # FiftyOne
    dataset = None
    buffer = []
    if args.fo:
        os.makedirs(args.frames_dir, exist_ok=True)
        try:
            dataset = fo.load_dataset(args.fo_dataset)
        except Exception:
            dataset = fo.Dataset(args.fo_dataset)

        dataset.info.setdefault("runs", {})
        dataset.info["runs"][args.run_id] = {
            "video": os.path.abspath(args.video),
            "car_model": get_model_name(args.car_seg_model),
            "plate_model": get_model_name(args.plate_det_model),
            "conf_car": float(args.conf_car),
            "conf_plate": float(args.conf_plate),
            "infer_scale": float(args.infer_scale),
            "frame_skip": int(args.frame_skip),
            "plates_in_cars_only": bool(args.plates_in_cars_only),
            "roi": args.roi if args.use_roi else "FULL",
            "roi_mode": args.roi_mode if args.use_roi else "none",
            "ocr": bool(args.ocr),
            "ocr_min_conf": float(args.ocr_min_conf),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        dataset.save()

    last_cars: list[Box] = []
    last_plates: list[dict] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        raw = frame.copy()

        do_infer = (frame_idx % args.frame_skip == 0)

        if do_infer:
            # inference image
            if args.infer_scale != 1.0:
                iw = max(16, int(w * args.infer_scale))
                ih = max(16, int(h * args.infer_scale))
                inf = cv2.resize(raw, (iw, ih), interpolation=cv2.INTER_LINEAR)
            else:
                iw, ih = w, h
                inf = raw

            sx, sy = w / iw, h / ih

            # --- Cars ---
            cars_inf, _ = car_det.predict_cars(inf, conf=args.conf_car, imgsz=args.imgsz)
            cars = [scale_box(b, sx, sy) for b in cars_inf]

            if args.use_roi and args.roi_mode in ("cars", "both"):
                cars = [b for b in cars if center_in_roi(b, roi)]

            last_cars = cars

            # --- Plates ---
            recs = []

            if args.plates_in_cars_only and last_cars:
                crops, parents = [], []
                for cb in last_cars:
                    cx1, cy1, cx2, cy2 = clamp_box(cb.x1, cb.y1, cb.x2, cb.y2, w, h)
                    crop = raw[cy1:cy2, cx1:cx2]
                    if crop.size == 0:
                        continue
                    crops.append(crop)
                    parents.append((cx1, cy1, cx2, cy2))

                if crops:
                    dets = plate_model.predict(
                        crops,
                        conf=args.conf_plate,
                        imgsz=args.plate_imgsz,
                        device=args.device,
                        verbose=False,
                    )

                    for det_res, (cx1, cy1, cx2, cy2) in zip(dets, parents):
                        for pb in yolo_result_to_boxes(det_res):
                            px1, py1 = pb.x1 + cx1, pb.y1 + cy1
                            px2, py2 = pb.x2 + cx1, pb.y2 + cy1
                            recs.append({"box": Box(px1, py1, px2, py2, pb.conf, pb.cls)})

            else:
                det_res = plate_model.predict(
                    inf,
                    conf=args.conf_plate,
                    imgsz=args.plate_imgsz,
                    device=args.device,
                    verbose=False,
                )[0]

                for pb in yolo_result_to_boxes(det_res):
                    pb_full = scale_box(pb, sx, sy)
                    recs.append({"box": pb_full})

            if args.use_roi and args.roi_mode in ("plates", "both"):
                recs = [r for r in recs if center_in_roi(r["box"], roi)]

            last_plates = recs

            # --- OCR (only for plates with conf >= ocr_min_conf) ---
            if args.ocr and ocr_engine is not None:
                for i, r in enumerate(last_plates):
                    b = r["box"]

                    #print(f"[OCR] plate#{i} conf={b.conf:.3f} thr={args.ocr_min_conf:.3f}")

                    if float(b.conf) < float(args.ocr_min_conf):
                        r["text"] = ""
                        #print("  -> skip (low conf)")
                        continue

                    x1, y1, x2, y2 = clamp_box(b.x1, b.y1, b.x2, b.y2, w, h)
                    crop = raw[y1:y2, x1:x2]
                    #print(f"  crop shape={crop.shape if crop.size else None}")

                    if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 30:
                        r["text"] = ""
                        #print("  -> skip (empty/too small crop)")
                        continue

                    # guarda el crop para inspeccionar
                    os.makedirs("debug_ocr", exist_ok=True)
                    cv2.imwrite(f"debug_ocr/frame{frame_idx:06d}_p{i}_conf{b.conf:.2f}.jpg", crop)

                    try:
                        txt = ocr_engine.text_extraction(crop)
                        r["text"] = txt
                        print("  OCR raw:", repr(txt))
                    except Exception as e:
                        r["text"] = ""
                        #print("  [OCR ERROR]", repr(e))

            # --- FiftyOne logging on inferred frames ---
            if args.fo and dataset is not None:
                frame_path = os.path.join(args.frames_dir, args.run_id, f"frame_{frame_idx:06d}.jpg")
                os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                cv2.imwrite(frame_path, raw)

                car_dets = []
                for b in last_cars:
                    x1, y1, x2, y2 = clamp_box(b.x1, b.y1, b.x2, b.y2, w, h)
                    car_dets.append(
                        fol.Detection(
                            label="car",
                            bounding_box=normalize_bbox_xyxy(x1, y1, x2, y2, w, h),
                            confidence=float(b.conf),
                        )
                    )

                plate_dets = []
                for r in last_plates:
                    b = r["box"]
                    x1, y1, x2, y2 = clamp_box(b.x1, b.y1, b.x2, b.y2, w, h)

                    det = fol.Detection(
                        label="plate",
                        bounding_box=normalize_bbox_xyxy(x1, y1, x2, y2, w, h),
                        confidence=float(b.conf),
                    )

                    # attach OCR text (if any)
                    txt = r.get("text", "")
                    if txt:
                        det["ocr_text"] = safe_first_line(txt, 64)

                    plate_dets.append(det)

                s = fo.Sample(filepath=frame_path)
                s["run_id"] = args.run_id
                s["frame_number"] = int(frame_idx)
                s["car_dets"] = fol.Detections(detections=car_dets)
                s["plate_dets"] = fol.Detections(detections=plate_dets)
                buffer.append(s)

                if len(buffer) >= args.fo_flush:
                    dataset.add_samples(buffer)
                    buffer.clear()
                    dataset.save()

        # --- Draw ROI ---
        if args.use_roi:
            rx1, ry1, rx2, ry2 = roi
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
            cv2.putText(frame, "ROI", (rx1, max(0, ry1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # --- Draw cars ---
        for b in last_cars:
            x1, y1, x2, y2 = clamp_box(b.x1, b.y1, b.x2, b.y2, w, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, f"car {b.conf:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        # --- Draw plates + OCR text ---
        for r in last_plates:
            b = r["box"]
            x1, y1, x2, y2 = clamp_box(b.x1, b.y1, b.x2, b.y2, w, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"plate {b.conf:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            txt = safe_first_line(r.get("text", ""), args.ocr_max_len)
            if txt:
                cv2.putText(frame, txt, (x1, min(h - 5, y2 + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        writer.write(frame)

        if args.show:
            cv2.imshow("ALPR (with OCR)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Flush remaining FO samples
    if args.fo and dataset is not None and buffer:
        dataset.add_samples(buffer)
        buffer.clear()
        dataset.save()

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print("[DONE] Saved:", args.out)
    if args.fo and dataset is not None and args.launch_fo:
        session = fo.launch_app(dataset)
        session.wait()


if __name__ == "__main__":
    main()
