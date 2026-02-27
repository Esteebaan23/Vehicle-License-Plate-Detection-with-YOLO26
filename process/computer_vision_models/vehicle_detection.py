from ultralytics import YOLO
from typing import List, Tuple
import numpy as np

from process.utils.geometry import Box

COCO_CAR_ID = 2

class VehicleDetection:
    def __init__(self, weights: str, device=None):
        self.model = YOLO(weights)
        self.device = device

    def predict_cars(self, frame: np.ndarray, conf: float, imgsz: int) -> Tuple[List[Box], object]:
        res = self.model.predict(
            frame, conf=conf, imgsz=imgsz, classes=[COCO_CAR_ID],
            device=self.device, verbose=False
        )[0]
        boxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for (x1,y1,x2,y2), c, k in zip(xyxy, confs, clss):
                boxes.append(Box(int(x1), int(y1), int(x2), int(y2), float(c), int(k)))
        return boxes, res
