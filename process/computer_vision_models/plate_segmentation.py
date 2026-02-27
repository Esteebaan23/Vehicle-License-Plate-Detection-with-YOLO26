import numpy as np
import cv2
from typing import Tuple, Optional

class PlateSegmentation:
    """
    Pseudo-segmentation para placas AMARILLAS:
    - mascara por HSV (amarillo)
    - morph cleanup
    - devuelve mask + bbox refinado (opcional)
    """
    def __init__(self):
        pass

    def yellow_mask(self, plate_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([15, 60, 60], dtype=np.uint8)
        upper = np.array([40, 255, 255], dtype=np.uint8)
        m = cv2.inRange(hsv, lower, upper)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
        return m

    def mask_processing(self, plate_bgr: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        if plate_bgr is None or plate_bgr.size == 0:
            return None
        if mask is None or mask.size == 0:
            return None
        # background to white
        out = plate_bgr.copy()
        out[mask == 0] = (255,255,255)
        return out

    def refine_bbox_from_mask(self, mask: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        if (x2-x1) < 30 or (y2-y1) < 10:
            return None
        return x1, y1, x2, y2
