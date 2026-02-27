from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls: int

def clamp_box(x1, y1, x2, y2, w, h) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2

def scale_box(b: Box, sx: float, sy: float) -> Box:
    return Box(
        x1=int(b.x1 * sx),
        y1=int(b.y1 * sy),
        x2=int(b.x2 * sx),
        y2=int(b.y2 * sy),
        conf=b.conf,
        cls=b.cls,
    )

def normalize_bbox_xyxy(x1, y1, x2, y2, w, h) -> List[float]:
    x = x1 / w
    y = y1 / h
    bw = max(0.0, (x2 - x1) / w)
    bh = max(0.0, (y2 - y1) / h)
    return [x, y, bw, bh]

def parse_roi(roi_str: str, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    parts = [p.strip() for p in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError("roi debe ser x1,y1,x2,y2 (pixeles o 0-1)")
    vals = [float(x) for x in parts]
    if all(v <= 1.0 for v in vals):
        x1 = int(vals[0] * frame_w); y1 = int(vals[1] * frame_h)
        x2 = int(vals[2] * frame_w); y2 = int(vals[3] * frame_h)
    else:
        x1, y1, x2, y2 = map(int, vals)
    return clamp_box(x1, y1, x2, y2, frame_w, frame_h)

def center_in_roi(b: Box, roi: Tuple[int,int,int,int]) -> bool:
    rx1, ry1, rx2, ry2 = roi
    cx = (b.x1 + b.x2) // 2
    cy = (b.y1 + b.y2) // 2
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)
