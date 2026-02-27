import cv2
import numpy as np
from process.ocr_extraction.ocr import OcrProcess


class TextExtraction:
    def __init__(self):
        self.ocr = OcrProcess()
        self.min_vertical_distance = 12

    def clahe(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        updated = cv2.merge((l2, a, b))
        return cv2.cvtColor(updated, cv2.COLOR_LAB2BGR)

    def exposure_level(self, hist: np.ndarray) -> str:
        hist = hist / (np.sum(hist) + 1e-9)
        percent_over = float(np.sum(hist[200:]))
        percent_under = float(np.sum(hist[:50]))
        if percent_over > 0.75:
            return "Overexposed"
        elif percent_under > 0.75:
            return "Underexposed"
        else:
            return "Properly exposed"

    def image_contrast(self, img: np.ndarray) -> np.ndarray:
        # acepta BGR; devuelve imagen lista para OCR (GRAY)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        exp = self.exposure_level(hist)
        if exp in ("Overexposed", "Underexposed"):
            img = self.clahe(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        contrast = cv2.Laplacian(gray, cv2.CV_64F).var()
        if contrast < 100:
            # ojo: equalizeHist espera GRAY y devuelve GRAY
            gray = cv2.equalizeHist(gray)

        return gray

    def same_line(self, yi1, yi2):
        return abs(yi1 - yi2) < self.min_vertical_distance

    def process_text_line(self, text_detected) -> str:
        full_text = ""
        lines_list = []
        for i, text in enumerate(text_detected):
            text_bbox, text_extracted, text_confidence = self.ocr.extractor_text_line(text)
            lines_list.append(text_bbox)
            if i > 0:
                full_text += " " if self.same_line(lines_list[i][1], lines_list[i - 1][1]) else "\n"
            full_text += text_extracted
        return full_text.strip()

    def text_extraction(self, plate_image_crop: np.ndarray) -> str:
        # Prepro para OCR (si tu image_contrast devuelve GRAY o BGR, ambas valen)
        proc = self.image_contrast(plate_image_crop)

        out = self.ocr.text_detection(proc)

        # ✅ Caso A: tu ocr.py devuelve (number_line_text, text_detected)
        if isinstance(out, tuple) and len(out) == 2:
            _, text_detected = out

        # ✅ Caso B: tu ocr.py devuelve solo text_detected
        else:
            text_detected = out

        full_text = self.process_text_line(text_detected)
        return full_text