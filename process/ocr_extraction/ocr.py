# OCR/ocr.py
from typing import List, Tuple
import numpy as np

try:
    import easyocr
except ImportError as e:
    raise ImportError(
        "Falta easyocr. Instala con: pip install easyocr"
    ) from e


BBox = List[Tuple[float, float]]  # 4 puntos (x,y)
Detection = Tuple[BBox, str, float]


class OcrProcess:
    def __init__(self, languages=None, gpu: bool = True):
        if languages is None:
            # placas usualmente: letras + números (inglés funciona bien)
            languages = ["en"]
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def text_detection(self, image: np.ndarray) -> List[Detection]:
        """
        image: np.ndarray (GRAY o BGR). Idealmente ya preprocesada.
        return: [(bbox4pts, text, conf), ...]
        """
        # easyocr: detail=1 -> devuelve bbox, text, conf
        results = self.reader.readtext(image, detail=1)

        out: List[Detection] = []
        for bbox, text, conf in results:
            # bbox viene como list of 4 points
            bbox4 = [(float(p[0]), float(p[1])) for p in bbox]
            out.append((bbox4, str(text), float(conf)))
        return out

    # Si quieres mantener el nombre que tú usabas:
    def extractor_text_line(self, det: Detection):
        """
        Para compatibilidad con tu estilo anterior:
        input: (bbox,text,conf) -> devuelve igual
        """
        return det
