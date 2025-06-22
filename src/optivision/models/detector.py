from typing import List
from pydantic import BaseModel
from ultralytics import YOLO
import numpy as np
from PIL import Image

from ..config import get_settings

settings = get_settings()
_yolo = YOLO(settings.yolo_weights.as_posix())


class BBox(BaseModel):
    class_name: str
    confidence: float
    xyxy: List[float]  # [x1,y1,x2,y2]


def detect_objects(image: Image.Image) -> List[BBox]:
    """Run YOLOv8 on a PIL image and return structured results."""
    results = _yolo(image)
    boxes = results[0].boxes

    detections: List[BBox] = []
    for box in boxes:
        detections.append(
            BBox(
                class_name=results[0].names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                xyxy=[float(x) for x in box.xyxy[0].tolist()],
            )
        )
    return detections
