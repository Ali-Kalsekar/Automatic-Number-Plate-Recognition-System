from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from ultralytics import YOLO

from utils.models import VehicleDetection


class VehicleDetector:
    """Detect vehicles using YOLOv8."""

    vehicle_classes = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        use_gpu_if_available: bool = True,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = 0 if use_gpu_if_available and torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)

    def detect(self, frame, confidence_threshold: float | None = None) -> list[VehicleDetection]:
        """Detect vehicles and return normalized detection objects."""

        if frame is None or frame.size == 0:
            return []

        threshold = self.confidence_threshold if confidence_threshold is None else confidence_threshold
        results = self.model.predict(
            source=frame,
            conf=threshold,
            classes=list(self.vehicle_classes.keys()),
            verbose=False,
            device=self.device,
        )

        detections: list[VehicleDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls.item()) if box.cls is not None else -1
                if class_id not in self.vehicle_classes:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf.item()) if box.conf is not None else 0.0
                detections.append(
                    VehicleDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        class_id=class_id,
                        label=self.vehicle_classes[class_id],
                    )
                )

        return detections
