from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from utils.image_processing import crop_with_bbox
from utils.models import PlateDetection


class PlateDetector:
    """Detect license plates using either a YOLO model or contour heuristics."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float = 0.4,
        min_plate_area: int = 1200,
        use_gpu_if_available: bool = True,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.min_plate_area = min_plate_area
        self.device = 0 if use_gpu_if_available and torch.cuda.is_available() else "cpu"
        self.model: YOLO | None = None
        self.use_model = False

        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            self.use_model = True

    def detect(self, vehicle_image: np.ndarray) -> PlateDetection | None:
        """Detect a license plate inside a vehicle crop."""

        if vehicle_image is None or vehicle_image.size == 0:
            return None

        if self.use_model and self.model is not None:
            result = self._detect_with_model(vehicle_image)
            if result is not None:
                return result

        return self._detect_with_contours(vehicle_image)

    def _detect_with_model(self, vehicle_image: np.ndarray) -> PlateDetection | None:
        """Detect plates using a YOLO plate detector if available."""

        results = self.model.predict(  # type: ignore[union-attr]
            source=vehicle_image,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
        )

        best_detection: PlateDetection | None = None
        best_confidence = 0.0
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                confidence = float(box.conf.item()) if box.conf is not None else 0.0
                if confidence < best_confidence:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = crop_with_bbox(vehicle_image, (x1, y1, x2, y2), padding=2)
                if crop.size == 0:
                    continue
                best_confidence = confidence
                best_detection = PlateDetection(bbox=(x1, y1, x2, y2), confidence=confidence, image=crop, source="yolo")

        return best_detection

    def _detect_with_contours(self, vehicle_image: np.ndarray) -> PlateDetection | None:
        """Detect plates with OpenCV contour heuristics."""

        candidates = self._find_candidates(vehicle_image, lower_half_only=True)
        if not candidates:
            candidates = self._find_candidates(vehicle_image, lower_half_only=False)

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        _, bbox = candidates[0]
        x1, y1, x2, y2 = bbox
        crop = crop_with_bbox(vehicle_image, bbox, padding=3)
        if crop.size == 0:
            return None

        confidence = self._estimate_confidence(vehicle_image, bbox)
        return PlateDetection(bbox=(x1, y1, x2, y2), confidence=confidence, image=crop, source="contour")

    def _find_candidates(self, vehicle_image: np.ndarray, lower_half_only: bool) -> list[tuple[float, tuple[int, int, int, int]]]:
        """Search for rectangular contour candidates in a crop."""

        height, width = vehicle_image.shape[:2]
        offset_y = height // 3 if lower_half_only else 0
        search_region = vehicle_image[offset_y:, :] if lower_half_only else vehicle_image

        if search_region.size == 0:
            return []

        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY) if len(search_region.shape) == 3 else search_region.copy()
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_plate_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            aspect_ratio = w / float(h)
            if aspect_ratio < 2.0 or aspect_ratio > 7.5:
                continue

            if w < width * 0.08 or h < height * 0.03:
                continue

            score = area * min(aspect_ratio, 7.5)
            bbox = (x, y + offset_y, x + w, y + h + offset_y)
            candidates.append((score, bbox))

        return candidates

    def _estimate_confidence(self, vehicle_image: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        """Estimate a heuristic confidence score for a contour candidate."""

        x1, y1, x2, y2 = bbox
        plate_width = max(1, x2 - x1)
        plate_height = max(1, y2 - y1)
        aspect_ratio = plate_width / float(plate_height)
        target_ratio = 4.5
        ratio_score = max(0.0, 1.0 - abs(aspect_ratio - target_ratio) / target_ratio)
        area_ratio = (plate_width * plate_height) / float(vehicle_image.shape[0] * vehicle_image.shape[1])
        area_score = min(1.0, area_ratio * 10.0)
        confidence = 0.45 + 0.35 * ratio_score + 0.2 * area_score
        return round(float(min(confidence, 0.95)), 3)
