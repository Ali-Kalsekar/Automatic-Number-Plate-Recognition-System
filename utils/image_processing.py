from __future__ import annotations

from pathlib import Path
import re
from typing import List, Sequence

import cv2
import numpy as np

PLATE_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$"),
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$"),
    re.compile(r"^[A-Z0-9]{6,12}$"),
)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    """Resize a frame while keeping aspect ratio."""

    if target_width <= 0:
        return frame

    height, width = frame.shape[:2]
    if width <= target_width:
        return frame

    scale = target_width / float(width)
    target_height = int(height * scale)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def clip_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """Clip a bounding box to image dimensions."""

    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    return x1, y1, x2, y2


def crop_with_bbox(image: np.ndarray, bbox: tuple[int, int, int, int], padding: int = 0) -> np.ndarray:
    """Crop an image using a bounding box and optional padding."""

    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding
    x1, y1, x2, y2 = clip_bbox((x1, y1, x2, y2), width, height)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return image[y1:y2, x1:x2].copy()


def normalize_plate_text(text: str) -> str:
    """Normalize OCR output by keeping only alphanumeric uppercase characters."""

    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid_plate_text(text: str) -> bool:
    """Validate a candidate license plate string."""

    if not text:
        return False
    return any(pattern.fullmatch(text) for pattern in PLATE_PATTERNS)


def preprocess_plate_variants(plate_image: np.ndarray) -> list[np.ndarray]:
    """Generate preprocessing variants to improve OCR robustness."""

    if plate_image.size == 0:
        return []

    if len(plate_image.shape) == 3:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_image.copy()

    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    denoised = cv2.bilateralFilter(blur, 9, 75, 75)

    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_inv = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    adaptive_inv = cv2.bitwise_not(adaptive)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)

    return [gray, denoised, cleaned, otsu_inv, adaptive, adaptive_inv]
