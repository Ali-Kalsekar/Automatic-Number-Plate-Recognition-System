from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from utils.models import BBox


VEHICLE_COLOR = (0, 180, 0)
PLATE_COLOR = (0, 215, 255)
TEXT_COLOR = (255, 255, 255)
BACKGROUND_COLOR = (0, 0, 0)
FPS_COLOR = (255, 200, 0)


def _draw_filled_label(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    """Draw a high-contrast label background and text."""

    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top_left = (x, max(0, y - text_height - baseline - 8))
    bottom_right = (x + text_width + 10, y)
    cv2.rectangle(frame, top_left, bottom_right, BACKGROUND_COLOR, -1)
    cv2.rectangle(frame, top_left, bottom_right, color, 2)
    cv2.putText(frame, text, (x + 5, y - baseline - 4), font, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def draw_vehicle_box(frame: np.ndarray, bbox: BBox, label: str, confidence: float) -> None:
    """Annotate a detected vehicle."""

    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), VEHICLE_COLOR, 2)
    header = f"{label} {confidence:.2f}"
    _draw_filled_label(frame, header, (x1, y1), VEHICLE_COLOR)


def draw_plate_box(
    frame: np.ndarray,
    bbox: BBox,
    plate_text: str,
    confidence: float,
    show_confidence: bool = True,
) -> None:
    """Annotate a detected license plate and recognized text."""

    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), PLATE_COLOR, 2)
    if plate_text:
        label = f"Plate: {plate_text}"
        if show_confidence:
            label = f"{label} | Conf: {confidence:.2f}"
        _draw_filled_label(frame, label, (x1, y2 + 24 if y2 + 24 < frame.shape[0] else y1 + 24), PLATE_COLOR)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Draw FPS on the frame."""

    text = f"FPS: {fps:.2f}"
    cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, FPS_COLOR, 2, cv2.LINE_AA)


def draw_status(frame: np.ndarray, text: str, origin: tuple[int, int] = (15, 60)) -> None:
    """Draw a simple status string on the frame."""

    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
