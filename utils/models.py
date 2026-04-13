from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

BBox = Tuple[int, int, int, int]


@dataclass(slots=True)
class VehicleDetection:
    """Vehicle detection result."""

    bbox: BBox
    confidence: float
    class_id: int
    label: str


@dataclass(slots=True)
class PlateDetection:
    """License plate detection result."""

    bbox: BBox
    confidence: float
    image: np.ndarray
    source: str = "contour"


@dataclass(slots=True)
class OCRResult:
    """OCR output result."""

    text: str
    confidence: float
