from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import pytesseract
from pytesseract import Output, TesseractNotFoundError

from utils.image_processing import is_valid_plate_text, normalize_plate_text, preprocess_plate_variants
from utils.models import OCRResult


class PlateReader:
    """Read and validate license plate text using Tesseract OCR."""

    def __init__(self, language: str = "eng", psm: int = 7, oem: int = 3) -> None:
        self.language = language
        self.psm = psm
        self.oem = oem
        self.whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def read_plate(self, plate_image) -> OCRResult:
        """Read text from a plate image and return the best candidate."""

        if plate_image is None or plate_image.size == 0:
            return OCRResult(text="", confidence=0.0)

        best_text = ""
        best_confidence = 0.0

        for variant in preprocess_plate_variants(plate_image):
            try:
                candidate_text, confidence = self._ocr_variant(variant)
            except TesseractNotFoundError as exc:
                raise RuntimeError(
                    "Tesseract OCR is not installed or not available on PATH."
                ) from exc

            normalized = normalize_plate_text(candidate_text)
            if not normalized:
                continue

            if not is_valid_plate_text(normalized) and len(normalized) < 6:
                continue

            if confidence >= best_confidence:
                best_text = normalized
                best_confidence = confidence

        return OCRResult(text=best_text, confidence=best_confidence)

    def _ocr_variant(self, image) -> tuple[str, float]:
        """Run OCR on a single preprocessing variant."""

        config = f"--oem {self.oem} --psm {self.psm} -c tessedit_char_whitelist={self.whitelist}"
        data = pytesseract.image_to_data(image, lang=self.language, config=config, output_type=Output.DICT)

        tokens: list[str] = []
        confidences: list[float] = []
        for text, confidence_text in zip(data.get("text", []), data.get("conf", []), strict=False):
            cleaned = text.strip()
            if cleaned:
                tokens.append(cleaned)
            try:
                confidence = float(confidence_text)
            except (TypeError, ValueError):
                confidence = -1.0
            if confidence >= 0:
                confidences.append(confidence)

        if not tokens:
            raw_text = pytesseract.image_to_string(image, lang=self.language, config=config)
            tokens = [raw_text]

        candidate_text = "".join(tokens)
        candidate_text = re.sub(r"\s+", "", candidate_text)

        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        if confidence > 1.0:
            confidence = confidence / 100.0

        return candidate_text, confidence
