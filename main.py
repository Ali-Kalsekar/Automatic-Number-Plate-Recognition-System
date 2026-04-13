from __future__ import annotations

import logging
import shutil
from pathlib import Path
import time
from typing import Any

import cv2
import yaml

from database.db_manager import DatabaseManager
from detector.plate_detector import PlateDetector
from detector.vehicle_detector import VehicleDetector
from ocr.plate_reader import PlateReader
from utils.draw import draw_fps, draw_plate_box, draw_status, draw_vehicle_box
from utils.fps import FPSCounter
from utils.image_processing import clip_bbox, crop_with_bbox, ensure_directory, resize_frame

WINDOW_TITLE = "Automatic Number Plate Recognition System"
DEFAULT_CONFIG_PATH = Path("config/config.yaml")
WINDOWS_TESSERACT_PATHS = (
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration and apply defaults."""

    defaults: dict[str, Any] = {
        "video_source": 0,
        "confidence_threshold": 0.5,
        "vehicle_confidence_threshold": 0.35,
        "frame_resize_width": 1280,
        "save_to_database": True,
        "save_to_csv": True,
        "csv_path": "output/plates_log.csv",
        "sqlite_path": "output/plates.db",
        "save_plate_images": False,
        "plate_images_dir": "output/plates",
        "window_title": WINDOW_TITLE,
        "plate_model_path": "",
        "plate_min_area": 1200,
        "ocr_language": "eng",
        "ocr_psm": 7,
        "ocr_oem": 3,
        "use_gpu_if_available": True,
        "tesseract_cmd": "",
        "display_plate_confidence": True,
    }

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file_handle:
            loaded_config = yaml.safe_load(file_handle) or {}
        defaults.update(loaded_config)

    return defaults


def normalize_video_source(source: Any) -> int | str:
    """Normalize the configured video source for OpenCV."""

    if isinstance(source, int):
        return source

    if isinstance(source, str):
        stripped = source.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped

    return 0


def open_capture(video_source: int | str) -> cv2.VideoCapture:
    """Open a webcam or video file using OpenCV."""

    capture = cv2.VideoCapture(video_source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_source}")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def resolve_tesseract_cmd(configured_path: str) -> str:
    """Resolve the Tesseract executable path from config, PATH, or common Windows install locations."""

    candidate = configured_path.strip()
    if candidate and Path(candidate).exists():
        return candidate

    which_path = shutil.which("tesseract")
    if which_path:
        return which_path

    for path in WINDOWS_TESSERACT_PATHS:
        if path.exists():
            return str(path)

    raise RuntimeError(
        "Tesseract OCR was not found. Install it or set config/config.yaml -> tesseract_cmd to the executable path."
    )


def save_plate_image(plate_image, output_dir: Path, plate_number: str) -> None:
    """Save a cropped plate image for auditing or later review."""

    ensure_directory(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_plate = plate_number if plate_number else "unknown"
    file_path = output_dir / f"{timestamp}_{safe_plate}.jpg"
    cv2.imwrite(str(file_path), plate_image)


def main() -> None:
    """Run the ANPR system."""

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = load_config(DEFAULT_CONFIG_PATH)
    window_title = config.get("window_title", WINDOW_TITLE)

    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = resolve_tesseract_cmd(str(config.get("tesseract_cmd", "")))
    logging.info("Using Tesseract executable: %s", pytesseract.pytesseract.tesseract_cmd)

    video_source = normalize_video_source(config.get("video_source", 0))
    vehicle_threshold = float(config.get("vehicle_confidence_threshold", 0.35))
    plate_threshold = float(config.get("confidence_threshold", 0.5))
    use_gpu = bool(config.get("use_gpu_if_available", True))

    capture = open_capture(video_source)
    fps_counter = FPSCounter()

    plate_model_path = str(config.get("plate_model_path", "")).strip() or None
    plate_detector = PlateDetector(
        model_path=plate_model_path,
        confidence_threshold=plate_threshold,
        min_plate_area=int(config.get("plate_min_area", 1200)),
        use_gpu_if_available=use_gpu,
    )
    vehicle_detector = VehicleDetector(
        model_path="yolov8n.pt",
        confidence_threshold=vehicle_threshold,
        use_gpu_if_available=use_gpu,
    )
    plate_reader = PlateReader(
        language=str(config.get("ocr_language", "eng")),
        psm=int(config.get("ocr_psm", 7)),
        oem=int(config.get("ocr_oem", 3)),
    )

    sqlite_path = Path(config.get("sqlite_path", "output/plates.db"))
    csv_path = Path(config.get("csv_path", "output/plates_log.csv"))
    save_to_database = bool(config.get("save_to_database", True))
    save_to_csv = bool(config.get("save_to_csv", True))
    save_plate_images_flag = bool(config.get("save_plate_images", False))
    plate_images_dir = Path(config.get("plate_images_dir", "output/plates"))
    display_plate_confidence = bool(config.get("display_plate_confidence", True))

    with DatabaseManager(
        sqlite_path=sqlite_path,
        csv_path=csv_path,
        save_to_database=save_to_database,
        save_to_csv=save_to_csv,
    ) as database_manager:
        seen_session: set[str] = set()

        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    logging.info("Video stream ended or frame could not be read.")
                    break

                frame = resize_frame(frame, int(config.get("frame_resize_width", 1280)))
                display_frame = frame.copy()

                frame_start = time.perf_counter()
                vehicle_detections = vehicle_detector.detect(frame, confidence_threshold=vehicle_threshold)

                for vehicle in vehicle_detections:
                    x1, y1, x2, y2 = clip_bbox(vehicle.bbox, frame.shape[1], frame.shape[0])
                    vehicle_crop = crop_with_bbox(frame, (x1, y1, x2, y2))
                    if vehicle_crop.size == 0:
                        draw_vehicle_box(display_frame, (x1, y1, x2, y2), vehicle.label, vehicle.confidence)
                        continue

                    plate_detection = plate_detector.detect(vehicle_crop)
                    if plate_detection is None:
                        draw_vehicle_box(display_frame, (x1, y1, x2, y2), vehicle.label, vehicle.confidence)
                        continue

                    plate_x1, plate_y1, plate_x2, plate_y2 = plate_detection.bbox
                    global_plate_bbox = clip_bbox(
                        (x1 + plate_x1, y1 + plate_y1, x1 + plate_x2, y1 + plate_y2),
                        frame.shape[1],
                        frame.shape[0],
                    )
                    ocr_result = plate_reader.read_plate(plate_detection.image)
                    plate_text = ocr_result.text
                    plate_confidence = ocr_result.confidence if display_plate_confidence else plate_detection.confidence
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                    if plate_text and plate_text not in seen_session:
                        inserted = database_manager.save_plate(plate_text, timestamp)
                        if inserted:
                            seen_session.add(plate_text)
                            logging.info("Saved plate %s at %s", plate_text, timestamp)
                            if save_plate_images_flag:
                                save_plate_image(plate_detection.image, plate_images_dir, plate_text)

                    draw_vehicle_box(display_frame, (x1, y1, x2, y2), vehicle.label, vehicle.confidence)
                    draw_plate_box(
                        display_frame,
                        global_plate_bbox,
                        plate_text if plate_text else "",
                        plate_confidence,
                        show_confidence=display_plate_confidence,
                    )

                fps = fps_counter.update()
                processing_time_ms = (time.perf_counter() - frame_start) * 1000.0
                draw_fps(display_frame, fps)
                draw_status(display_frame, f"Processing: {processing_time_ms:.1f} ms", origin=(15, 58))

                cv2.imshow(window_title, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
