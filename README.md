# Automatic Number Plate Recognition System
> Last automated login update: 2026-04-24 11:53:29

A production-oriented Automatic Number Plate Recognition (ANPR) system built with **OpenCV**, **YOLOv8**, and **Tesseract OCR**.

The application detects vehicles in real time, locates license plates, extracts plate text, displays annotated video, and stores unique plate numbers in SQLite or CSV.

## Features

- Real-time vehicle detection using `YOLOv8n`
- License plate detection on each detected vehicle
- OCR-based plate reading with image preprocessing
- Live bounding boxes for vehicles and plates
- On-screen plate text and confidence display
- Real-time FPS counter
- Duplicate prevention for saved plate numbers
- SQLite and CSV logging
- Webcam or video file support
- Optional plate image saving

## Project Structure

```text
anpr_project/
  main.py
  detector/
    vehicle_detector.py
    plate_detector.py
  ocr/
    plate_reader.py
  database/
    db_manager.py
  utils/
    draw.py
    fps.py
    image_processing.py
  config/
    config.yaml
  output/
    plates_log.csv
  requirements.txt
```

## Requirements

- Python 3.10 or later
- Windows, Linux, or macOS
- Tesseract OCR installed on the system
- Webcam or a video file for input

## Installation

1. Clone or open the project folder.
2. Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Install Tesseract OCR if it is not already installed.
   - On Windows, the app looks for:
     - `C:\Program Files\Tesseract-OCR\tesseract.exe`
     - `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`

## Configuration

Edit [config/config.yaml](config/config.yaml) to adjust the runtime behavior:

```yaml
video_source: 0
confidence_threshold: 0.5
save_to_database: true
save_to_csv: true
sqlite_path: output/plates.db
csv_path: output/plates_log.csv
tesseract_cmd: C:\Program Files\Tesseract-OCR\tesseract.exe
```

### Common Settings

- `video_source`: `0` for webcam, or a video file path
- `confidence_threshold`: OCR/plate detection confidence threshold
- `vehicle_confidence_threshold`: vehicle detection confidence threshold
- `frame_resize_width`: resize input frame for faster processing
- `save_to_database`: enable SQLite storage
- `save_to_csv`: enable CSV storage
- `save_plate_images`: save cropped plate images
- `tesseract_cmd`: full path to the Tesseract executable

## Run the Application

From the project directory, run:

```powershell
python main.py
```

The window title is:

```text
Automatic Number Plate Recognition System
```

Press `q` to exit.

## Output

Detected plates are saved to:

- SQLite database: `output/plates.db`
- CSV log: `output/plates_log.csv`

If enabled, plate images are saved to:

- `output/plates/`

## Detection Pipeline

1. Capture webcam or video frame
2. Detect vehicles with YOLOv8
3. Crop each detected vehicle region
4. Detect license plate region
5. Preprocess the plate image
6. Read the plate text using Tesseract OCR
7. Draw annotations and show FPS
8. Save unique results to storage

## Troubleshooting

### Tesseract not found

If OCR fails at startup, set `tesseract_cmd` in [config/config.yaml](config/config.yaml) to your installed executable path.

### Webcam not opening

If `video_source: 0` does not work, try:

- another webcam index such as `1`
- a full path to a video file

### Slow performance

To improve real-time speed:

- lower `frame_resize_width`
- use a faster GPU-enabled environment if available
- reduce unnecessary background applications

## Notes

- The first run may download `yolov8n.pt` automatically.
- Plate detection can use either the contour-based fallback or a custom YOLO plate model if provided in the config.

## License

This project is licensed under the [MIT License](LICENSE).
