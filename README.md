markdown
# Object Detection System for Visually Impaired Persons — AI-Integrated Smart Glasses

A lightweight, real-time object detection and audio-notification system designed for smart glasses to assist visually impaired users. This repository provides reference scripts and guidance to run object detection on a camera feed (webcam / camera module on glasses / edge device), announce detected objects with TTS (text-to-speech), and prepare models for edge deployment.

Status: Draft / Example implementation — adapt models, device-specific optimizations, and TTS backend to your hardware.

---

## Table of contents

- Overview
- Key features
- Supported models & runtimes
- Hardware recommendations
- Quick start
- Usage examples
- Python API examples
- Training & dataset notes
- Exporting & optimizing models
- Project structure
- Troubleshooting
- Contributing
- License
- Short Hindi summary

---

## Overview

The goal of this project is to provide an accessible reference implementation for real-time object detection that:

- Detects objects in the user's view (camera mounted on glasses).
- Announces detected object labels, confidence, and relative positions (left/center/right, near/far).
- Is simple to run locally and portable to edge devices (Raspberry Pi, NVIDIA Jetson, Android glasses).

This repository is a starting point — it includes example code layout and instructions for running inference and TTS. Replace example models with your chosen architecture and optimize for your hardware.

---

## Key features

- Real-time object detection demo (camera or video).
- TTS-based audio feedback for detected objects.
- Relative-position hints (left/center/right) and distance heuristics.
- Example utilities to export/optimize models for edge deployment (ONNX / TFLite / TensorRT).
- Extensible: plug in different detection backbones (YOLOv5/YOLOv8, MobileNet-SSD, EfficientDet, etc.)

---

## Supported models & runtimes

Example workflows in this repo assume one of the following:

- PyTorch models (.pt) — YOLOv5 / YOLOv8 compatible
- ONNX models (.onnx) — run with onnxruntime
- TensorFlow Lite (.tflite) — for ARM devices (optional)
- TensorRT engines (Jetson) — for optimized NVIDIA inference

TTS libraries: pyttsx3 (offline), gTTS (online), or platform-specific TTS.

---

## Hardware recommendations

- Development: Desktop / Laptop with GPU (for training / profiling)
- Edge prototype: NVIDIA Jetson Nano / Xavier (GPU & TensorRT)
- Lightweight: Raspberry Pi 4 (with Coral USB TPU for acceleration) or Android-based smart glasses
- Camera: small wide-angle module mounted on frames or glasses

---

## Quick start (local)

1. Clone the repository:

bash
git clone https://github.com/mamatakhatke/Object-Detection-System-for-Visually-Impaired-Persons-Using-AI-Integrated-Smart-Glasses.git
cd Object-Detection-System-for-Visually-Impaired-Persons-Using-AI-Integrated-Smart-Glasses


2. (Optional) Create and activate a virtual environment:

bash
python3 -m venv venv
source venv/bin/activate


3. Install dependencies:

bash
pip install -r requirements.txt


4. Add model weights:

- Place your model weights under `models/`, e.g. `models/yolov5s.pt` or `models/model.onnx`.

5. Run inference (webcam):

bash
python src/inference.py --model models/yolov5s.pt --source 0 --tts --conf 0.35


Parameters:
- `--model` Path to model weights (.pt / .onnx / .tflite)
- `--source` Camera index (0) or path to video/image
- `--tts` Enable audio announcements
- `--conf` Minimum confidence threshold

---

## Usage examples

Run inference on a video file, no speech:

bash
python src/inference.py --model models/model.onnx --source demo/video.mp4 --no-tts


Run using CPU only:

bash
python src/inference.py --model models/yolov5s.pt --source 0 --device cpu


Export a PyTorch model to ONNX (YOLOv5 example):

bash
python export.py --weights models/yolov5s.pt --img 640 --include onnx


---

## Python API examples

Basic single-frame detection (example):

python
from src.detector import Detector
from src.tts import TTS
import cv2

detector = Detector(model_path="models/yolov5s.pt", device="cuda:0")
tts = TTS(voice="en", rate=150)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    detections = detector.predict(frame, conf_thres=0.35)
    # detections -> list of dicts: {label, conf, bbox, center, relative_pos, distance_est}
    for d in detections:
        tts.speak(f"{d['label']} {d['relative_pos']} {int(d['conf']*100)} percent")
    # optional: draw boxes / show frame


TTS helper (example using pyttsx3):

python
import pyttsx3

class TTS:
    def _init_(self, voice=None, rate=150):
        self.engine = pyttsx3.init()
        if voice:
            voices = self.engine.getProperty('voices')
            # pick voice if available
        self.engine.setProperty('rate', rate)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()


---

## Training & dataset notes

- Prepare dataset in YOLO or COCO format.
- For YOLOv5 training (example):

bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
python train.py --img 640 --batch 16 --epochs 50 --data data/custom.yaml --weights yolov5s.pt


- Fine-tune on domain-specific classes to reduce false positives and improve accuracy.

---

## Exporting & optimizing for edge

- Convert PyTorch (.pt) → ONNX → TensorRT (Jetson) or TFLite (ARM).
- Use lower input resolution and smaller models (YOLOv5n, MobileNet-SSD) for better FPS.
- Use hardware libraries: TensorRT (NVIDIA), OpenVINO (Intel), TFLite + Edge TPU (Coral).

Example ONNX export (YOLOv5):

bash
python export.py --weights models/yolov5s.pt --img 640 --include onnx


For Jetson, create a TensorRT engine via trtexec or a conversion script.

---

## Project structure (example)

- models/                # model weight files (do not commit large weights)
- src/
  - inference.py         # CLI entrypoint for camera / video inference
  - detector.py          # model wrapper: load, preprocess, postprocess
  - tts.py               # text-to-speech helper
  - utils.py             # drawing, position calculation, helpers
  - export.py            # optional: export utilities
- data/                  # dataset samples & annotation guidance
- scripts/               # conversion/optimization scripts
- requirements.txt
- README.md
- LICENSE

---

## Troubleshooting & tips

- Slow detection: switch to smaller model, reduce input size, or use hardware acceleration.
- Poor TTS quality: try different TTS backend and tune speaking rate.
- No detections: verify model weights and class mappings; test with a known example image.
- Announcing too frequently: debounce announcements or announce only new objects / changes.
- Position estimation: tune the mapping from bbox center to left/center/right based on camera FoV.

Suggested announcement policy:
- Announce an object when it first appears, then repeat only if it moves significantly or after a delay (e.g., 3–5 seconds).

---

## Contributing

Contributions are welcome — please follow these steps:

1. Open an issue describing the change/feature.
2. Fork the repository and create a branch for your work.
3. Add tests and update documentation where appropriate.
4. Submit a pull request referencing the issue.

Areas for contribution:
- Hardware-specific integrations (smart glasses SDKs)
- Additional model support (TFLite, OpenVINO)
- Improved relative distance estimation and context-aware announcements
- Unit / integration tests

---

## License

Choose a license for your project (e.g., MIT). Add a LICENSE file to the repo root.

---

## Acknowledgements

- YOLO (ultralytics) projects
- PyTorch, ONNX, TensorRT
- TTS libraries: pyttsx3, gTTS
- OpenCV for camera and image processing

---
