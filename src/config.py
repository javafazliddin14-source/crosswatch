"""All tunable constants in one place."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
CACHE_DIR = ROOT / ".cache"

SEED = 0
PIPELINE_VERSION = "v2"   # bump to invalidate cached tracks (v2: per-video registration)

# ---- detector / tracker (Part A) ----------------------------------------------
DETECTOR_WEIGHTS = WEIGHTS_DIR / "yolo11m.pt"
TRACKER_CFG = ROOT / "src" / "bytetrack.yaml"
SAMPLE_FPS = 5.0          # frames per second fed to the detector
INFER_WIDTH = 1280        # frames are downscaled to this width before inference
IMGSZ = 1280              # YOLO letterbox size
DET_CONF = 0.2

# COCO ids
PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK = 0, 1, 2, 3, 5, 7
VEHICLES = (CAR, MOTORCYCLE, BUS, TRUCK)
ANIMALS = (15, 16, 17, 18, 19)  # cat, dog, horse, sheep, cow
TRACK_CLASSES = [PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK, *ANIMALS]
CLASS_NAMES = {PERSON: "person", BICYCLE: "bicycle", CAR: "car", MOTORCYCLE: "motorcycle",
               BUS: "bus", TRUCK: "truck", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow"}

# ---- risk estimator (Part B) ---------------------------------------------------
RISK_DETECT_EVERY_SEC = 0.2   # run the detector on one frame every 0.2 s (5 Hz)
RISK_INFER_WIDTH = 960
RISK_IMGSZ = 960
