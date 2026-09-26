"""Road-user detection (YOLO) + multi-object tracking (ByteTrack).

Output is a flat float32 array, one row per (sampled frame, tracked box):

    t_sec, track_id, class_id, conf, x1, y1, x2, y2      (box coords normalised to [0, 1])

Normalised coordinates keep every downstream rule independent of the input
resolution and of the resize used for inference.
"""
from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np

from src import config as C
from src import registration
from src.scene import SCENE
from src.signals import lamp_scores
from src.video import VideoMeta, iter_frames

COLS = ("t", "tid", "cls", "conf", "x1", "y1", "x2", "y2")
T, TID, CLS, CONF, X1, Y1, X2, Y2 = range(len(COLS))

_MODEL = None


def seed_everything(seed: int = C.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_model():
    """YOLO weights are shipped in weights/ so the evaluation run needs no internet."""
    global _MODEL
    if _MODEL is None:
        os.environ.setdefault("YOLO_OFFLINE", "1")
        from ultralytics import YOLO

        seed_everything()
        _MODEL = YOLO(str(C.DETECTOR_WEIGHTS), task="detect")
    return _MODEL


def _cache_path(meta: VideoMeta) -> Path:
    st = os.stat(meta.path)
    key = f"{Path(meta.path).name}|{st.st_size}|{int(st.st_mtime)}|{C.PIPELINE_VERSION}|{C.SAMPLE_FPS}|{C.INFER_WIDTH}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return C.CACHE_DIR / f"tracks_{Path(meta.path).stem}_{digest}.npz"


def track_video(meta: VideoMeta, progress=None, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run detector + tracker over the whole video at C.SAMPLE_FPS. Cached on disk.

    Returns (rows, signal, H):
      rows    tracker rows in this video's own normalised coordinates (see COLS);
      signal  (n_samples, 3): t, red score, green score of the camera-facing signal heads;
      H       normalised homography video -> reference view (src/registration.py).
    """
    cache = _cache_path(meta)
    if use_cache and cache.exists():
        d = np.load(cache)
        return d["rows"], d["signal"], d["H"]

    H = registration.estimate(meta)
    # lamp boxes are measured in the reference view: map them into this video (+ margin for sway)
    lamp_boxes = registration.map_boxes(np.linalg.inv(H), np.array(SCENE.signal_heads, np.float64))
    pad = np.array([-0.004, -0.004, 0.004, 0.004])
    lamp_boxes = np.clip(lamp_boxes + pad, 0.0, 1.0).tolist()

    model = load_model()
    dev = device()
    rows: list[np.ndarray] = []
    signal: list[tuple[float, float, float]] = []
    n_samples = max(1, int(meta.duration * C.SAMPLE_FPS))
    first = True
    frames = iter_frames(meta, C.SAMPLE_FPS, C.INFER_WIDTH, crops=lamp_boxes)
    for k, (t, frame, lamp_crops) in enumerate(frames):
        signal.append((t, *lamp_scores(lamp_crops)))
        res = model.track(frame, persist=not first, tracker=str(C.TRACKER_CFG), imgsz=C.IMGSZ,
                          conf=C.DET_CONF, iou=0.5, classes=C.TRACK_CLASSES, device=dev,
                          quantize="fp16" if dev != "cpu" else None, verbose=False)[0]
        first = False
        b = res.boxes
        if b is not None and b.id is not None and len(b):
            h, w = frame.shape[:2]
            xyxy = b.xyxy.cpu().numpy() / np.array([w, h, w, h], np.float32)
            out = np.empty((len(b), len(COLS)), np.float32)
            out[:, T] = t
            out[:, TID] = b.id.cpu().numpy()
            out[:, CLS] = b.cls.cpu().numpy()
            out[:, CONF] = b.conf.cpu().numpy()
            out[:, X1:] = xyxy
            rows.append(out)
        if progress is not None and k % 10 == 0:
            progress(min(1.0, (k + 1) / n_samples))
    arr = np.concatenate(rows) if rows else np.zeros((0, len(COLS)), np.float32)
    sig = np.array(signal, np.float32).reshape(-1, 3)
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, rows=arr, signal=sig, H=H)
    return arr, sig, H


def to_reference(rows: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Tracker rows with boxes mapped into the reference view (where all scene geometry lives)."""
    out = rows.copy()
    if len(out):
        out[:, X1:] = registration.map_boxes(H, rows[:, X1:].astype(np.float64))
    return out
