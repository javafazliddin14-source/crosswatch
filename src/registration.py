"""Register each video to the reference view the scene geometry was measured in.

"Same camera and angle" is only approximately true: across the samples the
framing shifts by up to ~3 % of the frame (re-aimed / re-zoomed camera). All
scene geometry (crossings, stop line, lamp boxes, zones, learned flow field)
lives in the reference view (sample C3896, weights/scene_ref.jpg), so for
every video we estimate a homography from a median background plate to the
reference with SIFT + RANSAC and
  * map track coordinates INTO the reference view before the rules run;
  * map the reference lamp boxes OUT to video pixels before decoding.
Falls back to the identity when registration is not trustworthy.
"""
from __future__ import annotations

import cv2
import numpy as np

from src import config as C
from src.video import VideoMeta

REF_PATH = C.WEIGHTS_DIR / "scene_ref.jpg"
W, H = 1280, 720           # registration works at this resolution (the reference image's size)
N_FRAMES = 9
MIN_INLIERS = 40
MAX_SHIFT = 0.15           # reject transforms moving a corner by more than 15 % of the frame

_ref_cache = None


def _prep(gray: np.ndarray) -> np.ndarray:
    """Local contrast normalisation so midday and dusk frames still match."""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _reference():
    global _ref_cache
    if _ref_cache is None:
        img = cv2.imread(str(REF_PATH), cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT_create(4000)
        _ref_cache = sift.detectAndCompute(_prep(img), None)
    return _ref_cache


def background(meta: VideoMeta, n: int = N_FRAMES) -> np.ndarray:
    """Median of n frames spread over the clip (moving traffic vanishes), gray, W x H."""
    cap = cv2.VideoCapture(meta.path)
    frames = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) / n * max(1, meta.n_frames - 1)))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        raise RuntimeError(f"cannot read frames from {meta.path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def estimate(meta: VideoMeta) -> np.ndarray:
    """3x3 homography mapping NORMALISED video coords -> normalised reference coords."""
    if not REF_PATH.exists():
        return np.eye(3)
    k_ref, d_ref = _reference()
    k_vid, d_vid = cv2.SIFT_create(4000).detectAndCompute(_prep(background(meta)), None)
    if d_vid is None or len(k_vid) < MIN_INLIERS:
        return np.eye(3)
    matches = cv2.BFMatcher().knnMatch(d_vid, d_ref, k=2)
    good = [m for m, n2 in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n2.distance]
    if len(good) < MIN_INLIERS:
        return np.eye(3)
    src = np.float32([k_vid[g.queryIdx].pt for g in good])
    dst = np.float32([k_ref[g.trainIdx].pt for g in good])
    Hpx, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if Hpx is None or int(mask.sum()) < MIN_INLIERS:
        return np.eye(3)
    S = np.diag([W, H, 1.0])
    Hn = np.linalg.inv(S) @ Hpx @ S
    corners = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    if np.abs(apply(Hn, corners) - corners).max() > MAX_SHIFT:
        return np.eye(3)
    return Hn


def apply(Hn: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map (n, 2) normalised points through a normalised homography."""
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, Hn).reshape(-1, 2)


def map_boxes(Hn: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """(n, 4) x1, y1, x2, y2 -> axis-aligned bounding boxes of the mapped corners."""
    if len(boxes) == 0:
        return boxes
    x1, y1, x2, y2 = boxes.T
    corners = np.stack([np.stack([x1, y1], 1), np.stack([x2, y1], 1), np.stack([x2, y2], 1), np.stack([x1, y2], 1)], 1)
    m = apply(Hn, corners.reshape(-1, 2)).reshape(-1, 4, 2)
    return np.concatenate([m.min(1), m.max(1)], 1).astype(boxes.dtype)
