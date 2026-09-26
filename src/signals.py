"""Traffic-signal state read straight from the pixels of the signal heads that face the camera.

Two heads are visible from this camera and switch together (verified on the
samples): a vehicle head on the median pole and a pedestrian head on the
left-hand pole. We count bright, saturated red and green pixels in each lamp
region; combining both heads survives one of them being occluded by a truck.
"""
from __future__ import annotations

import cv2
import numpy as np

RED, GREEN, UNKNOWN = 1, 0, -1


def lamp_scores(crops: list[np.ndarray]) -> tuple[float, float]:
    """(red, green) = fraction of lit red / green pixels over the signal-head crops.

    `crops` are the SCENE.signal_heads regions cut from the full-resolution frame.
    """
    red = green = total = 0
    for crop in crops:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hch, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lit = (s > 80) & (v > 110)
        red += int(np.count_nonzero(lit & ((hch < 10) | (hch > 168))))
        green += int(np.count_nonzero(lit & (hch > 55) & (hch < 100)))
        total += hch.size
    return red / max(total, 1), green / max(total, 1)


def classify(red: float, green: float, min_frac: float = 0.0015) -> int:
    if max(red, green) < min_frac:
        return UNKNOWN
    return RED if red > green else GREEN


def state_timeline(scores: np.ndarray, min_hold: int = 5) -> np.ndarray:
    """Per-sample state from an (n, 2) array of (red, green) scores.

    Unknown samples inherit the previous known state, and a switch must hold
    for `min_hold` consecutive samples (1 s at 5 Hz) to suppress flicker from
    passing cars' tail lights.
    """
    raw = np.array([classify(r, g) for r, g in scores], dtype=int)
    known = raw[raw != UNKNOWN]
    cur = int(known[0]) if len(known) else UNKNOWN
    out = np.empty_like(raw)
    run_val, run_len = cur, 0
    for i, s in enumerate(raw):
        if s != UNKNOWN and s != cur:
            run_len = run_len + 1 if s == run_val else 1
            run_val = s
            if run_len >= min_hold:
                cur = s
                out[i - min_hold + 1:i] = s    # the switch happened when the run began
        else:
            run_len = 0
        out[i] = cur
    return out
