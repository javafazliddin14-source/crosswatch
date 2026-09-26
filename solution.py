"""
solution.py - entry point imported by the organizers' harness (run_submission.py).

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B

Part A: YOLO11 + ByteTrack at 5 Hz -> trajectories; traffic-signal state read
from the camera-facing signal heads; per-class rules on trajectories + scene
layout (src/rules.py). Part B: a causal tracker on the streamed frames and a
time-to-collision / hard-braking risk model (src/risk.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import analyze  # noqa: E402
from src.risk import RiskModel  # noqa: E402

# Official class ids (14). Remove entries you never predict; never add.
CLASSES: list[str] = [
    "accident",            # collision between road users / with a fixed object
    "near_miss",           # sharp braking or swerving to avoid a collision, no contact
    "red_light",           # crossing the stop line on red
    "wrong_way",           # driving against the traffic direction / in the oncoming lane
    "illegal_u_turn",      # U-turn where prohibited
    "stopped_vehicle",     # stationary on the carriageway >= 10 s, not queued at a signal
    "jaywalking",          # pedestrian on the carriageway outside a crossing
    "failure_to_yield",    # driving through a crossing while a pedestrian is on it
    "illegal_turn",        # turn from the wrong lane or in a prohibited direction
    "solid_line_crossing", # lane change / manoeuvre across a solid marking
    "stop_line",           # stopped past the stop line on red
    "congestion",          # standstill / crawling traffic across all lanes of a direction
    "road_obstacle",       # debris, animal or fallen object on the carriageway
    "fire_smoke",          # visible fire or smoke from a vehicle or on the road
]

RISK_HORIZON_SEC = 5.0


def detect_events(video_path: str) -> list[list]:
    """Part A - [[start_sec, end_sec, label], ...]; same-class segments never overlap."""
    return [ev for ev in analyze(video_path).events if ev[2] in CLASSES]


class RiskEstimator:
    """Part B - causal: sees frames in order, never opens the video, never uses Part A output."""

    def reset(self, meta: dict) -> None:
        self.model = RiskModel(fps=float(meta.get("fps") or 25.0))

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return self.model.step(frame, t_sec)
