"""Full analysis of one video for the website: events, risk curve, stats and an annotated video.

Used by the live demo (website/server.py) and by tools/export_results.py for
the sample videos, so both show exactly what the submission computes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import config as C
from src.detect_track import CLS, T
from src.pipeline import analyze
from src.render import render
from src.risk import RiskModel
from src.video import iter_frames


def risk_curve(meta, progress=None, width: int = C.RISK_INFER_WIDTH) -> list[list[float]]:
    """Stream every frame through the causal RiskModel, as the harness does (frames pre-scaled)."""
    model = RiskModel(fps=meta.fps)
    curve = []
    n = max(1, meta.n_frames)
    for k, (t, frame) in enumerate(iter_frames(meta, meta.fps, width)):
        curve.append([round(t, 3), round(model.step(frame, t), 4)])
        if progress is not None and k % 50 == 0:
            progress(min(1.0, k / n))
    return curve


def per_second_counts(rows: np.ndarray, duration: float) -> dict[str, list[float]]:
    """Mean number of detected objects per sampled frame, per second, by class."""
    n = int(np.ceil(duration))
    out = {}
    if not len(rows):
        return out
    sec = np.clip(rows[:, T].astype(int), 0, max(0, n - 1))
    frames_per_sec = np.bincount(np.unique(np.round(rows[:, T], 3)).astype(int).clip(0, n - 1), minlength=n)
    for c in (C.PERSON, C.CAR, C.BUS, C.TRUCK, C.MOTORCYCLE, C.BICYCLE):
        cnt = np.bincount(sec[rows[:, CLS] == c], minlength=n).astype(float)
        out[C.CLASS_NAMES[c]] = np.round(cnt / np.maximum(frames_per_sec, 1), 2).tolist()
    return out


def analyze_video(video_path: str, out_dir: str, progress=None, render_fps: float = 10.0,
                  precomputed: dict | None = None) -> dict:
    """`precomputed` = one video's entry of a predictions.json ({"events", "risk"}): reuse the
    harness output instead of recomputing, so the website shows exactly what was submitted."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = progress or (lambda stage, p: None)

    res = analyze(video_path, progress=lambda p: report("detecting & tracking", 0.55 * p))
    if precomputed is not None:
        res.events = precomputed["events"]
        curve = precomputed["risk"]
    else:
        curve = risk_curve(res.meta, progress=lambda p: report("accident risk (causal)", 0.55 + 0.25 * p))
    render(res.meta, res.rows, res.events, str(out / "annotated.mp4"), res.H, res.times, res.signal,
           risk=curve, fps=render_fps, progress=lambda p: report("rendering video", 0.8 + 0.2 * p))
    step = max(1, round(res.meta.fps / 5))          # 5 Hz is plenty for charts
    result = {
        "video": Path(video_path).name,
        "meta": {"fps": res.meta.fps, "width": res.meta.width, "height": res.meta.height,
                 "n_frames": res.meta.n_frames, "duration": round(res.meta.duration, 2)},
        "events": res.events,
        "risk": curve[::step],
        "signal": [[round(float(t), 2), int(s)] for t, s in zip(res.times[::5], res.signal[::5])],
        "counts": per_second_counts(res.rows, res.meta.duration),
        "n_tracks": {name: sum(1 for tr in res.tracks.values() if tr.cls == c)
                     for c, name in C.CLASS_NAMES.items()
                     if any(tr.cls == c for tr in res.tracks.values())},
    }
    (out / "result.json").write_text(json.dumps(result))
    return result
