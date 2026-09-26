"""Calibrate Part B's hazard -> probability mapping on the sample videos.

    python -m tools.calibrate_risk --videos C:/data/samples

The samples contain no accident, so they define "normal traffic": we report
the hazard distribution and how many alarm runs (score >= 0.5, runs < 2 s
apart merged, as in evaluate.py) each candidate CALIB_B would raise. Pick the
smallest B that keeps false alarms to ~0 on normal footage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src import config as C
from src.risk import RiskModel
from src.video import iter_frames, probe


def hazard_series(path: str) -> np.ndarray:
    meta = probe(path)
    model = RiskModel(fps=1.0 / C.RISK_DETECT_EVERY_SEC)   # we already feed 5 Hz frames: detect on each
    out = []
    for t, frame in iter_frames(meta, 1.0 / C.RISK_DETECT_EVERY_SEC, C.RISK_INFER_WIDTH):
        model.step(frame, t)
        out.append((t, *model.last_components))
    return np.array(out)


def alarm_runs(times: np.ndarray, on: np.ndarray, merge_gap: float = 2.0) -> int:
    runs, last_end = 0, None
    prev = False
    for t, o in zip(times, on):
        if o and not prev:
            if last_end is None or t - last_end >= merge_gap:
                runs += 1
        if o:
            last_end = t
        prev = o
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    args = ap.parse_args()
    series = {}
    for p in sorted(Path(args.videos).glob("*.[mM][pP]4")):
        cache = C.CACHE_DIR / f"hazard_{p.stem}.npy"
        if cache.exists():
            series[p.name] = np.load(cache)
        else:
            series[p.name] = hazard_series(str(p))
            np.save(cache, series[p.name])
        s = series[p.name]
        print(f"{p.name}: conflict p50/p99/max {np.percentile(s[:, 1], 50):.2f}/{np.percentile(s[:, 1], 99):.2f}/{s[:, 1].max():.2f}"
              f"  brake p50/p99/max {np.percentile(s[:, 2], 50):.2f}/{np.percentile(s[:, 2], 99):.2f}/{s[:, 2].max():.2f}")
    from src.risk import BRAKE_WEIGHT
    for b in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        n = 0
        for s in series.values():
            hz = np.maximum.reduce([s[:, 1], BRAKE_WEIGHT * s[:, 2], 0.5 * (s[:, 1] + s[:, 2])])
            n += alarm_runs(s[:, 0], hz >= b)
        mins = sum(s[-1, 0] for s in series.values()) / 60
        print(f"CALIB_B={b:.2f}: {n} alarm runs over {mins:.1f} min of normal traffic")


if __name__ == "__main__":
    main()
