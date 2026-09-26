"""Turning per-time flags into clean [start, end] segments."""
from __future__ import annotations

import numpy as np


def flags_to_segments(times: np.ndarray, flags: np.ndarray, dt: float) -> list[tuple[float, float]]:
    """Maximal runs of True in `flags` (sampled at `times`) -> [(start, end)].

    A run covering samples t_i..t_j becomes [t_i, t_j + dt]: each sample stands
    for the interval up to the next one.
    """
    segs, start = [], None
    for t, f in zip(times, flags):
        if f and start is None:
            start = t
        elif not f and start is not None:
            segs.append((float(start), float(prev) + dt))
            start = None
        prev = t
    if start is not None:
        segs.append((float(start), float(prev) + dt))
    return segs


def merge_segments(segs: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    """Union overlapping segments and those closer than `gap` seconds."""
    out: list[list[float]] = []
    for s, e in sorted(segs):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def clean(segs: list[tuple[float, float]], gap: float, min_len: float, duration: float) -> list[tuple[float, float]]:
    """Merge fragments, drop blips, clip to the video."""
    out = []
    for s, e in merge_segments(segs, gap):
        s, e = max(0.0, s), min(duration, e)
        if e - s >= min_len:
            out.append((round(s, 2), round(e, 2)))
    return out
