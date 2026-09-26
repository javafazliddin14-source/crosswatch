"""Per-track trajectories and kinematics from the raw tracker rows.

Positions use the bottom-centre of the box (the ground-contact point), which is
where a vehicle actually is on the road plane. Speeds are also given relative
to the box height ("own heights per second") so a single threshold works for
both near and far vehicles despite the perspective.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src import config as C
from src.detect_track import CLS, CONF, T, TID, X1, X2, Y1, Y2


@dataclass
class Track:
    tid: int
    cls: int                  # majority class over the track's life
    t: np.ndarray             # (n,) seconds
    x: np.ndarray             # (n,) ground-point x, normalised, smoothed
    y: np.ndarray             # (n,) ground-point y, normalised, smoothed
    w: np.ndarray             # (n,) box width, normalised
    h: np.ndarray             # (n,) box height, normalised
    conf: np.ndarray
    vx: np.ndarray = field(default=None)   # (n,) normalised units / s
    vy: np.ndarray = field(default=None)
    speed_rel: np.ndarray = field(default=None)  # |v| / box height  (heights per second)

    @property
    def start(self) -> float:
        return float(self.t[0])

    @property
    def end(self) -> float:
        return float(self.t[-1])

    @property
    def is_vehicle(self) -> bool:
        return self.cls in C.VEHICLES

    @property
    def is_person(self) -> bool:
        return self.cls == C.PERSON

    def at(self, t: float) -> int | None:
        """Index of the sample at time t (nearest, within half a sample period)."""
        i = int(np.searchsorted(self.t, t))
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.t) and abs(self.t[j] - t) <= 0.6 / C.SAMPLE_FPS:
                if best is None or abs(self.t[j] - t) < abs(self.t[best] - t):
                    best = j
        return best

    def box(self, i: int) -> tuple[float, float, float, float]:
        return (self.x[i] - self.w[i] / 2, self.y[i] - self.h[i], self.x[i] + self.w[i] / 2, self.y[i])


def _smooth(a: np.ndarray, k: int) -> np.ndarray:
    if len(a) < k or k <= 1:
        return a.copy()
    pad = k // 2
    ap = np.pad(a, pad, mode="edge")
    return np.convolve(ap, np.ones(k) / k, mode="valid")


def _velocity(t: np.ndarray, p: np.ndarray, half_window_s: float = 0.6) -> np.ndarray:
    """Central-difference velocity over +-half_window_s (robust to jitter)."""
    v = np.zeros_like(p)
    if len(t) < 2:
        return v
    lo = np.searchsorted(t, t - half_window_s)
    hi = np.searchsorted(t, t + half_window_s, side="right") - 1
    dt = t[hi] - t[lo]
    ok = dt > 1e-6
    v[ok] = (p[hi[ok]] - p[lo[ok]]) / dt[ok]
    return v


def build_tracks(rows: np.ndarray, min_len: int = 3) -> dict[int, Track]:
    tracks: dict[int, Track] = {}
    if len(rows) == 0:
        return tracks
    rows = rows[np.lexsort((rows[:, T], rows[:, TID]))]
    ids, starts = np.unique(rows[:, TID], return_index=True)
    bounds = list(starts) + [len(rows)]
    for k, tid in enumerate(ids):
        r = rows[bounds[k]:bounds[k + 1]]
        if len(r) < min_len:
            continue
        cls_vals, counts = np.unique(r[:, CLS].astype(int), return_counts=True)
        cls = int(cls_vals[np.argmax(counts)])
        w = r[:, X2] - r[:, X1]
        h = r[:, Y2] - r[:, Y1]
        x = (r[:, X1] + r[:, X2]) / 2
        y = r[:, Y2]
        tr = Track(tid=int(tid), cls=cls, t=r[:, T].astype(np.float64),
                   x=_smooth(x, 3), y=_smooth(y, 3), w=_smooth(w, 3), h=_smooth(h, 3), conf=r[:, CONF])
        tr.vx = _velocity(tr.t, tr.x)
        tr.vy = _velocity(tr.t, tr.y)
        tr.speed_rel = np.hypot(tr.vx, tr.vy) / np.maximum(tr.h, 1e-3)
        tracks[tr.tid] = tr
    return tracks


def frames_index(tracks: dict[int, Track]) -> dict[float, list[tuple[Track, int]]]:
    """time -> [(track, sample index)] for every sampled frame that has boxes."""
    idx: dict[float, list[tuple[Track, int]]] = {}
    for tr in tracks.values():
        for i, t in enumerate(tr.t):
            idx.setdefault(round(float(t), 3), []).append((tr, i))
    return idx
