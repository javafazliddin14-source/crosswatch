"""Part B - causal accident-risk model.

Runs its own detector + tracker on every k-th streamed frame (5 Hz), keeps a
short history per track and scores two classic precursors of a collision:

* time-to-collision (TTC) between road users: footprints (bottom slice of the
  box, ~ the ground-contact area) extrapolated at constant velocity for up
  to 3 s; the first time two footprints overlap is the TTC. Only pairs that
  are actually closing fast count, so slow queues don't raise alarms;
* hard braking: a vehicle losing most of its speed within ~1 s.

The raw hazard is mapped through a logistic calibrated so that ordinary
traffic on the sample videos stays well below the 0.5 alarm threshold, and a
short decay keeps an alarm up for a moment after the precursor disappears.
Strictly causal: only frames already passed to `step` are used.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

import cv2
import numpy as np

from src import config as C
from src.detect_track import device, load_model

HISTORY_S = 2.0
HORIZON_S = 3.0          # TTC look-ahead
TTC_STEP_S = 0.1
MIN_CLOSING = 0.8        # heights / s: pairs closing slower than this are ignored
MIN_SPEED = 1.0          # heights / s: at least one of the pair must move this fast
TTC_SCALE = 0.8          # s: conflict term = exp(-TTC / TTC_SCALE) * closing factor
CLOSING_FULL = 3.0       # heights / s of closing speed that counts fully
BRAKE_START, BRAKE_SPAN = 2.5, 3.0   # heights/s lost within ~1 s: ordinary stops stay below 2.5
BRAKE_WEIGHT = 0.8
DECAY_PER_S = 0.5        # score multiplier per second once the hazard is gone
CALIB_A, CALIB_B = 10.0, 0.72  # p = sigmoid(A * (hazard - B)); normal traffic peaks at 0.63 (tools/calibrate_risk.py)


class RiskModel:
    def __init__(self, fps: float):
        self.fps = fps
        self.every = max(1, round(fps * C.RISK_DETECT_EVERY_SEC))
        self.n = 0
        self.score = 0.0
        self.last_t = 0.0
        self.hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=int(HISTORY_S * C.SAMPLE_FPS) + 2))
        self.cls: dict[int, int] = {}
        self.first = True
        self.last_components = (0.0, 0.0)
        self.model = load_model()
        self.dev = device()

    # ---- detection ----------------------------------------------------------------
    def _detect(self, frame: np.ndarray, t: float) -> list[int]:
        h, w = frame.shape[:2]
        scale = C.RISK_INFER_WIDTH / w
        small = cv2.resize(frame, (C.RISK_INFER_WIDTH, int(h * scale) // 2 * 2), interpolation=cv2.INTER_AREA) \
            if scale < 1 else frame
        res = self.model.track(small, persist=not self.first, tracker=str(C.TRACKER_CFG), imgsz=C.RISK_IMGSZ,
                               conf=C.DET_CONF, iou=0.5, classes=C.TRACK_CLASSES, device=self.dev,
                               quantize="fp16" if self.dev != "cpu" else None, verbose=False)[0]
        self.first = False
        seen = []
        b = res.boxes
        if b is None or b.id is None:
            return seen
        sh, sw = small.shape[:2]
        for (x1, y1, x2, y2), tid, c in zip(b.xyxy.cpu().numpy(), b.id.cpu().numpy().astype(int),
                                            b.cls.cpu().numpy().astype(int)):
            self.hist[tid].append((t, (x1 + x2) / 2 / sw, y2 / sh, (x2 - x1) / sw, (y2 - y1) / sh))
            self.cls[tid] = c
            seen.append(tid)
        return seen

    # ---- kinematics ---------------------------------------------------------------
    def _state(self, tid: int):
        """(x, y, w, h, vx, vy, speed_rel, decel_rel) from the track history, or None."""
        hs = self.hist[tid]
        if len(hs) < 3:
            return None
        t, x, y, w, h = map(np.array, zip(*hs))
        span = t[-1] - t[0]
        if span < 0.3:
            return None
        k = max(0, len(t) - 1 - int(0.6 * C.SAMPLE_FPS))   # velocity over the last ~0.6 s
        dt = max(t[-1] - t[k], 1e-3)
        vx, vy = (x[-1] - x[k]) / dt, (y[-1] - y[k]) / dt
        hh = max(float(np.median(h)), 1e-3)
        speed = math.hypot(vx, vy) / hh
        # speed ~1 s ago, for braking
        decel = 0.0
        if span >= 1.2:
            j = int(np.searchsorted(t, t[-1] - 1.2))
            j2 = min(j + 3, len(t) - 1)
            dt0 = max(t[j2] - t[j], 1e-3)
            v0 = math.hypot(x[j2] - x[j], y[j2] - y[j]) / dt0 / hh
            decel = max(0.0, v0 - speed)
        return x[-1], y[-1], float(np.median(w[-3:])), hh, vx, vy, speed, decel

    @staticmethod
    def _footprint(x, y, w, h):
        d = 0.3 * h
        return x - w / 2, y - d, x + w / 2, y

    @staticmethod
    def _overlap(fa, fb) -> bool:
        return fa[0] < fb[2] and fb[0] < fa[2] and fa[1] < fb[3] and fb[1] < fa[3]

    def _ttc(self, a, b) -> tuple[float, float] | None:
        """(time to footprint contact, closing speed in heights/s) or None if not converging.

        Footprints that already overlap are skipped: from this elevated view that is
        a queued car occluding the next one, not a prediction of anything.
        """
        ax, ay, aw, ah, avx, avy, asp, _ = a
        bx, by, bw, bh, bvx, bvy, bsp, _ = b
        if max(asp, bsp) < MIN_SPEED:
            return None
        rx, ry = bx - ax, by - ay
        rvx, rvy = bvx - avx, bvy - avy
        dist = math.hypot(rx, ry)
        closing = -(rx * rvx + ry * rvy) / max(dist, 1e-6) / max(min(ah, bh), 1e-3)
        if closing < MIN_CLOSING:
            return None
        if self._overlap(self._footprint(ax, ay, aw, ah), self._footprint(bx, by, bw, bh)):
            return None
        for s in np.arange(TTC_STEP_S, HORIZON_S + 1e-9, TTC_STEP_S):
            if self._overlap(self._footprint(ax + avx * s, ay + avy * s, aw, ah),
                             self._footprint(bx + bvx * s, by + bvy * s, bw, bh)):
                return float(s), closing
        return None

    def _hazard(self, seen: list[int]) -> tuple[float, float]:
        """(conflict term, braking term), each in [0, 1]."""
        states = {}
        for tid in seen:
            st = self._state(tid)
            if st is not None and self.cls.get(tid) in (*C.VEHICLES, C.PERSON, C.BICYCLE):
                states[tid] = st
        ids = list(states)
        conflict = brake = 0.0
        for i in range(len(ids)):
            a = states[ids[i]]
            veh_a = self.cls[ids[i]] in C.VEHICLES
            if veh_a:   # hard braking: far beyond an ordinary stop at the line
                brake = max(brake, min(1.0, max(0.0, a[7] - BRAKE_START) / BRAKE_SPAN))
            for j in range(i + 1, len(ids)):
                if not (veh_a or self.cls[ids[j]] in C.VEHICLES):
                    continue           # person-person pairs are not traffic conflicts
                b = states[ids[j]]
                if math.hypot(a[0] - b[0], a[1] - b[1]) > 0.25:
                    continue
                hit = self._ttc(a, b)
                if hit is not None:
                    ttc, closing = hit
                    conflict = max(conflict, math.exp(-ttc / TTC_SCALE) * min(1.0, closing / CLOSING_FULL))
        return conflict, brake

    # ---- public -------------------------------------------------------------------
    def step(self, frame: np.ndarray, t: float) -> float:
        k = self.n
        self.n += 1
        if k % self.every:
            return self.score
        seen = self._detect(frame, t)
        # drop tracks not seen for a while
        for tid in [tid for tid, hs in self.hist.items() if hs and t - hs[-1][0] > HISTORY_S]:
            del self.hist[tid]
        conflict, brake = self._hazard(seen)
        self.last_components = (conflict, brake)
        hz = max(conflict, BRAKE_WEIGHT * brake, 0.5 * (conflict + brake))
        p = 1.0 / (1.0 + math.exp(-CALIB_A * (hz - CALIB_B)))
        decayed = self.score * (DECAY_PER_S ** (t - self.last_t))
        self.score = float(max(p, decayed))
        self.last_t = t
        return self.score
