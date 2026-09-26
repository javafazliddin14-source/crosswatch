"""Annotated, browser-playable (H.264) videos of the pipeline output."""
from __future__ import annotations

import cv2
import numpy as np

from src import config as C
from src import registration
from src.detect_track import CLS, T, TID, X1, X2, Y1, Y2
from src.scene import SCENE
from src.signals import GREEN, RED
from src.video import VideoMeta, iter_frames

EVENT_COLORS = {  # BGR
    "accident": (40, 40, 230), "near_miss": (0, 140, 255), "red_light": (60, 60, 255),
    "wrong_way": (200, 0, 200), "illegal_u_turn": (180, 80, 220), "stopped_vehicle": (0, 200, 255),
    "jaywalking": (255, 170, 0), "failure_to_yield": (255, 90, 90), "illegal_turn": (160, 100, 255),
    "solid_line_crossing": (100, 220, 220), "stop_line": (80, 160, 255), "congestion": (80, 80, 180),
    "road_obstacle": (0, 255, 160), "fire_smoke": (0, 0, 180),
}
CLASS_COLORS = {C.PERSON: (255, 200, 0), C.BICYCLE: (255, 120, 0), C.CAR: (80, 220, 80),
                C.MOTORCYCLE: (0, 200, 255), C.BUS: (220, 120, 255), C.TRUCK: (255, 120, 180)}


def _writer(path: str, size: tuple[int, int], fps: float):
    import imageio_ffmpeg

    gen = imageio_ffmpeg.write_frames(path, size, fps=fps, codec="libx264", pix_fmt_in="bgr24",
                                      output_params=["-crf", "28", "-preset", "veryfast", "-movflags", "+faststart"],
                                      macro_block_size=2)   # imageio encodes yuv420p: plays in every browser
    gen.send(None)
    return gen


def _draw_scene(img: np.ndarray, H_inv: np.ndarray) -> None:
    """Crossings and stop line, mapped from the reference view into this video."""
    h, w = img.shape[:2]
    over = img.copy()
    for poly in SCENE.crosswalks.values():
        cv2.fillPoly(over, [(registration.apply(H_inv, poly) * [w, h]).astype(np.int32)], (255, 255, 255))
    cv2.addWeighted(over, 0.12, img, 0.88, 0, img)
    (x1, y1), (x2, y2) = registration.apply(H_inv, SCENE.stop_line)
    cv2.line(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 0, 255), 2)


def render(meta: VideoMeta, rows: np.ndarray, events: list[list], out_path: str, H: np.ndarray | None = None,
           signal_times: np.ndarray | None = None, signal: np.ndarray | None = None,
           risk: list[list] | None = None, fps: float = 10.0, width: int = 960, progress=None) -> None:
    by_t: dict[float, np.ndarray] = {}
    if len(rows):
        ts = np.round(rows[:, T], 3)
        for t in np.unique(ts):
            by_t[float(t)] = rows[ts == t]
    box_times = np.array(sorted(by_t))
    risk_arr = np.array(risk, np.float32) if risk else None
    H_inv = np.linalg.inv(H) if H is not None else np.eye(3)

    gen = None
    n_total = max(1, int(meta.duration * fps))
    for k, (t, frame) in enumerate(iter_frames(meta, fps, width)):
        img = frame.copy()
        h, w = img.shape[:2]
        if gen is None:
            gen = _writer(out_path, (w, h), fps)
        _draw_scene(img, H_inv)
        # nearest detection sample (<= 0.2 s old)
        if len(box_times):
            i = int(np.searchsorted(box_times, t + 1e-6)) - 1
            if i >= 0 and t - box_times[i] <= 0.25:
                for r in by_t[float(box_times[i])]:
                    col = CLASS_COLORS.get(int(r[CLS]), (200, 200, 200))
                    p1 = (int(r[X1] * w), int(r[Y1] * h))
                    p2 = (int(r[X2] * w), int(r[Y2] * h))
                    cv2.rectangle(img, p1, p2, col, 1)
                    cv2.putText(img, f"{C.CLASS_NAMES.get(int(r[CLS]), '?')} {int(r[TID])}", (p1[0], p1[1] - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
        # active events banner
        active = [e for e in events if e[0] <= t < e[1]]
        y0 = 22
        for s, e, lab in active:
            col = EVENT_COLORS.get(lab, (255, 255, 255))
            txt = f"{lab.replace('_', ' ').upper()}  {s:.1f}-{e:.1f}s"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img, (8, y0 - th - 6), (16 + tw, y0 + 5), col, -1)
            cv2.putText(img, txt, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            y0 += th + 14
        # clock, signal, risk
        hud = f"t={t:6.1f}s"
        cv2.putText(img, hud, (w - 130, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        if signal is not None and len(signal):
            st = int(signal[min(len(signal) - 1, int(np.searchsorted(signal_times, t)))])
            col = (0, 0, 255) if st == RED else (0, 200, 0) if st == GREEN else (128, 128, 128)
            cv2.circle(img, (w - 150, 16), 7, col, -1)
        if risk_arr is not None and len(risk_arr):
            j = min(len(risk_arr) - 1, int(np.searchsorted(risk_arr[:, 0], t)))
            r = float(risk_arr[j, 1])
            bw = int(120 * r)
            cv2.rectangle(img, (w - 130, 32), (w - 10, 42), (60, 60, 60), -1)
            cv2.rectangle(img, (w - 130, 32), (w - 130 + bw, 42), (0, 0, 255) if r >= 0.5 else (0, 200, 255), -1)
            cv2.putText(img, "risk", (w - 170, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        gen.send(np.ascontiguousarray(img))
        if progress is not None and k % 20 == 0:
            progress(min(1.0, k / n_total))
    if gen is not None:
        gen.close()
