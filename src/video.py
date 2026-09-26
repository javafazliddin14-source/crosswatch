"""Video probing and fast, downscaled, sub-sampled frame reading.

Decoding full-resolution frames with OpenCV and throwing most of them away is
the single biggest cost of the pipeline. Instead we let ffmpeg (the static
binary shipped with the `imageio-ffmpeg` wheel, so nothing has to be installed
system-wide) decode with all cores, drop frames and resize before the pixels
ever reach Python.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    path: str
    fps: float
    width: int
    height: int
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def probe(path: str) -> VideoMeta:
    """Same numbers the organizers' harness sees (it uses OpenCV too)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    meta = VideoMeta(
        path=str(path),
        fps=float(fps),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return meta


def output_size(meta: VideoMeta, max_width: int) -> tuple[int, int]:
    """Downscaled (w, h), both even, keeping the aspect ratio."""
    if meta.width <= max_width:
        w, h = meta.width, meta.height
    else:
        w, h = max_width, round(meta.height * max_width / meta.width)
    return w - w % 2, h - h % 2


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


CROP_W, CROP_H = 48, 96   # size every full-resolution crop is resampled to


def _crop_px(meta: VideoMeta, box) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = int(x1 * meta.width), int(y1 * meta.height)
    return cx, cy, max(2, int((x2 - x1) * meta.width)), max(2, int((y2 - y1) * meta.height))


def iter_frames(meta: VideoMeta, sample_fps: float, max_width: int,
                crops: list | None = None) -> Iterator[tuple]:
    """Yield (t_sec, BGR frame) at roughly `sample_fps`, resized to `max_width`.

    With `crops` (normalised boxes) yields (t_sec, frame, [crop, ...]) where the
    crops are cut from the FULL-resolution frame (then resampled to CROP_W x
    CROP_H) - used for the tiny signal lamps, which downscaling would wash out.
    Timestamps are derived from source frame indices so they line up with the
    harness (t = idx / fps). Uses ffmpeg when available, OpenCV otherwise.
    """
    step = max(1, round(meta.fps / sample_fps))
    w, h = output_size(meta, max_width)
    exe = _ffmpeg_exe()
    if exe is None:
        yield from _iter_opencv(meta, step, (w, h), crops)
        return

    # select every `step`-th decoded frame; -vsync passthrough keeps 1 output per selected input
    sel = f"select='not(mod(n\\,{step}))'"
    extra_h = 0
    if crops:
        n = len(crops)
        extra_h = CROP_H
        parts = [f"[0:v]{sel},split={n + 1}[full]{''.join(f'[c{i}in]' for i in range(n))}",
                 f"[full]scale={w}:{h}:flags=area[s]"]
        for i, box in enumerate(crops):
            cx, cy, cw, ch = _crop_px(meta, box)
            parts.append(f"[c{i}in]crop={cw}:{ch}:{cx}:{cy},scale={CROP_W}:{CROP_H}[c{i}]")
        stack = "".join(f"[c{i}]" for i in range(n))
        parts.append(f"{stack}hstack=inputs={n}[cc]" if n > 1 else "[c0]null[cc]")
        parts.append(f"[cc]pad={w}:{CROP_H}[p];[s][p]vstack")
        filt = ["-filter_complex", ";".join(parts)]
    else:
        filt = ["-vf", f"{sel},scale={w}:{h}:flags=area"]
    cmd = [exe, "-v", "error", "-threads", "0", "-i", meta.path, "-an", *filt,
           "-vsync", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    out_h = h + extra_h
    frame_bytes = w * out_h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes * 4)
    k = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            img = np.frombuffer(buf, np.uint8).reshape(out_h, w, 3)
            t = (k * step) / meta.fps
            k += 1
            if crops:
                strip = img[h:]
                yield t, img[:h], [strip[:, i * CROP_W:(i + 1) * CROP_W] for i in range(len(crops))]
            else:
                yield t, img
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def _iter_opencv(meta: VideoMeta, step: int, size: tuple[int, int], crops: list | None) -> Iterator[tuple]:
    cap = cv2.VideoCapture(meta.path)
    idx = 0
    while True:
        if idx % step == 0:
            ok, frame = cap.read()
            if not ok:
                break
            small = frame
            if (frame.shape[1], frame.shape[0]) != size:
                small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            if crops:
                cut = []
                for box in crops:
                    cx, cy, cw, ch = _crop_px(meta, box)
                    cut.append(cv2.resize(frame[cy:cy + ch, cx:cx + cw], (CROP_W, CROP_H), interpolation=cv2.INTER_AREA))
                yield idx / meta.fps, small, cut
            else:
                yield idx / meta.fps, small
        elif not cap.grab():
            break
        idx += 1
    cap.release()
