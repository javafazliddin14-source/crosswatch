"""Learn the traffic-flow field and road mask of the camera view from the sample videos.

    python -m tools.build_scene --videos C:/data/samples

For every moving vehicle sample we add its heading to a per-cell direction
histogram (cells of a GRID_W x GRID_H grid) along a short swath around the
ground point. The histogram is what `wrong_way` compares a heading against;
cells with enough moving vehicles form the road mask used by `jaywalking`.
Output: weights/scene_flow.npz (committed, so evaluation needs no samples).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.detect_track import to_reference, track_video
from src.scene import GRID_H, GRID_W, N_DIR, SCENE_FLOW_PATH, direction_bin
from src.tracks import build_tracks
from src.video import probe

MIN_SPEED_REL = 0.6    # heights / s: clearly moving, not jitter of a parked car


def accumulate(tracks, hist: np.ndarray) -> None:
    for tr in tracks.values():
        if not tr.is_vehicle:
            continue
        for i in range(len(tr.t)):
            if tr.speed_rel[i] < MIN_SPEED_REL:
                continue
            b = direction_bin(tr.vx[i], tr.vy[i])
            # splat over the central half of the footprint so thin lanes are covered
            r = int(tr.y[i] * GRID_H)
            c0, c1 = int((tr.x[i] - tr.w[i] / 4) * GRID_W), int((tr.x[i] + tr.w[i] / 4) * GRID_W)
            for c in range(max(0, c0), min(GRID_W, c1 + 1)):
                if 0 <= r < GRID_H:
                    hist[r, c, b] += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    args = ap.parse_args()
    hist = np.zeros((GRID_H, GRID_W, N_DIR), np.float64)
    for p in sorted(Path(args.videos).glob("*.[mM][pP]4")):
        print("tracking", p.name)
        rows, _, H = track_video(probe(str(p)))
        tracks = build_tracks(to_reference(rows, H))
        accumulate(tracks, hist)

    # light spatial smoothing so sparse cells borrow from neighbours
    for b in range(N_DIR):
        hist[..., b] = cv2.GaussianBlur(hist[..., b], (5, 5), 1.0)
    count = hist.sum(-1)
    norm = hist / np.maximum(count[..., None], 1e-9)
    road = count >= 3.0
    road = cv2.morphologyEx(road.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    SCENE_FLOW_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SCENE_FLOW_PATH, hist=norm.astype(np.float32), count=count.astype(np.float32), road=road)
    print(f"wrote {SCENE_FLOW_PATH}: road cells {road.sum()} / {road.size}")


if __name__ == "__main__":
    main()
