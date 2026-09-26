"""Exploratory data analysis of the sample videos -> website/static/data/eda.json + images.

    python -m tools.eda --videos C:/data/samples

Uses the cached tracks (run the pipeline or tools/build_scene.py first).
Images: median background plate, vehicle / pedestrian ground-point heatmaps,
trajectories coloured by heading, the learned flow field and the road mask.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src import config as C
from src.detect_track import to_reference, track_video
from src.rules import crossing_index
from src.scene import GRID_H, GRID_W, N_DIR, SCENE
from src.signals import GREEN, RED, state_timeline
from src.tracks import build_tracks
from src.video import iter_frames, probe

FINDINGS = [
    "<b>Decoding is the bottleneck, not the network.</b> Each clip is 4K H.264 at ~150 Mb/s; OpenCV reads it at ~1x real time. "
    "Letting ffmpeg drop to 5 Hz and 1280 px before Python brings the whole Part A to ~0.4x real time (T4 budget is 3x for A + B together).",
    "<b>The signal state is visible.</b> A vehicle head on the median pole and a pedestrian head on the left pole face the camera and switch together. "
    "Cutting them from the full-resolution frame (they are ~15 px at 4K) gives a clean red/green timeline: ~42 s red, ~33 s green.",
    "<b>The heads we read govern the approach with the stop line.</b> ~95 % of stop-line crossings happen on our green; the rest cluster "
    "within 1-5 s of the change every cycle (the approach's own lamp lags), so red-light running only counts after a 4 s grace period.",
    "<b>People don't walk on the stripes.</b> The pedestrian heatmap peaks in a band 1-2 m beside the north zebra. Using the painted polygon "
    "flagged hundreds of false jaywalkers; we widened the crossing zone for jaywalking and keep the strict stripes for failure-to-yield.",
    "<b>Raised islands sit inside the asphalt.</b> The learned road mask bleeds over the median, the refuge and three brick islands where "
    "people legitimately wait, so they are carved out by hand.",
    "<b>The far kerb lane is a car park.</b> Cars load and wait there for minutes (one for a whole clip). Those are parked, not 'stopped vehicles'.",
    "<b>The south-east exit jams.</b> Every few cycles 12-15 vehicles stand still there with no signal in view: that is our congestion signal. "
    "The approach queue on red is normal and is only congestion if it doesn't clear on green.",
    "<b>Lighting varies: two clips are midday, one early evening, one dusk.</b> We keep a low detector threshold (0.2) and let ByteTrack's low-score association carry tracks through dark frames.",
    "<b>'Same camera and angle' is only approximately true.</b> C3902 is framed ~3 % differently (up to 60 px at 1080p) and C3905 ~2 %. "
    "Hand-measured geometry silently broke there (the lamp boxes missed the lamps, so C3902 had no signal phase at all). Every video is now "
    "registered to the reference view by a homography before any rule runs.",
    "<b>The camera sways slightly</b> (~12 px at 4K in wind), so every region of interest has a margin.",
    "<b>Strong perspective:</b> a car near the bottom is ~4x taller than one at the stop line. All speed thresholds are in <i>own box heights per second</i>.",
]

OUT_DATA = C.ROOT / "website" / "static" / "data"
OUT_IMG = C.ROOT / "website" / "static" / "media" / "eda"
W, H = 1280, 720


def background_and_light(meta, n: int = 40) -> tuple[np.ndarray, list[list[float]]]:
    """Median of n frames spread over the video (moving objects vanish) + brightness per frame."""
    fps = max(0.05, n / max(meta.duration, 1))
    frames, light = [], []
    for t, f in iter_frames(meta, fps, W):
        frames.append(f)
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        light.append([round(t, 1), round(float(hsv[..., 2].mean()), 1)])
    return np.median(np.stack(frames), axis=0).astype(np.uint8), light


def heat_overlay(bg: np.ndarray, pts: np.ndarray, color_map=cv2.COLORMAP_INFERNO) -> np.ndarray:
    heat = np.zeros((H // 4, W // 4), np.float32)
    if len(pts):
        xs = np.clip((pts[:, 0] * W / 4).astype(int), 0, W // 4 - 1)
        ys = np.clip((pts[:, 1] * H / 4).astype(int), 0, H // 4 - 1)
        np.add.at(heat, (ys, xs), 1)
    heat = cv2.GaussianBlur(heat, (0, 0), 2.5)
    heat = np.log1p(heat)
    heat = (255 * heat / max(heat.max(), 1e-6)).astype(np.uint8)
    heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)
    col = cv2.applyColorMap(heat, color_map)
    alpha = (heat.astype(np.float32) / 255)[..., None] ** 0.6
    base = (bg * 0.55).astype(np.float32)
    return (base * (1 - alpha) + col * alpha).astype(np.uint8)


def heading_color(vx: float, vy: float) -> tuple[int, int, int]:
    hue = int((np.degrees(np.arctan2(vy, vx)) % 360) / 2)
    c = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def trajectories_image(bg: np.ndarray, tracks_list) -> np.ndarray:
    img = (bg * 0.5).astype(np.uint8)
    for tracks in tracks_list:
        for tr in tracks.values():
            if not tr.is_vehicle or len(tr.t) < 10:
                continue
            dx, dy = tr.x[-1] - tr.x[0], tr.y[-1] - tr.y[0]
            if np.hypot(dx, dy) < 0.05:
                continue
            pts = np.stack([tr.x * W, tr.y * H], 1).astype(np.int32)
            cv2.polylines(img, [pts], False, heading_color(dx, dy), 1, cv2.LINE_AA)
    return img


def flow_image(bg: np.ndarray) -> np.ndarray:
    img = (bg * 0.5).astype(np.uint8)
    if SCENE.flow_hist is None:
        return img
    over = img.copy()
    for r in range(GRID_H):
        for c in range(GRID_W):
            if SCENE.road_mask[r, c]:
                cv2.rectangle(over, (c * W // GRID_W, r * H // GRID_H), ((c + 1) * W // GRID_W, (r + 1) * H // GRID_H),
                              (90, 90, 90), -1)
    cv2.addWeighted(over, 0.5, img, 0.5, 0, img)
    for r in range(1, GRID_H, 3):
        for c in range(1, GRID_W, 3):
            if SCENE.flow_count[r, c] < 20:
                continue
            h = SCENE.flow_hist[r, c]
            cx, cy = (c + 0.5) * W / GRID_W, (r + 0.5) * H / GRID_H
            for b in np.argsort(h)[::-1][:2]:
                if h[b] < 0.2:
                    continue
                ang = b * 2 * np.pi / N_DIR - np.pi
                L = 18 * h[b]
                p2 = (int(cx + L * np.cos(ang)), int(cy + L * np.sin(ang)))
                cv2.arrowedLine(img, (int(cx), int(cy)), p2, heading_color(np.cos(ang), np.sin(ang)), 2,
                                cv2.LINE_AA, tipLength=0.4)
    return img


def signal_cycles(times: np.ndarray, state: np.ndarray) -> dict:
    phases = []
    start = 0
    for i in range(1, len(state) + 1):
        if i == len(state) or state[i] != state[start]:
            phases.append((int(state[start]), float(times[start]), float(times[i - 1])))
            start = i
    inner = phases[1:-1]   # first/last phases are cut by the clip boundaries
    reds = [e - s for st, s, e in inner if st == RED]
    greens = [e - s for st, s, e in inner if st == GREEN]
    return {"phases": [[st, round(s, 1), round(e, 1)] for st, s, e in phases],
            "mean_red_s": round(float(np.mean(reds)), 1) if reds else None,
            "mean_green_s": round(float(np.mean(greens)), 1) if greens else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    args = ap.parse_args()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_IMG.mkdir(parents=True, exist_ok=True)

    videos, all_tracks, veh_pts, ped_pts, bg = [], [], [], [], None
    for p in sorted(Path(args.videos).glob("*.[mM][pP]4")):
        meta = probe(str(p))
        rows, sig, H = track_video(meta)
        tracks = build_tracks(to_reference(rows, H))     # heatmaps are drawn in the reference view
        all_tracks.append(tracks)
        b, light = background_and_light(meta)
        bg = b if bg is None else bg
        cv2.imwrite(str(OUT_IMG / f"bg_{p.stem}.jpg"), b, [cv2.IMWRITE_JPEG_QUALITY, 85])
        state = state_timeline(sig[:, 1:3])
        n_sec = int(np.ceil(meta.duration))
        per_sec = {}
        for name, cls in (("car", C.CAR), ("bus", C.BUS), ("truck", C.TRUCK), ("person", C.PERSON),
                          ("motorcycle", C.MOTORCYCLE)):
            sel = rows[rows[:, 2] == cls]
            cnt = np.bincount(sel[:, 0].astype(int).clip(0, n_sec - 1), minlength=n_sec) / C.SAMPLE_FPS
            per_sec[name] = np.round(cnt, 1).tolist()
        speeds = [float(np.median(tr.speed_rel)) for tr in tracks.values() if tr.is_vehicle and len(tr.t) > 10]
        on_green = on_red = 0
        for tr in tracks.values():
            k = crossing_index(tr) if tr.is_vehicle else None
            if k is not None:
                st = state[min(len(state) - 1, int(np.searchsorted(sig[:, 0], tr.t[k])))]
                on_red += st == RED
                on_green += st == GREEN
        for tr in tracks.values():
            moving = tr.speed_rel > 0.6
            pts = np.stack([tr.x, tr.y], 1)
            if tr.is_vehicle:
                veh_pts.append(pts[moving])
            elif tr.is_person:
                ped_pts.append(pts)
        videos.append({
            "name": p.name, "width": meta.width, "height": meta.height, "fps": round(meta.fps, 3),
            "n_frames": meta.n_frames, "duration": round(meta.duration, 1),
            "size_gb": round(p.stat().st_size / 1e9, 2), "light": light,
            "counts_per_sec": per_sec,
            "unique_tracks": {C.CLASS_NAMES[c]: sum(1 for tr in tracks.values() if tr.cls == c)
                              for c in (C.CAR, C.BUS, C.TRUCK, C.PERSON, C.MOTORCYCLE, C.BICYCLE)},
            "median_vehicle_speed_rel": round(float(np.median(speeds)), 2) if speeds else None,
            "signal": signal_cycles(sig[:, 0], state),
            "stop_line_crossings": {"green": int(on_green), "red": int(on_red)},
        })
        print("eda", p.name)

    veh = np.concatenate(veh_pts) if veh_pts else np.zeros((0, 2))
    ped = np.concatenate(ped_pts) if ped_pts else np.zeros((0, 2))
    cv2.imwrite(str(OUT_IMG / "heat_vehicles.jpg"), heat_overlay(bg, veh), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(OUT_IMG / "heat_people.jpg"), heat_overlay(bg, ped, cv2.COLORMAP_OCEAN), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(OUT_IMG / "trajectories.jpg"), trajectories_image(bg, all_tracks), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(OUT_IMG / "flow.jpg"), flow_image(bg), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (OUT_DATA / "eda.json").write_text(json.dumps({"videos": videos, "findings": FINDINGS}))
    print("wrote", OUT_DATA / "eda.json")


if __name__ == "__main__":
    main()
