"""Part A end to end: video -> registered tracks + signal timeline -> rule events."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.detect_track import to_reference, track_video
from src.rules import EventContext, run_rules
from src.signals import state_timeline
from src.tracks import Track, build_tracks
from src.video import VideoMeta, probe


@dataclass
class Analysis:
    meta: VideoMeta
    rows: np.ndarray            # raw tracker rows in the video's own coordinates (see detect_track.COLS)
    H: np.ndarray               # normalised homography video -> reference view
    tracks: dict[int, Track]    # trajectories in the reference view
    times: np.ndarray           # sampled frame times
    signal_scores: np.ndarray   # (n, 2) red / green lamp scores
    signal: np.ndarray          # (n,) smoothed state
    events: list[list]


def analyze(video_path: str, progress=None, use_cache: bool = True) -> Analysis:
    meta = probe(video_path)
    rows, sig, H = track_video(meta, progress=progress, use_cache=use_cache)
    tracks = build_tracks(to_reference(rows, H))
    times = sig[:, 0].astype(np.float64)
    state = state_timeline(sig[:, 1:3])
    ctx = EventContext(tracks=tracks, times=times, signal=state, duration=meta.duration)
    events = run_rules(ctx)
    return Analysis(meta, rows, H, tracks, times, sig[:, 1:3], state, events)
