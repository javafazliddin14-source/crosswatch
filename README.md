# CrossWatch - traffic events from a fixed CCTV intersection camera

WIUT Hackathon 2026, Computer Vision track. `solution.py` implements the official interface:
`detect_events(video_path)` (Part A) and the causal `RiskEstimator` (Part B).

## Install and run

```bash
pip install -r requirements.txt           # or: docker build -t crosswatch .
sh weights/download.sh                    # only if weights/yolo11m.pt is missing (run once, with internet)
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

* Weights: `weights/yolo11m.pt` (40 MB, Ultralytics YOLO11m, COCO) ships in the repo; `weights/download.sh`
  re-fetches it and checks the SHA-256. `weights/scene_flow.npz` (64 KB) is the scene model learned from the samples, and
  `weights/scene_ref.jpg` is the reference view (median background of sample C3896) every clip is registered to.
* No internet is needed at run time (`YOLO_OFFLINE=1` is set; nothing is downloaded).
* `ffmpeg` is **not** required system-wide: the `imageio-ffmpeg` wheel bundles a static binary.
* Measured cold runtime (no cache) with the official harness on an RTX 5050 laptop GPU: Part A 0.27-0.32x the video
  duration, Part B 0.60-0.75x, total 0.93-1.04x against the 3x budget. A T4 is in the same class. The harness's own
  full-resolution 4K OpenCV decode for Part B is the largest fixed cost.

`predictions_samples.json` is our output on the four sample videos, produced by exactly the commands above.

## Approach

```
.mp4 ──ffmpeg (5 Hz, 1280 px + full-res lamp crops)──┬─ YOLO11m ─ ByteTrack ─ trajectories ─┐
                                                     └─ lamp colour ─ red/green timeline ────┼─ rules ─ segments
                              scene model: crossings, stop line, islands, zones, flow field ─┘
```

| Stage | Learned or rule-based | Details |
|---|---|---|
| View registration | classical CV | framings differ by up to ~3 % between clips; SIFT + RANSAC homography from a median background to `weights/scene_ref.jpg`; tracks are mapped into the reference view, lamp boxes out of it (`src/registration.py`) |
| Detection | learned (pretrained, not fine-tuned) | YOLO11m COCO, 1280 px, fp16, conf 0.2; person, bicycle, car, motorcycle, bus, truck, animals |
| Tracking | algorithmic | ByteTrack (`src/bytetrack.yaml`), buffer tuned for 5 Hz |
| Signal state | rule-based | lit red/green pixels (HSV) in the two camera-facing heads, cut from the full-res frame; 1 s debounce (`src/signals.py`) |
| Scene | hand-measured + learned statistics | crossings, stop line, islands, curb-parking lane, traffic zones (`src/scene.py`); per-cell heading histogram and road mask from the samples (`tools/build_scene.py`) |
| Events | rule-based | `src/rules.py`, one function per class, thresholds in perspective-invariant units (box heights / s) |
| Part B | rule-based on learned detections | own causal tracker at 5 Hz; time-to-collision between footprints + hard braking -> logistic, calibrated on the samples (`tools/calibrate_risk.py`) |

Classes emitted: `jaywalking`, `failure_to_yield`, `red_light`, `stop_line`, `stopped_vehicle`, `wrong_way`,
`congestion`, `road_obstacle`. Not emitted (no reliable signal without labels; a wrong class costs a full macro-F1 slot):
`accident`, `near_miss`, `illegal_u_turn`, `illegal_turn`, `solid_line_crossing`, `fire_smoke`.

Full write-up with figures: the team website (`website/`), sections *Approach*, *EDA*, *Report*.

## Repository layout

```
solution.py            interface (thin wrapper over src/)
run_submission.py      organisers' harness, unchanged
evaluate.py            organisers' metric, unchanged
src/                   video.py (decode), detect_track.py, tracks.py, signals.py, scene.py, rules.py,
                       segments.py, pipeline.py (Part A), risk.py (Part B), render.py, analyze.py
tools/                 build_scene.py, calibrate_risk.py, eda.py, export_results.py
weights/               yolo11m.pt, scene_flow.npz, download.sh, SHA256SUMS
website/               server.py (FastAPI: site + live demo API), static/ (HTML/CSS/JS, data, media)
predictions_samples.json
```

Rebuild everything derived from the samples:

```bash
python -m tools.build_scene   --videos samples     # weights/scene_flow.npz
python -m tools.calibrate_risk --videos samples    # prints alarm counts per threshold
python run_submission.py --videos samples --out predictions_samples.json --team crosswatch
python -m tools.eda           --videos samples     # website/static/data/eda.json + figures
python -m tools.export_results --videos samples    # annotated videos + per-video JSON for the site
python -m website.server                           # http://localhost:8000
```

## Website and live demo

`python -m website.server` serves the site and the demo API on `http://localhost:8000` (`$PORT` if set).
The demo accepts `.mp4` uploads of up to 200 MB and 3 minutes, runs the exact submission pipeline (Part A + causal Part B
+ annotated render) in a background queue, and reports progress. It runs on CPU too (slower).

To publish it, any Docker host works. For a Hugging Face Space (Docker SDK, free CPU tier), push this repository with the
root `Dockerfile` and set the Space's start command to `python -m website.server` and `PORT=7860`. The static part
(`website/static/`) also works on its own (GitHub Pages / Netlify), minus the upload demo.

## Data and licences

| What | Licence | Use |
|---|---|---|
| COCO 2017 (through the pretrained YOLO11m weights) | CC BY 4.0 | detector pretraining (by Ultralytics) |
| Sample videos from the organisers | organisers' terms | scene model, calibration, EDA; not redistributed in this repo |

No other dataset was used, and no model was fine-tuned.

## Open-source code used

* [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0): YOLO11 inference and its ByteTrack implementation.
* ByteTrack algorithm: Zhang et al., 2022 (MIT).
* [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) (BSD-2) for the bundled ffmpeg binary.
* Chart.js (MIT) on the website.

## Determinism

Seeds are fixed (`src/config.py: SEED = 0`: Python, NumPy, torch); cuDNN runs in deterministic mode with benchmarking
off. Frame sampling is index-based (every 6th frame at 29.97 fps). Remaining non-determinism: fp16 GPU inference can
differ in the last bits across GPU models, which can flip a borderline detection; on the same machine two runs produce
identical `predictions.json`. Detections are cached in `.cache/` keyed by file name, size, mtime and pipeline version;
delete it to recompute from scratch.

## Team

<!-- team:start -->
| Member | Role | Did | Links |
|---|---|---|---|
| Sherzodov Abduraxmon | Team lead · website, EDA & report | Coordinated the team and the submission. EDA of the sample videos, the annotated-video renderer, the website with the live demo, and the technical report. |  |
| Fazliddinov Javokhirshoh | Computer vision lead | Detection and tracking (YOLO11 + ByteTrack), the fast ffmpeg decoding path, registering each video to the reference view, the runtime budget, and the Part B accident-risk model and its calibration. | [GitHub](https://github.com/javafazliddin14-source) |
| Baqbergenov Dauletbay | Scene modelling & rules | The intersection map (crossings, stop line, islands, zones), reading the traffic-light state, the per-class event rules, and checking every detected event against the footage. |  |
<!-- team:end -->
