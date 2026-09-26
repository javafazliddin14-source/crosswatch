"""Export what the in-browser demo needs into website/static/demo/.

    python -m tools.export_web

* scene.json: the scene geometry from src/scene.py (crossings, stop line, lamp boxes,
  islands, zones, road mask), so the browser uses exactly the same map as the Python pipeline;
* yolo11n_960x544.onnx: YOLO11n (COCO) for onnxruntime-web, sized for 16:9 frames;
* scene_ref.jpg: the reference view the demo aligns uploaded videos to.
"""
from __future__ import annotations

import json
import shutil

import numpy as np

from src import config as C
from src import scene as S

OUT = C.ROOT / "website" / "static" / "demo"
WEB_MODEL = C.WEIGHTS_DIR / "yolo11n.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"


def _poly(p: np.ndarray) -> list:
    return np.round(p, 5).tolist()


def export_scene() -> None:
    sc = S.SCENE
    data = {
        "stripes": {k: _poly(v) for k, v in S.STRIPES.items()},
        "crosswalks": {k: _poly(v) for k, v in sc.crosswalks.items()},
        "stop_line": _poly(sc.stop_line),
        "signal_heads": [list(map(float, b)) for b in sc.signal_heads],
        "non_road": [_poly(p) for p in S.NON_ROAD],
        "curb_parking": [_poly(p) for p in S.CURB_PARKING],
        "zones": {k: {"poly": _poly(p), "min_count": n, "signalled": sig} for k, (p, n, sig) in sc.traffic_zones.items()},
        "grid": [S.GRID_W, S.GRID_H],
        "road_core": sc.road_core.astype(np.uint8).ravel().tolist(),
    }
    (OUT / "scene.json").write_text(json.dumps(data, separators=(",", ":")))


def export_model() -> None:
    if not WEB_MODEL.exists():
        import urllib.request

        urllib.request.urlretrieve(MODEL_URL, WEB_MODEL)
    from ultralytics import YOLO

    path = YOLO(str(WEB_MODEL)).export(format="onnx", imgsz=[544, 960], opset=17, simplify=True, dynamic=False)
    shutil.move(path, OUT / "yolo11n_960x544.onnx")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    export_scene()
    export_model()
    shutil.copy(C.WEIGHTS_DIR / "scene_ref.jpg", OUT / "scene_ref.jpg")
    for p in sorted(OUT.iterdir()):
        print(f"{p.stat().st_size:>10,}  {p.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
