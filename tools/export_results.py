"""Run the full analysis on every sample video and publish it to the website.

    python -m tools.export_results --videos C:/data/samples

Writes website/static/media/results/<stem>.mp4 (annotated, H.264) and
website/static/data/results/<stem>.json, plus an index (results.json).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src import config as C
from src.analyze import analyze_video

OUT_DATA = C.ROOT / "website" / "static" / "data" / "results"
OUT_MEDIA = C.ROOT / "website" / "static" / "media" / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--pred", default=str(C.ROOT / "predictions_samples.json"),
                    help="reuse events + risk from this harness output (skipped if missing)")
    args = ap.parse_args()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_MEDIA.mkdir(parents=True, exist_ok=True)
    pred = json.loads(Path(args.pred).read_text())["videos"] if Path(args.pred).exists() else {}
    index = []
    for p in sorted(Path(args.videos).glob("*.[mM][pP]4")):
        work = C.CACHE_DIR / "export" / p.stem
        res = analyze_video(str(p), str(work), precomputed=pred.get(p.name),
                            progress=lambda s, x: print(f"\r{p.name}: {s} {x:.0%}   ", end=""))
        print()
        shutil.copy(work / "annotated.mp4", OUT_MEDIA / f"{p.stem}.mp4")
        (OUT_DATA / f"{p.stem}.json").write_text(json.dumps(res))
        index.append({"stem": p.stem, "name": p.name, "duration": res["meta"]["duration"],
                      "n_events": len(res["events"])})
    (OUT_DATA.parent / "results.json").write_text(json.dumps(index))
    print("exported", len(index), "videos")


if __name__ == "__main__":
    main()
