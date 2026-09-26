"""Team website + live demo backend.

    python -m website.server            # http://localhost:8000

Serves the static site from website/static and a small job API for the demo:

    POST /api/jobs            multipart upload (field "video", .mp4, <= MAX_UPLOAD_MB, <= MAX_DURATION_S)
    GET  /api/jobs/{id}       {"state", "stage", "progress", "result"?, "error"?}
    GET  /api/jobs/{id}/video annotated H.264 mp4 once the job is done

Jobs run one at a time in a worker thread (the model is not re-entrant and the
demo host is a CPU box); queued jobs report their position.
"""
from __future__ import annotations

import queue
import shutil
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from src.analyze import analyze_video  # noqa: E402
from src.video import probe  # noqa: E402

MAX_UPLOAD_MB = 200
MAX_DURATION_S = 180
JOB_DIR = ROOT / ".cache" / "jobs"
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Traffic event detection demo")
jobs: dict[str, dict] = {}
work: "queue.Queue[str]" = queue.Queue()


def _worker() -> None:
    while True:
        jid = work.get()
        job = jobs[jid]
        job.update(state="running", stage="starting", progress=0.0)

        def report(stage: str, p: float) -> None:
            job.update(stage=stage, progress=round(p, 3))

        try:
            job["result"] = analyze_video(job["path"], job["dir"], progress=report)
            job.update(state="done", stage="done", progress=1.0)
        except Exception as exc:  # surfaced to the page, never crashes the server
            traceback.print_exc()
            job.update(state="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            job["finished"] = time.time()
            _gc_jobs()


def _gc_jobs(keep_s: float = 3600) -> None:
    now = time.time()
    for jid in [j for j, v in jobs.items() if v.get("finished") and now - v["finished"] > keep_s]:
        shutil.rmtree(jobs.pop(jid)["dir"], ignore_errors=True)


@app.post("/api/jobs")
async def create_job(video: UploadFile = File(...)):
    if not (video.filename or "").lower().endswith(".mp4"):
        raise HTTPException(400, "please upload an .mp4 file")
    jid = uuid.uuid4().hex[:12]
    d = JOB_DIR / jid
    d.mkdir(parents=True, exist_ok=True)
    path = d / "input.mp4"
    size = 0
    with open(path, "wb") as f:
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                f.close()
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB")
            f.write(chunk)
    try:
        meta = probe(str(path))
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "could not decode this video")
    if meta.duration > MAX_DURATION_S:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, f"video is {meta.duration:.0f} s long; the demo accepts up to {MAX_DURATION_S} s")
    jobs[jid] = {"state": "queued", "stage": "queued", "progress": 0.0, "path": str(path), "dir": str(d),
                 "name": video.filename, "created": time.time()}
    work.put(jid)
    return {"id": jid}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    job = jobs.get(jid)
    if job is None:
        raise HTTPException(404, "unknown job")
    out = {k: job.get(k) for k in ("state", "stage", "progress", "error", "result", "name")}
    if job["state"] == "queued":
        out["queue_position"] = sum(1 for v in jobs.values() if v["state"] == "queued" and v["created"] <= job["created"])
    return out


@app.get("/api/jobs/{jid}/video")
def job_video(jid: str):
    job = jobs.get(jid)
    if job is None or job["state"] != "done":
        raise HTTPException(404, "not ready")
    return FileResponse(Path(job["dir"]) / "annotated.mp4", media_type="video/mp4")


@app.get("/api/health")
def health():
    return {"ok": True, "queued": work.qsize()}


@app.get("/predictions_samples.json")
def sample_predictions():
    return FileResponse(ROOT / "predictions_samples.json", media_type="application/json")


@app.on_event("startup")
def _start_worker() -> None:
    threading.Thread(target=_worker, daemon=True).start()


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

if __name__ == "__main__":
    import os

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))   # hosts (HF Spaces, Render) set PORT
