"""Local web UI for ExMR.

Reconstruct a video with ExMR, preview the motion on a VRM avatar in the
browser and export the result as a VRM Animation (.vrma) next to the clip's
audio track.

The browser retargets the 3D joints onto the avatar (the server never needs the
avatar's rest pose) and bakes the bone quaternions; the server stores them as a
``.vrma`` and writes the audio as a ``.wav`` into the same folder, so an export
can be dropped straight into a VRM viewer or editor.

Adapted from the rtmlib VRM motion-capture app (web/src/mocap.js and
app/vrma.py) for ExMR's 3D output.

Run:
    python -m exmr.ui.server            # then open http://127.0.0.1:8010
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import vrma

ROOT = Path(__file__).resolve().parents[2]
UI_DIR = Path(__file__).resolve().parent
DIST = UI_DIR / "dist"
ASSETS = UI_DIR / "assets"
VIDEOS = UI_DIR / "videos"
OUTPUT_DIR = ROOT / "outputs" / "ui"
EXPORT_DIR = ROOT / "outputs" / "ui_exports"

# SMPL-X joints 0-21 are the body; the VRM humanoid has no finger bones.
BODY_JOINTS = 22
MAX_FRAMES = 6000
JOB_TIMEOUT = 2 * 60 * 60
UPLOAD_LIMIT = 512 * 1024 * 1024
VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv", ".m4v", ".avi"}
TOKEN = re.compile(r"[^A-Za-z0-9._ -]+")

# Files ExMR needs for the default mode (see exmr/assets.py).
REQUIRED = [
    "exmr/exmr.safetensors",
    "body_models/smplx/SMPLX_NEUTRAL.npz",
    "yolo/yolo11m.pt",
    "sapiens2/sapiens2_1b_pose.safetensors",
    "sam-3d-body-dinov3/model.ckpt",
    "sam-3d-body-dinov3/model_config.yaml",
    "sam-3d-body-dinov3/assets/mhr_model.pt",
    "hamer/hamer_ckpts/checkpoints/hamer.ckpt",
    "hamer/hamer_ckpts/model_config.yaml",
    "hamer/data/mano_mean_params.npz",
    "hamer/data/mano/MANO_RIGHT.pkl",
]


def clean_name(name: str, default: str = "clip") -> str:
    stem = TOKEN.sub("_", Path(name).name).strip(" ._")
    return stem or default


def checkpoint_dir() -> Path:
    env = os.environ.get("EXMR_CHECKPOINT_DIR")
    if not env:
        return ROOT / "checkpoints"
    path = Path(env)
    return path if path.is_absolute() else ROOT / path


def missing_files() -> list[str]:
    root = checkpoint_dir()
    return [name for name in REQUIRED if not (root / name).exists()]


def availability() -> dict:
    missing = missing_files()
    if missing:
        return {"ok": False, "missing": missing,
                "reason": f"{len(missing)} model file(s) missing - rerun "
                          f"tools/download_models.py (see docs/INSTALL.md)"}
    return {"ok": True, "missing": [], "reason": "ready"}


# --------------------------------------------------------------------- motion
def frames_from_npz(path: Path) -> dict:
    """ExMR results.npz -> per-frame 3D joints for the browser retargeter."""
    with np.load(path) as data:
        if "joints_global" not in data.files:
            raise RuntimeError(f"{path.name} has no joints_global array")
        joints = np.asarray(data["joints_global"], dtype=np.float32)
        fps = float(data["fps"]) if "fps" in data.files else 30.0

    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise RuntimeError(f"{path.name}: unexpected joint shape {joints.shape}")

    joints = joints[:, :BODY_JOINTS]
    if not 1.0 <= fps <= 240.0:
        fps = 30.0

    finite = np.isfinite(joints).all(axis=(1, 2))
    if not finite.any():
        raise RuntimeError(f"{path.name}: every frame has non-finite joints")
    if not finite.all():
        good = np.flatnonzero(finite)
        for index in np.flatnonzero(~finite):
            joints[index] = joints[good[np.abs(good - index).argmin()]]

    total = min(len(joints), MAX_FRAMES)
    root = joints[:, 0] - joints[0, 0]

    return {
        "fps": fps,
        "joints": BODY_JOINTS,
        "truncated": len(joints) > MAX_FRAMES,
        "frames": [
            {"t": round(i / fps, 4),
             "joints3d": np.round(joints[i], 4).tolist(),
             "root": np.round(root[i], 4).tolist()}
            for i in range(total)
        ],
    }


# ------------------------------------------------------------------------ job
class Job:
    """One ExMR run, polled by the browser for its phase and result."""

    def __init__(self, job_id: str, video: Path, static_cam: bool, snap: bool):
        self.id = job_id
        self.video = video
        self.static_cam = static_cam
        self.snap = snap
        self.folder = OUTPUT_DIR / job_id
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stage = -1
        self._status = {"state": "queued", "phase": "starting", "message": "",
                        "error": "", "started": time.time(), "elapsed": 0.0,
                        "frames": 0}
        self._motion: Optional[dict] = None

    def status(self) -> dict:
        with self._lock:
            data = dict(self._status)
        if data["state"] == "running":
            data["elapsed"] = round(time.time() - data["started"], 1)
        return data

    def _set(self, **fields):
        with self._lock:
            self._status.update(fields)

    def start(self):
        self._write_meta()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -- persistence: a finished job is still usable after a server restart
    def _write_meta(self):
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / "job.json").write_text(json.dumps({
                "id": self.id, "video": self.video.name,
                "static_cam": self.static_cam, "snap": self.snap,
            }), encoding="utf-8")
        except OSError:
            pass

    @classmethod
    def load(cls, folder: Path) -> Optional["Job"]:
        """Rebuild a finished job from its output folder."""
        try:
            meta = json.loads((folder / "job.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return None
        npz = folder / "results.npz"
        if not npz.exists():
            return None
        job = cls(str(meta.get("id") or folder.name), VIDEOS / str(meta.get("video", "")),
                  bool(meta.get("static_cam", True)), bool(meta.get("snap", False)))
        try:
            job._motion = frames_from_npz(npz)
        except Exception:  # noqa: BLE001 - a broken cache is simply ignored
            return None
        job._status.update(state="done", phase="done",
                           frames=len(job._motion["frames"]),
                           message=f"{len(job._motion['frames'])} frames")
        return job

    def cancel(self):
        self._cancel.set()
        proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except OSError:
                pass
        self._set(phase="cancelling", message="stopping ExMR")

    _PHASES = (
        (("extracting frames",), ("reading", "extracting frames")),
        (("computing tracks",), ("tracking", "finding the person")),
        (("computing keypoints",), ("keypoints", "estimating 2D keypoints")),
        (("computing body_features",), ("body", "sampling body features")),
        (("computing hand_features",), ("hands", "sampling hand features")),
        (("computing camera_rotations",), ("camera", "estimating camera motion")),
        (("running exmr",), ("generating", "refining hands and feet")),
    )

    def _phase_of(self, line: str):
        low = line.lower()
        for index, (needles, (phase, message)) in enumerate(self._PHASES):
            if any(needle in low for needle in needles):
                if index > self._stage:
                    self._stage = index
                    self._set(phase=phase, message=message)
                return

    def _run(self):
        self._set(state="running", phase="starting", message="loading models")
        try:
            npz = self._infer()
            self._set(phase="reading", message="reading the motion")
            motion = frames_from_npz(npz)
            with self._lock:
                self._motion = motion
            self._set(state="done", phase="done", frames=len(motion["frames"]),
                      message=f"{len(motion['frames'])} frames")
        except Exception as exc:  # noqa: BLE001
            if self._cancel.is_set():
                self._set(state="cancelled", phase="cancelled")
            else:
                self._set(state="error", phase="failed", error=str(exc))

    def _infer(self) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "tools" / "demo.py"),
               "--video", str(self.video), "--output_dir", str(self.folder),
               "--no_render", "--mode", "balanced", "--keep_frames"]
        if self.static_cam:
            cmd.append("-s")
        if self.snap:
            cmd.append("--snap_to_ground")

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        if os.name == "nt":
            env.setdefault("PYOPENGL_PLATFORM", "win32")

        log_tail: list[str] = []
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        self._proc = proc
        watchdog = threading.Timer(JOB_TIMEOUT, proc.kill)
        watchdog.start()
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                line = line.rstrip()
                if not line:
                    continue
                log_tail.append(line)
                del log_tail[:-40]
                self._phase_of(line)
        finally:
            watchdog.cancel()
            proc.wait()
            self._proc = None

        # The pipeline writes results.npz before cleaning up extracted frames.
        # A file-safety hook can intercept that rmtree (630+ files) and kill
        # the process, leaving a non-zero exit code even though the results
        # are complete. Check for the output first; only treat it as a real
        # failure if the npz is missing.
        found = self.folder / "results.npz"
        if proc.returncode != 0 and not found.exists():
            if self._cancel.is_set():
                raise RuntimeError("cancelled")
            raise RuntimeError("ExMR failed:\n" + "\n".join(log_tail[-8:]))
        if not found.exists():
            raise RuntimeError("ExMR produced no results")
        return found


# --------------------------------------------------------------------- export
jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        for folder in (VIDEOS, EXPORT_DIR, ASSETS):
            folder.mkdir(parents=True, exist_ok=True)
        yield

    app = FastAPI(title="ExMR", lifespan=lifespan)

    @app.get("/api/status")
    async def status():
        return {"available": availability(), "busy": any(
            j.status()["state"] in ("queued", "running") for j in jobs.values())}

    # -------------------------------------------------------------- videos
    @app.get("/api/videos")
    async def list_videos():
        items = []
        for path in sorted(VIDEOS.glob("*")):
            if path.suffix.lower() in VIDEO_EXT:
                items.append({"name": path.name,
                              "url": f"/videos/{path.name}",
                              "bytes": path.stat().st_size})
        return {"videos": items}

    @app.post("/api/upload")
    async def upload(name: str, request: Request):
        clean = clean_name(name)
        if Path(clean).suffix.lower() not in VIDEO_EXT:
            return JSONResponse({"error": "unsupported video"}, status_code=400)
        body = await request.body()
        if not body:
            return JSONResponse({"error": "empty file"}, status_code=400)
        if len(body) > UPLOAD_LIMIT:
            return JSONResponse({"error": "file too large"}, status_code=413)
        VIDEOS.mkdir(parents=True, exist_ok=True)
        destination = VIDEOS / clean
        digest = hashlib.sha256(body).hexdigest()
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                destination = VIDEOS / f"{Path(clean).stem}_{digest[:6]}{Path(clean).suffix}"
        destination.write_bytes(body)
        return {"name": destination.name, "url": f"/videos/{destination.name}"}

    # ---------------------------------------------------------------- jobs
    @app.post("/api/jobs")
    async def start_job(payload: dict):
        name = str(payload.get("video", "")).strip()
        path = (VIDEOS / Path(name).name).resolve()
        if VIDEOS.resolve() not in path.parents or not path.exists():
            return JSONResponse({"error": "video not found"}, status_code=400)
        check = availability()
        if not check["ok"]:
            return JSONResponse({"error": check["reason"]}, status_code=409)
        with jobs_lock:
            if any(j.status()["state"] in ("queued", "running")
                   for j in jobs.values()):
                return JSONResponse({"error": "a job is already running"},
                                    status_code=409)
            job = Job(uuid.uuid4().hex[:12], path,
                      bool(payload.get("static_cam", True)),
                      bool(payload.get("snap_to_ground", False)))
            jobs[job.id] = job
        job.start()
        return job.status() | {"id": job.id}

    def find_job(job_id: str) -> Optional[Job]:
        job = jobs.get(job_id)
        if job is None:
            folder = OUTPUT_DIR / Path(job_id).name
            job = Job.load(folder)
            if job is not None:
                with jobs_lock:
                    jobs.setdefault(job.id, job)
        return job

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str):
        job = find_job(job_id)
        if job is None:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        return job.status()

    @app.post("/api/jobs/{job_id}/cancel")
    async def job_cancel(job_id: str):
        job = find_job(job_id)
        if job is None:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        job.cancel()
        return job.status()

    @app.get("/api/jobs/{job_id}/motion")
    async def job_motion(job_id: str):
        job = find_job(job_id)
        if job is None or job._motion is None:
            return JSONResponse({"error": "no motion for this job"},
                                status_code=404)
        return job._motion

    @app.get("/api/jobs/{job_id}/bvh")
    async def job_bvh(job_id: str):
        """Serve the BVH file produced by ExMR next to results.npz."""
        job = find_job(job_id)
        if job is None:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        bvh_files = list(job.folder.glob("*.bvh"))
        if not bvh_files:
            return JSONResponse({"error": "no BVH for this job"},
                                status_code=404)
        return FileResponse(bvh_files[0], media_type="text/plain",
                            filename=bvh_files[0].name)

    # -------------------------------------------------------------- export
    @app.post("/api/jobs/{job_id}/export")
    async def export(job_id: str, payload: dict):
        job = find_job(job_id)
        if job is None:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        frames = payload.get("frames") or []
        if len(frames) < 2:
            return JSONResponse({"error": "need at least two frames"},
                                status_code=400)
        stem = clean_name(job.video.stem)
        folder = EXPORT_DIR / f"{stem}_{job_id}"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{stem}.vrma"
        try:
            info = vrma.write_vrma(target, {"fps": payload.get("fps", 30),
                                            "frames": frames})
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": str(exc)}, status_code=400)
        return {"animation": f"/exports/{folder.name}/{target.name}",
                "frames": len(frames), **info}

    # ------------------------------------------------------------- assets
    @app.get("/api/vrms")
    async def list_vrms():
        items = [{"name": p.name,
                  "url": f"/vrm/{p.name}?v={int(p.stat().st_mtime)}",
                  "bytes": p.stat().st_size}
                 for p in sorted(ASSETS.glob("*.vrm"))]
        return {"vrms": items}

    @app.post("/api/vrms")
    async def upload_vrm(name: str, request: Request):
        clean = clean_name(name)
        if not clean.lower().endswith(".vrm"):
            clean += ".vrm"
        body = await request.body()
        if not body:
            return JSONResponse({"error": "empty file"}, status_code=400)
        (ASSETS / clean).write_bytes(body)
        return {"name": clean, "url": f"/vrm/{clean}",
                "bytes": len(body)}

    @app.get("/api/idle")
    async def idle():
        path = ASSETS / "idle_loop.vrma"
        if not path.exists():
            return JSONResponse({"error": "no idle animation"}, status_code=404)
        return FileResponse(path)

    @app.get("/videos/{name}")
    async def video_file(name: str):
        path = VIDEOS / Path(name).name
        if not path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(path)

    @app.get("/vrm/{name}")
    async def vrm_file(name: str):
        path = ASSETS / Path(name).name
        if not path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(path, media_type="model/gltf-binary",
                            headers={"Cache-Control": "no-store"})

    @app.get("/exports/{folder}/{name}")
    async def export_file(folder: str, name: str):
        path = EXPORT_DIR / Path(folder).name / Path(name).name
        if not path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(path, filename=path.name)

    if DIST.exists():
        app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")

    return app


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    if not (DIST / "index.html").exists():
        sys.exit("Frontend not built: cd exmr/ui/web && npm install && npm run build")
    (UI_DIR / "videos").mkdir(parents=True, exist_ok=True)
    uvicorn.run(create_app(), host=args.host, port=args.port)


app = create_app()

if __name__ == "__main__":
    main()
