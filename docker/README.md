# VRMExMR container (camenduru/VRMExMR)

A single GPU image that runs the VRMExMR UI with **all models baked in**. The
container performs no downloads at runtime: ExMR weights, SAM 3D Body, Sapiens2,
RTMW, YOLO, HaMeR/MANO, SMPL-X and the pinned DINOv3 `torch.hub` code are all
inside the image.

## What goes into the image

| Path in image | Source in repo | Size |
|---|---|---|
| `/app/checkpoints` | `checkpoints/` | ~11.5 GB |
| `/root/.cache/torch/hub/facebookresearch_dinov3_main` | `docker/torch_hub.tar.gz` | ~13 MB |
| `/app/exmr/ui/dist` | built from `exmr/ui/web` (Node stage) | small |
| `/app/exmr/ui/videos` | `exmr/ui/videos/` sample clips | ~9 MB |

`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` are set,
and `TORCH_HOME` points at the bundled DINOv3 cache.

## Prerequisites

- Docker Desktop (or Docker Engine) with the NVIDIA container toolkit and a
  driver that supports CUDA 12.8 (>= 570).
- The repository's `checkpoints/` directory must be fully populated first
  (`tools/download_models.py`, see `docs/INSTALL.md`). The build fails early if
  the weights are absent because they are copied from the build context.
- About 30-40 GB of free disk for the image and build cache.

## Build

```bash
docker build -t camenduru/vrmexmr:latest .
```

## Run

```bash
docker run --rm --gpus all --shm-size=8g -p 8010:8010 camenduru/vrmexmr:latest
```

Open <http://127.0.0.1:8010>.

`--shm-size=8g` is required: PyTorch's DataLoader workers pass image batches
through `/dev/shm`, and Docker's default 64 MB fills up during SAM 3D Body
feature extraction. The job then appears to hang at "sampling body features"
(workers die, the parent waits on them forever) instead of failing.

The image also sets `EXMR_PIN_MEMORY=0`. Docker on WSL2 caps pinned host memory
at ~0.5-1 GB independent of `--shm-size`, `--ipc=host` or `ulimits`
([microsoft/WSL#14078](https://github.com/microsoft/WSL/issues/14078)), while
each queued SAM 3D Body batch pins ~57 MB. With pinning enabled the job dies
with `CUDA error: out of memory` in the pin-memory thread after ~250 frames.
Disabling it only skips the CPU-to-GPU staging copy; measured throughput was
unchanged. Native (non-container) runs keep pinning by default.

Persist jobs and exported `.vrma` files:

```bash
docker run --rm --gpus all --shm-size=8g -p 8010:8010 \
  -v "$PWD/outputs:/app/outputs" \
  camenduru/vrmexmr:latest
```

Or with Compose:

```bash
docker compose up --build
```

## Notes

- The UI always runs ExMR with `--no_render` (the browser preview and the
  `.vrma` export are the output), so `nvdiffrast` is intentionally **not**
  installed and no rendered MP4 is produced by the container.
- Uploaded clips are written to `/app/exmr/ui/videos` inside the container.
  Mount a volume there if you want them to survive `docker run --rm`.
- CUDA needs `--gpus all`; without it the server still starts but jobs fail when
  they try to initialize the models on the GPU.
