# syntax=docker/dockerfile:1.7
#
# VRMExMR - self-contained image for camenduru/VRMExMR
#
# Every ExMR model file is baked into the image from ./checkpoints, together with
# the pinned DINOv3 torch.hub code (docker/torch_hub.tar.gz), so the container
# never downloads weights or code at runtime.
#
# Build (from the repository root, with checkpoints/ fully populated):
#   docker build -t camenduru/vrmexmr:latest .
#
# Run (--shm-size is required for PyTorch's DataLoader workers):
#   docker run --rm --gpus all --shm-size=8g -p 8010:8010 camenduru/vrmexmr:latest
#
# Then open http://127.0.0.1:8010

###############################################################################
# 1. front end (Vite + three.js) -> /src/exmr/ui/dist
###############################################################################
FROM node:22-alpine AS web

WORKDIR /src/exmr/ui/web
COPY exmr/ui/web/package.json exmr/ui/web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY exmr/ui/web/ ./
RUN npm run build

###############################################################################
# 2. runtime (CUDA 12.8 + cuDNN 9, matching torch 2.8.0 / cu128)
###############################################################################
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3-pip \
        ffmpeg \
        git \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libegl1 \
        libturbojpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --upgrade pip setuptools wheel

# torch / torchvision are pinned by their own index, as in docs/INSTALL.md.
RUN pip install \
        torch==2.8.0 \
        torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app

# Dependency layer first, so source edits do not invalidate the heavy install.
COPY requirements.txt constraints.txt pyproject.toml ./
# ExMR's pinned requirements, the VRMExMR UI server (fastapi/uvicorn are not part of
# ExMR's own requirements) and the two pinned upstream packages installed without deps.
RUN pip install -r requirements.txt -c constraints.txt \
    && pip install fastapi==0.162.0 uvicorn==0.54.0 \
    && pip install --no-deps \
        "git+https://github.com/geopavlakos/hamer.git@3a01849f4148352e9260b69bf28b65d1671a4905" \
        "git+https://github.com/facebookresearch/sapiens2.git@7e5bae88456ac418ff0e58e74106c9fe192055d4"

# Application source, third-party submodule and the web UI build.
COPY . .
COPY --from=web /src/exmr/ui/dist ./exmr/ui/dist
RUN pip install --no-deps -e .

# --------------------------------------------------------------------------
# Model files: copied from the build context, never fetched at runtime.
#   checkpoints/                              ~11.5 GB, all ExMR weights
#   docker/torch_hub.tar.gz                   pinned DINOv3 code for torch.hub
# --------------------------------------------------------------------------
RUN mkdir -p /root/.cache/torch/hub \
    && printf 'facebookresearch/dinov3\n' >> /root/.cache/torch/hub/trusted_list
ADD docker/torch_hub.tar.gz /root/.cache/torch/

# --------------------------------------------------------------------------
# Runtime configuration
# --------------------------------------------------------------------------
ENV TORCH_HOME=/root/.cache/torch \
    EXMR_CHECKPOINT_DIR=/app/checkpoints \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    YOLO_CONFIG_DIR=/app/.cache/ultralytics \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    EXMR_PIN_MEMORY=0

RUN mkdir -p \
        /app/exmr/ui/videos \
        /app/outputs/ui \
        /app/outputs/ui_exports \
        /app/.cache/ultralytics \
        /app/.cache/matplotlib

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8010/api/status || exit 1

CMD ["python", "-m", "exmr.ui.server", "--host", "0.0.0.0", "--port", "8010"]
