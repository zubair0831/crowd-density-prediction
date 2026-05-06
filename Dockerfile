# ═══════════════════════════════════════════════════════════════════════════════
# Dockerfile — Crowd Intelligence System  v3.2
# Build context: the SDP/ folder  →  docker build -t crowd-backend .
#
# Base: python:3.11-slim  (~150 MB vs 3.6 GB for the CUDA image)
# PyTorch: CPU-only wheels (~800 MB total) — correct for Mac / no-GPU hosts
#
# If you later deploy to a Linux GPU server, replace the torch install line
# with the cu121 index URL:
#   https://download.pytorch.org/whl/cu121
# ═══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# ── System packages ───────────────────────────────────────────────────────────
# libgl1 + libglib2.0-0 — required by OpenCV at runtime (even headless build)
# libgomp1              — required by PyTorch CPU kernels
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch CPU wheels (install first, separately) ────────────────────────────
# The CPU wheel set is ~800 MB vs 3.6 GB for the CUDA base image.
# Doing this in its own layer means Docker won't re-download torch on every
# source-file change — only when requirements.txt changes.
RUN pip install --no-cache-dir \
        torch==2.2.0 \
        torchvision==0.17.0 \
        --index-url https://download.pytorch.org/whl/cpu

# ── Remaining Python dependencies ─────────────────────────────────────────────
COPY requirements.txt .
# torch/torchvision already satisfied above; pip skips them automatically
RUN pip install --no-cache-dir -r requirements.txt \
 && pip uninstall -y opencv-python 2>/dev/null || true \
 && pip install --no-cache-dir "opencv-python-headless>=4.9.0.80"

# ── Backend source files ──────────────────────────────────────────────────────
COPY backend.py \
     analyzers.py \
     constants.py \
     database.py \
     segformer_walkable.py \
     walkable_area_estimator.py \
     zones.py \
     ./

# ── YOLO weights ──────────────────────────────────────────────────────────────
COPY yolov8n.pt .

# ── P2PNet (local repo — already cloned, copy as-is) ─────────────────────────
COPY CrowdCounting-P2PNet/ ./CrowdCounting-P2PNet/

# ── HuggingFace model cache dir ───────────────────────────────────────────────
# SegFormer (~400 MB) downloads here on first run.
# docker-compose mounts a named volume so it persists across restarts.
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
RUN mkdir -p /app/.cache/huggingface

# ── Runtime env ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "backend:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "30"]