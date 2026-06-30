# ──────────────────────────────────────────────────────────────────
# TRELLIS RunPod Serverless Worker
# Base: CUDA 11.8 + Ubuntu 22.04 full dev toolkit
# TRELLIS version: v1 (JeffreyXiang/TRELLIS-image-large, ~2B params)
# ──────────────────────────────────────────────────────────────────
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# ── Environment variables ─────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
# GPU architectures to compile CUDA extensions for:
# 8.0 = A100, 8.6 = RTX 3090/3080, 8.9 = RTX 4090
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX"
# Parallel compilation jobs (reduces compile time)
ENV MAX_JOBS=4

# ── TRELLIS runtime config (FIX: not in original guide) ───────────
# nvdiffrast needs a GL context. On a headless serverless GPU there is
# no display, so force the EGL backend or rendering crashes at inference.
ENV PYOPENGL_PLATFORM=egl
# TRELLIS reads these. native spconv avoids startup autotuning stalls.
ENV ATTN_BACKEND=flash-attn
ENV SPCONV_ALGO=native

# ── System packages ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    ninja-build \
    build-essential \
    cmake \
    libglfw3-dev \
    libgles2-mesa-dev \
    libegl1-mesa-dev \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN pip install --upgrade pip setuptools wheel

# ── PyTorch with CUDA 11.8 support ───────────────────────────────
# This must come before any CUDA extension installs
RUN pip install \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu118

# ── Core Python libraries ─────────────────────────────────────────
RUN pip install \
    pillow \
    "imageio[ffmpeg]" \
    tqdm \
    easydict \
    scipy \
    ninja \
    einops \
    omegaconf \
    opencv-python-headless \
    rembg \
    onnxruntime \
    trimesh \
    xatlas \
    pyvista \
    pymeshfix \
    igraph \
    transformers \
    diffusers \
    accelerate \
    huggingface_hub \
    sentencepiece \
    runpod

# ── xformers (pre-built — fast to install) ───────────────────────
RUN pip install xformers==0.0.27.post2 \
    --index-url https://download.pytorch.org/whl/cu118

# ── flash-attn (compiles from source — takes 20–40 min) ──────────
# This is the slowest step. Do not interrupt it.
RUN pip install flash-attn==2.6.3 --no-build-isolation

# ── Clone TRELLIS repository with all submodules ─────────────────
WORKDIR /workspace
RUN git clone --recurse-submodules \
    https://github.com/microsoft/TRELLIS.git

WORKDIR /workspace/TRELLIS

# ── spconv: sparse 3D convolutions ───────────────────────────────
RUN pip install spconv-cu118

# ── nvdiffrast: differentiable rasterization ─────────────────────
RUN pip install git+https://github.com/NVlabs/nvdiffrast.git

# ── NVIDIA Kaolin: 3D deep learning library ──────────────────────
RUN pip install kaolin==0.17.0 \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html

# ── diffoctreerast: TRELLIS custom CUDA octree renderer ──────────
# Compiles from source. Takes 5–15 min.
RUN pip install --no-build-isolation \
    "git+https://github.com/JeffreyXiang/diffoctreerast.git"

# ── mip-gaussian-rasterization: for Gaussian splatting output ────
# Compiles from source. Takes 5–10 min.
RUN pip install --no-build-isolation \
    "git+https://github.com/autonomousvision/mip-splatting.git@main#subdirectory=submodules/diff-gaussian-rasterization"

# ── Install TRELLIS itself as a Python package ────────────────────
RUN pip install -e .

# ── Pre-download rembg background-removal model (FIX) ─────────────
# preprocess_image=True uses rembg/u2net, which otherwise downloads
# ~170MB on the FIRST request and can stall the first job. Bake it in.
RUN python -c "from rembg import new_session; new_session('u2net')" || true

# ── Copy your handler files into the image ────────────────────────
COPY inference.py /workspace/inference.py
COPY handler.py /workspace/handler.py

# ── Set working directory and startup command ─────────────────────
WORKDIR /workspace
CMD ["python", "-u", "handler.py"]
