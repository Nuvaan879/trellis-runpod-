"""
TRELLIS inference wrapper.
The model is loaded ONCE when the RunPod worker starts.
Each request calls run_inference() without reloading the model.
"""

import os
import io
import base64
import tempfile
import sys

import torch
from PIL import Image

# TRELLIS is installed at /workspace/TRELLIS
sys.path.insert(0, '/workspace/TRELLIS')

# ── Global pipeline object (None until first load) ────────────────
pipeline = None


def load_model():
    """
    Load the TRELLIS pipeline onto the GPU.

    Called once when the RunPod worker starts.
    After that, the pipeline stays in GPU memory for all requests.

    Model source priority:
      1. If MODEL_PATH env var is set → load from that path (network volume)
      2. Otherwise → download from Hugging Face (first time takes ~10 min)
    """
    global pipeline
    if pipeline is not None:
        print("[TRELLIS] Model already loaded, skipping.", flush=True)
        return pipeline

    from trellis.pipelines import TrellisImageTo3DPipeline

    # Check if model is on a RunPod Network Volume (faster) or needs download
    model_path = os.environ.get("MODEL_PATH", "JeffreyXiang/TRELLIS-image-large")
    print(f"[TRELLIS] Loading pipeline from: {model_path}", flush=True)

    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)
    pipeline.cuda()

    print("[TRELLIS] Pipeline loaded and on GPU. Ready.", flush=True)
    return pipeline


def run_inference(image_base64: str, seed: int = 42) -> str:
    """
    Run TRELLIS image-to-3D generation.

    Args:
        image_base64 : Base64-encoded input image (PNG or JPG)
        seed         : Random seed (same seed + same image = same output)

    Returns:
        Base64-encoded GLB file string
    """
    from trellis.utils import postprocessing_utils

    pipe = load_model()

    # ── 1. Decode the base64 image ────────────────────────────────
    print("[TRELLIS] Decoding input image...", flush=True)
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    print(f"[TRELLIS] Image size: {image.size}", flush=True)

    # ── 2. Run the TRELLIS pipeline ───────────────────────────────
    print(f"[TRELLIS] Running inference (seed={seed})...", flush=True)
    with torch.no_grad():
        outputs = pipe.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],  # gaussian for texture, mesh for geometry
            preprocess_image=True,         # auto-removes background
        )
    print("[TRELLIS] Inference done. Exporting GLB...", flush=True)

    # ── 3. Combine gaussian texture + mesh into a GLB file ────────
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],   # contains texture/appearance info
        outputs['mesh'][0],       # contains geometry (vertices, faces)
        simplify=0.95,            # 0.95 = slight mesh simplification (smaller file)
        texture_size=1024,        # 1024x1024 texture (quality / size balance)
    )

    # ── 4. Save to a temp file, read bytes, then delete the temp ──
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        glb_path = tmp.name
    glb.export(glb_path)

    with open(glb_path, "rb") as f:
        glb_bytes = f.read()
    os.unlink(glb_path)  # delete temp file

    file_size_kb = len(glb_bytes) / 1024
    print(f"[TRELLIS] GLB exported successfully ({file_size_kb:.1f} KB)", flush=True)

    # ── 5. Free VRAM so the worker survives many sequential jobs ──
    torch.cuda.empty_cache()

    # ── 6. Return as base64 string ────────────────────────────────
    return base64.b64encode(glb_bytes).decode("utf-8")
