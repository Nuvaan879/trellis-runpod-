"""
RunPod Serverless handler for TRELLIS image-to-3D.

─── API Request Format ───────────────────────────────────────────
POST https://api.runpod.ai/v2/<endpoint-id>/run
Headers:
  Authorization: Bearer <your-runpod-api-key>
  Content-Type: application/json
Body:
  {
    "input": {
      "image": "<base64-encoded PNG or JPG>",
      "seed": 42                              (optional, default: 42)
    }
  }

─── API Response ─────────────────────────────────────────────────
Success:
  { "glb": "<base64-encoded GLB file>" }
Error:
  { "error": "error message here" }
"""

import runpod
from inference import load_model, run_inference

# ── Pre-load model at worker startup ─────────────────────────────
# This runs ONCE when the worker container starts.
# All subsequent requests reuse the loaded model (much faster).
# If this fails, the worker will show as unhealthy in RunPod.
print("[Handler] Worker starting. Pre-loading TRELLIS model...", flush=True)
load_model()
print("[Handler] Model loaded and ready. Waiting for jobs.", flush=True)


def handler(job):
    """
    RunPod calls this function once per incoming API request.

    `job` is a dict like:
      {
        "id": "job-xxxx",
        "input": {
          "image": "...",
          "seed": 42
        }
      }
    """
    try:
        job_input = job.get("input", {})

        # ── Validate required input ───────────────────────────────
        image_base64 = job_input.get("image")
        if not image_base64:
            return {
                "error": "Missing required field: 'image'. "
                         "Provide a base64-encoded PNG or JPG."
            }

        # ── Get optional parameters ───────────────────────────────
        seed = int(job_input.get("seed", 42))

        # ── Run the 3D generation ─────────────────────────────────
        glb_base64 = run_inference(image_base64, seed=seed)

        return {"glb": glb_base64}

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"[Handler] ERROR:\n{error_msg}", flush=True)
        return {"error": str(e)}


# ── Start the RunPod serverless worker loop ───────────────────────
# This keeps the worker alive and ready to receive jobs.
runpod.serverless.start({"handler": handler})
