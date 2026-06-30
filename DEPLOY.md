# TRELLIS RunPod — Deployment Checklist

This picks up **after** the GitHub Actions build pushes the image to Docker Hub.

- **Image:** `YOUR_DOCKERHUB_USERNAME/trellis-api:v1.0.0`
- **Repo:** https://github.com/Nuvaan879/trellis-runpod-

Replace `YOUR_DOCKERHUB_USERNAME`, `YOUR_HF_TOKEN`, `YOUR_ENDPOINT_ID`,
`YOUR_RUNPOD_API_KEY` below with your real values.

---

## 0. Verify the image is on Docker Hub
After the green build, confirm it exists:
- https://hub.docker.com/r/YOUR_DOCKERHUB_USERNAME/trellis-api → you should see the `v1.0.0` tag.
- Make sure the repo is **Public** (Docker Hub → repo → Settings → Visibility),
  otherwise RunPod can't pull it without credentials.

---

## 1. (Recommended) Network volume for model weights
Loads the model in ~30s instead of downloading ~10GB on every cold start.

### 1a. Create the volume
RunPod → **Storage → + Network Volume**
- Name: `trellis-models`
- Size: `50 GB`
- **Region: note it** — your serverless endpoint MUST be in the same region.

### 1b. Download weights into it (temporary pod)
RunPod → **Pods → + Deploy**
- GPU: cheapest available (RTX 3090 / A4000)
- Template: `RunPod Pytorch 2.4`
- Attach volume `trellis-models` at mount path `/runpod-volume`
- Deploy → Connect → **Start Web Terminal**, then run:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login        # paste YOUR_HF_TOKEN when prompted
huggingface-cli download \
  JeffreyXiang/TRELLIS-image-large \
  --local-dir /runpod-volume/models/TRELLIS-image-large
ls -lh /runpod-volume/models/TRELLIS-image-large/
```

When done, **Stop** the pod (you're billed only while it runs).

> Skip this whole section to keep it simple — the worker will just download
> weights from Hugging Face on first cold start (slower, but works). If you
> skip it, do NOT set `MODEL_PATH` in step 2.

---

## 2. Create the serverless endpoint
RunPod → **Serverless → + New Endpoint → Custom Container**

| Field | Value |
|-------|-------|
| Endpoint Name | `trellis-api` |
| Container Image | `YOUR_DOCKERHUB_USERNAME/trellis-api:v1.0.0` |
| Container Disk | `20 GB` |
| GPU | check **RTX 4090** (24GB) + **A100 (40GB)** as fallback |
| Min Workers | `0` |
| Max Workers | `3` |
| Idle Timeout | `30` sec |
| Execution Timeout | `300` sec |

**Volume** (only if you did step 1):
- Network Volume: `trellis-models`
- Mount Path: `/runpod-volume`
- Endpoint region MUST match the volume's region.

**Environment variables** (+ Add Variable):
| Name | Value |
|------|-------|
| `HF_TOKEN` | `YOUR_HF_TOKEN` |
| `MODEL_PATH` | `/runpod-volume/models/TRELLIS-image-large` *(only if you did step 1)* |

Click **Deploy Endpoint**. First deploy pulls the image (a few min), starts a
worker, loads the model, then shows **Ready**.

---

## 3. Get credentials
- **Endpoint ID:** shown on the endpoint's page (e.g. `abc1xyz2efg3`).
- **API key:** RunPod → Settings → **API Keys → + API Key** → copy (starts `rp_...`).

---

## 4. Test the live endpoint
`test_input.json` is already in this repo. Submit a job:

```bash
# extract the base64 image from test_input.json and submit
IMG=$(python3 -c "import json;print(json.load(open('test_input.json'))['input']['image'])")

curl -s -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"input\": {\"image\": \"$IMG\", \"seed\": 42}}"
```

Copy the returned `id`, then poll:

```bash
curl -s -X GET https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/JOB_ID \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY"
```

When `status` is `COMPLETED`, `output.glb` holds the base64 GLB. Save it:

```bash
python3 - <<'PY'
import base64, json
resp = json.load(open('status.json'))   # paste the COMPLETED response into status.json
open('output.glb','wb').write(base64.b64decode(resp['output']['glb']))
print('saved output.glb')
PY
```

Open `output.glb` in any GLB viewer (https://gltf-viewer.donmccurdy.com, Blender, Windows 3D Viewer).

---

## Troubleshooting (this build specifically)
- **Worker stuck "Initializing" / unhealthy** → endpoint **Logs** tab. Most likely
  the model load failed: check `MODEL_PATH` matches the actual path on the volume,
  or remove `MODEL_PATH` to fall back to HF download.
- **First job very slow** → cold start (worker boot + model load). Subsequent jobs
  on a warm worker are ~30–120s.
- **CUDA out of memory** → use A100 (40GB), or lower `texture_size` 1024→512 in `inference.py`.
- **Image won't pull** → Docker Hub repo is private; make it Public or add Docker Hub
  credentials in RunPod → Settings → Credentials.
