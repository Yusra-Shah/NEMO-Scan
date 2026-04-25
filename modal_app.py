"""
NEMO Scan — Modal serverless deployment
Serves the FastAPI web app with real float16 AI inference on CPU.

─── One-time setup ─────────────────────────────────────────────────────────────
1.  pip install modal
2.  modal setup               (opens browser to authenticate)
3.  modal secret create nemo-scan-mongodb MONGODB_URI="mongodb+srv://..."

─── Deploy ─────────────────────────────────────────────────────────────────────
    modal deploy modal_app.py

    The public HTTPS URL is printed after deploy and shown in modal.com/apps.

─── Local test before deploying ────────────────────────────────────────────────
    modal serve modal_app.py  (live-reloads on file save, Ctrl+C to stop)

─── Notes ──────────────────────────────────────────────────────────────────────
- Weights (330 MB float16) are uploaded from weights/lung/float16/ on every
  deploy via modal.Mount.  For a permanent volume (upload once, reuse across
  deploys) switch weights_mount to a modal.Volume — see comment below.
- keep_warm=0 means the container spins down when idle (free-tier friendly).
  Cold start takes ~60-90 s on CPU while models load.  Set keep_warm=1 for
  instant responses but that incurs continuous cost.
- cpu=2 + memory=4096 comfortably fits all 7 float16 models (~330 MB) plus
  inference overhead.
"""

import modal
from pathlib import Path

ROOT = Path(__file__).parent

# ── Container image ────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgomp1",
    )
    .pip_install(
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "python-multipart>=0.0.9",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "opencv-python-headless>=4.8.0",
        "pillow>=10.0.0",
        "pymongo>=4.6.0",
        "bcrypt>=4.1.0",
        "numpy>=1.24.0",
        "grad-cam>=1.5.0",
        "python-dotenv>=1.0.0",
        "reportlab>=4.0.0",
    )
)

# ── Source-code mount ──────────────────────────────────────────────────────────
# Uploaded on every `modal deploy`.  Heavy / generated directories are excluded.
_EXCLUDE = {
    "nemo_env", "weights", "__pycache__", ".git",
    "outputs", "domains", "gui",           # gui is PySide6-only, not needed
}

source_mount = modal.Mount.from_local_dir(
    str(ROOT),
    remote_path="/app",
    condition=lambda p: not any(
        seg in _EXCLUDE
        for seg in Path(p).parts
    ) and not p.endswith((".pyc", ".pyo", ".zip", ".tar", ".gz")),
)

# ── Weights mount ──────────────────────────────────────────────────────────────
# Float16 weights: 330 MB (half of float32).
# Uploaded from local weights/lung/float16/ on every deploy.
#
# Alternative (upload once, reuse across deploys):
#   weights_vol = modal.Volume.from_name("nemo-scan-weights", create_if_missing=True)
#   Then replace `mounts=[..., weights_mount]` with
#   `volumes={"/app/weights/lung/float16": weights_vol}`
#   and run `modal volume put nemo-scan-weights weights/lung/float16/ /`
#   once from the command line before the first deploy.
weights_mount = modal.Mount.from_local_dir(
    str(ROOT / "weights" / "lung" / "float16"),
    remote_path="/app/weights/lung/float16",
)

# ── Modal app ──────────────────────────────────────────────────────────────────
app = modal.App("nemo-scan")


@app.function(
    image=image,
    mounts=[source_mount, weights_mount],
    secrets=[modal.Secret.from_name("nemo-scan-mongodb")],
    cpu=2.0,
    memory=4096,   # 4 GB RAM — holds all 7 float16 models comfortably
    timeout=300,   # 5-min budget covers cold-start model loading (~60-90 s)
    keep_warm=0,   # 0 = spin down when idle (free tier); 1 = always-on
)
@modal.asgi_app()
def serve():
    """
    Entry point for Modal's ASGI serving.
    Importing web_app triggers FastAPI app creation and the startup lifecycle,
    which auto-detects weights/lung/float16/ and loads models in float16.
    """
    import sys
    sys.path.insert(0, "/app")
    from web_app import app as fastapi_app  # noqa: PLC0415
    return fastapi_app
