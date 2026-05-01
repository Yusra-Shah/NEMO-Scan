"""
PneumoScan — Modal serverless deployment (Modal SDK 1.4.x)
Serves the FastAPI web app with real float16 AI inference on CPU.

─── One-time setup ─────────────────────────────────────────────────────────────
1.  pip install modal                   (already installed)
2.  modal setup                         (opens browser to authenticate)
3.  modal secret create nemo-scan-mongodb MONGODB_URI="mongodb+srv://..."

─── Deploy ─────────────────────────────────────────────────────────────────────
    modal deploy modal_app.py

    The public HTTPS URL is printed after deploy and in modal.com/apps.

─── Local test before deploying ────────────────────────────────────────────────
    modal serve modal_app.py            (live-reloads; Ctrl+C to stop)

─── API changes from older Modal versions ───────────────────────────────────────
modal.Mount           → removed; use image.add_local_dir() / .add_local_file()
keep_warm             → renamed to min_containers
@modal.asgi_app()     → still valid (serves any ASGI/FastAPI app)
"""

from pathlib import Path
import modal

ROOT = Path(__file__).parent
FLOAT16_DIR = ROOT / "weights" / "lung" / "float16"

# ── Ignore helper ──────────────────────────────────────────────────────────────
# Determines which paths to skip when uploading the project root.
# Called with a Path relative to ROOT; returns True = skip.
_SKIP_DIRS = frozenset({
    "nemo_env", "weights", "__pycache__", ".git",
    "outputs", "domains", "gui",
})
_SKIP_EXTS = frozenset({".pyc", ".pyo", ".zip", ".tar", ".gz"})


def _ignore_source(p: Path) -> bool:
    return bool(set(p.parts) & _SKIP_DIRS) or p.suffix in _SKIP_EXTS


# ── Container image ────────────────────────────────────────────────────────────
# Build order matters for layer caching:
#   1. System packages   — changes rarely → cached almost always
#   2. Python packages   — changes on dep bumps → cached until requirements change
#   3. Source code       — changes often → NOT baked in (add_local_dir copy=False)
#   4. Model weights     — changes never → baked into a cached layer (copy=True)
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
    # workdir must come before any add_local_* calls.
    # Modal 1.4 rule: no build steps (workdir, run_commands, etc.) after
    # add_local_dir(copy=False); they ARE allowed after add_local_dir(copy=True).
    .workdir("/app")
    # Float16 weights baked into a cached image layer (copy=True).
    # 330 MB uploaded once; this layer is reused on every subsequent deploy
    # as long as the files in weights/lung/float16/ do not change.
    .add_local_dir(
        str(FLOAT16_DIR),
        remote_path="/app/weights/lung/float16",
        copy=True,
    )
    # Source code injected fresh on every deploy without rebuilding image layers
    # (copy=False, the default; Modal injects these files at container startup).
    # Must be the last step in the chain.
    .add_local_dir(
        str(ROOT),
        remote_path="/app",
        ignore=_ignore_source,
    )
)

# ── App ────────────────────────────────────────────────────────────────────────
app = modal.App("pneumo-scan")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("nemo-scan-mongodb")],
    cpu=2.0,
    memory=4096,        # 4 GB — holds all 7 float16 models (~330 MB) plus overhead
    timeout=300,        # max seconds per request; cold-start model loading ~60-90 s
    startup_timeout=180,# extra budget for container to become ready (model load)
    min_containers=0,   # spin down when idle (free-tier friendly); was keep_warm=0
)
@modal.asgi_app()
def serve():
    """
    Returns the FastAPI ASGI app to Modal's HTTP gateway.

    Importing web_app triggers the FastAPI startup event which auto-detects
    weights/lung/float16/ and loads all 7 models in float16 precision.
    The /app working directory is already on sys.path via .workdir('/app').
    """
    from web_app import app as fastapi_app
    return fastapi_app
