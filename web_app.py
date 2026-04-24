"""
NEMO Scan — FastAPI web backend for Hugging Face Spaces.
Mirrors the PySide6 desktop app; connects to the same MongoDB Atlas cluster.

Run locally:  python web_app.py
HF Spaces:    uvicorn web_app:app --host 0.0.0.0 --port 7860
"""

import os
import sys
import base64
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import database.db as db

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="NEMO Scan", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Inference engine (loaded once at startup) ─────────────────────────────────
_engine = None


@app.on_event("startup")
async def _startup():
    global _engine
    try:
        from core.inference.engine import InferenceEngine
        _engine = InferenceEngine()
        weights_dir = os.path.join(ROOT, "weights", "lung")
        await asyncio.get_event_loop().run_in_executor(
            None, _engine.load_models, weights_dir
        )
        print("✓ InferenceEngine loaded.")
    except Exception as exc:
        print(f"Warning: Could not load inference engine: {exc}")
        _engine = None


# ── Static files ──────────────────────────────────────────────────────────────
_static = os.path.join(ROOT, "static")
os.makedirs(_static, exist_ok=True)
app.mount("/static", StaticFiles(directory=_static), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(_static, "index.html"))


# ── Auth ──────────────────────────────────────────────────────────────────────
@app.post("/api/login")
async def login(request: Request):
    body = await request.json()
    email    = body.get("email", "").strip()
    password = body.get("password", "")
    if not email or not password:
        raise HTTPException(400, "Email and password are required.")
    try:
        doctor = db.login_doctor(email, password)
        if not doctor:
            raise HTTPException(401, "Invalid credentials.")
        return JSONResponse({"doctor": _serial(doctor)})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(401, str(exc))


# ── Scan ──────────────────────────────────────────────────────────────────────
@app.post("/api/scan")
async def scan(
    image:       UploadFile = File(...),
    doctor_id:   str        = Form(...),
    patient_id:  str        = Form(""),
    doctor_name: str        = Form(""),
):
    if _engine is None:
        raise HTTPException(503, "Inference engine not available — model weights not loaded.")

    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    heatmaps_dir = os.path.join(ROOT, "outputs", "heatmaps")
    os.makedirs(heatmaps_dir, exist_ok=True)

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _engine.predict_full(tmp_path, heatmaps_dir=heatmaps_dir),
        )
    except Exception as exc:
        os.unlink(tmp_path)
        raise HTTPException(500, f"Inference failed: {exc}")

    # Encode heatmap and original image as base64 for the browser
    hp = result.get("heatmap_path", "") or ""
    heatmap_b64 = ""
    if hp and os.path.exists(hp):
        with open(hp, "rb") as f:
            heatmap_b64 = base64.b64encode(f.read()).decode()

    with open(tmp_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    os.unlink(tmp_path)

    # Persist to MongoDB when a patient is linked
    scan_id = ""
    if patient_id and doctor_id:
        try:
            result_dict = {
                "prediction":    result.get("prediction", "Unknown"),
                "confidence":    result.get("confidence", 0.0),
                "ensemble_prob": result.get("ensemble_prob", 0.0),
                "severity":      result.get("severity", "None"),
                "subtype":       result.get("subtype", "N/A"),
            }
            votes = dict(result.get("model_votes", {}))
            votes.setdefault("attention_cnn", 0.0)
            saved = db.save_scan(
                patient_id=patient_id,
                doctor_id=doctor_id,
                image_path="",
                heatmap_path=hp,
                report_path="",
                result=result_dict,
                model_votes=votes,
                doctor_notes="",
                processing_time_ms=result.get("processing_time_ms", 0),
            )
            scan_id = str(saved.get("scan_id", ""))
        except Exception as exc:
            print(f"Warning: could not save scan to MongoDB: {exc}")

    return JSONResponse({
        "prediction":         result.get("prediction", ""),
        "confidence":         result.get("confidence", 0.0),
        "ensemble_prob":      result.get("ensemble_prob", 0.0),
        "severity":           result.get("severity", ""),
        "subtype":            result.get("subtype", ""),
        "heatmap_b64":        heatmap_b64,
        "image_b64":          image_b64,
        "model_votes":        result.get("model_votes", {}),
        "processing_time_ms": result.get("processing_time_ms", 0),
        "scan_id":            scan_id,
    })


# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.get("/api/dashboard")
async def dashboard(doctor_id: str):
    try:
        stats  = db.get_dashboard_stats(doctor_id)
        recent = db.get_recent_activity(doctor_id, limit=5)
        return JSONResponse({"stats": _serial(stats), "recent": _serial(recent)})
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ── Patients ──────────────────────────────────────────────────────────────────
@app.get("/api/patients")
async def get_patients(doctor_id: str, q: str = ""):
    try:
        patients = db.search_patients(q, doctor_id) if q else db.get_all_patients(doctor_id)
        return JSONResponse(_serial(patients))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: str):
    try:
        patient = db.get_patient_by_id(patient_id)
        if not patient:
            raise HTTPException(404, "Patient not found.")
        return JSONResponse(_serial(patient))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/patients/{patient_id}/scans")
async def get_patient_scans(patient_id: str):
    try:
        scans = db.get_scans_for_patient(patient_id)
        return JSONResponse(_serial(scans))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/patients")
async def register_patient(request: Request):
    body = await request.json()
    try:
        patient = db.register_patient(
            name=body["name"],
            age=int(body["age"]),
            gender=body.get("gender", "Other"),
            contact=body.get("contact", ""),
            symptoms=body.get("symptoms", ""),
            medical_history=body.get("medical_history", ""),
            assigned_doctor_id=body["doctor_id"],
            patient_id=body.get("patient_id") or None,
        )
        return JSONResponse(_serial(patient))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ── Serialisation helper ──────────────────────────────────────────────────────
def _serial(obj):
    """Recursively make MongoDB documents JSON-safe."""
    try:
        from bson import ObjectId
        _oid = ObjectId
    except ImportError:
        _oid = type(None)

    if isinstance(obj, list):
        return [_serial(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serial(v) for k, v in obj.items()}
    if isinstance(obj, _oid):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)
