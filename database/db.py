"""
NEMO Scan - Database Layer
database/db.py

All MongoDB operations live here. GUI never imports pymongo directly.
Collections: doctors, patients, scans, audit_log
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import bcrypt
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure

load_dotenv()

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

_client: Optional[MongoClient] = None
_db = None


def get_db():
    """Return the database handle. Creates connection on first call."""
    global _client, _db
    if _client is None:
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise EnvironmentError(
                "MONGODB_URI is not set. Check your .env file."
            )
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")  # fail fast if unreachable
        _db = _client[os.getenv("DB_NAME", "nemo_scan")]
        _ensure_indexes(_db)
    return _db


def close_connection():
    """Call this on application exit."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None


def _ensure_indexes(db):
    """Create indexes once on startup."""
    db.doctors.create_index([("email", ASCENDING)], unique=True)
    db.patients.create_index([("patient_id", ASCENDING)], unique=True)
    db.patients.create_index([("assigned_doctor_id", ASCENDING)])
    db.scans.create_index([("patient_id", ASCENDING)])
    db.scans.create_index([("doctor_id", ASCENDING)])
    db.scans.create_index([("scan_date", DESCENDING)])
    db.audit_log.create_index([("timestamp", DESCENDING)])
    db.audit_log.create_index([("doctor_id", ASCENDING)])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _generate_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def register_doctor(
    name: str,
    email: str,
    password: str,
    specialization: str = "General Physician",
) -> dict:
    """
    Create a new doctor account.
    Returns the created doctor document (without password_hash).
    Raises ValueError if email already exists.
    """
    db = get_db()

    if db.doctors.find_one({"email": email.lower().strip()}):
        raise ValueError(f"An account with email '{email}' already exists.")

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")

    doctor = {
        "doctor_id": _generate_id("DOC"),
        "name": name.strip(),
        "email": email.lower().strip(),
        "password_hash": password_hash,
        "specialization": specialization.strip(),
        "created_at": _now(),
        "last_login": None,
        "is_active": True,
        "total_scans": 0,
    }

    db.doctors.insert_one(doctor)
    _write_audit(
        action="doctor_registered",
        doctor_id=doctor["doctor_id"],
        details={"name": name, "email": email},
    )

    return _safe_doctor(doctor)


def login_doctor(email: str, password: str) -> Optional[dict]:
    """
    Verify credentials. Returns safe doctor dict on success, None on failure.
    Updates last_login on success.
    """
    db = get_db()
    record = db.doctors.find_one({"email": email.lower().strip()})

    if not record:
        return None
    if not record.get("is_active", True):
        return None

    if not bcrypt.checkpw(password.encode("utf-8"), record["password_hash"].encode("utf-8")):
        return None

    db.doctors.update_one(
        {"doctor_id": record["doctor_id"]},
        {"$set": {"last_login": _now()}},
    )

    _write_audit(
        action="doctor_login",
        doctor_id=record["doctor_id"],
        details={"email": email},
    )

    return _safe_doctor(record)


def _safe_doctor(doc: dict) -> dict:
    """Return doctor dict with password_hash removed."""
    return {k: v for k, v in doc.items() if k not in ("password_hash", "_id")}


def get_doctor_by_id(doctor_id: str) -> Optional[dict]:
    db = get_db()
    record = db.doctors.find_one({"doctor_id": doctor_id})
    return _safe_doctor(record) if record else None


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------

def register_patient(
    name: str,
    age: int,
    gender: str,
    contact: str,
    symptoms: str,
    medical_history: str,
    assigned_doctor_id: str,
    patient_id: Optional[str] = None,
) -> dict:
    """
    Register a new patient. Uses transaction + audit log.
    Returns the created patient document.
    Raises ValueError if patient_id already exists.
    """
    db = get_db()

    pid = patient_id.strip().upper() if patient_id else _generate_id("PAT")

    if db.patients.find_one({"patient_id": pid}):
        raise ValueError(f"Patient ID '{pid}' already exists.")

    patient = {
        "patient_id": pid,
        "name": name.strip(),
        "age": int(age),
        "gender": gender.strip(),
        "contact": contact.strip(),
        "symptoms": symptoms.strip(),
        "medical_history": medical_history.strip(),
        "assigned_doctor_id": assigned_doctor_id,
        "created_at": _now(),
        "total_scans": 0,
        "last_scan_date": None,
        "last_diagnosis": None,
    }

    with _client.start_session() as session:
        with session.start_transaction():
            db.patients.insert_one(patient, session=session)
            _write_audit(
                action="patient_registered",
                doctor_id=assigned_doctor_id,
                patient_id=pid,
                details={"name": name, "age": age, "gender": gender},
                session=session,
            )

    return {k: v for k, v in patient.items() if k != "_id"}


def get_patient_by_id(patient_id: str) -> Optional[dict]:
    db = get_db()
    record = db.patients.find_one({"patient_id": patient_id.upper()})
    return {k: v for k, v in record.items() if k != "_id"} if record else None


def search_patients(query: str, doctor_id: Optional[str] = None) -> list:
    """
    Search patients by name or patient_id substring.
    Optionally filter to a specific doctor's patients.
    Returns list of patient dicts.
    """
    db = get_db()
    text_filter = {
        "$or": [
            {"name": {"$regex": query, "$options": "i"}},
            {"patient_id": {"$regex": query, "$options": "i"}},
        ],
        "is_active": {"$ne": False},
    }
    if doctor_id:
        text_filter["assigned_doctor_id"] = doctor_id

    cursor = db.patients.find(text_filter).sort("name", ASCENDING).limit(50)
    return [{k: v for k, v in doc.items() if k != "_id"} for doc in cursor]


def get_all_patients(doctor_id: Optional[str] = None, limit: int = 200) -> list:
    db = get_db()
    filt: dict = {"is_active": {"$ne": False}}
    if doctor_id:
        filt["assigned_doctor_id"] = doctor_id
    cursor = db.patients.find(filt).sort("created_at", DESCENDING).limit(limit)
    return [{k: v for k, v in doc.items() if k != "_id"} for doc in cursor]


def update_patient_info(patient_id: str, updates: dict, doctor_id: str) -> bool:
    """Update editable patient fields."""
    db = get_db()
    allowed = {"name", "age", "gender", "contact", "symptoms", "medical_history"}
    safe_updates = {k: v for k, v in updates.items() if k in allowed}
    if not safe_updates:
        return False

    with _client.start_session() as session:
        with session.start_transaction():
            result = db.patients.update_one(
                {"patient_id": patient_id},
                {"$set": safe_updates},
                session=session,
            )
            _write_audit(
                action="patient_updated",
                doctor_id=doctor_id,
                patient_id=patient_id,
                details={"fields_updated": list(safe_updates.keys())},
                session=session,
            )

    return result.modified_count > 0


def deactivate_patient(patient_id: str, doctor_id: str) -> bool:
    """Soft-delete a patient by setting is_active: False. Data is preserved."""
    db = get_db()
    with _client.start_session() as session:
        with session.start_transaction():
            result = db.patients.update_one(
                {"patient_id": patient_id},
                {"$set": {"is_active": False, "deactivated_at": _now()}},
                session=session,
            )
            _write_audit(
                action="patient_deactivated",
                doctor_id=doctor_id,
                patient_id=patient_id,
                details={},
                session=session,
            )
    return result.modified_count > 0


# ---------------------------------------------------------------------------
# Scans
# ---------------------------------------------------------------------------

def save_scan(
    patient_id: str,
    doctor_id: str,
    image_path: str,
    heatmap_path: str,
    report_path: str,
    result: dict,
    model_votes: dict,
    doctor_notes: str = "",
    processing_time_ms: int = 0,
) -> dict:
    """
    Save a completed scan. Atomic transaction: scan insert + patient update + audit log.

    result dict structure:
        prediction        "Normal" | "Pneumonia"
        confidence        float 0-1
        ensemble_prob     float 0-1
        severity          "None" | "Mild" | "Moderate" | "Severe"
        subtype           "N/A" | "Bacterial" | "Viral"

    model_votes dict structure (one key per model, value = pneumonia probability float):
        densenet121, resnet50, efficientnet_b4, vit_b16,
        mobilenetv3, inception_v3, attention_cnn
    """
    db = get_db()

    scan_id = _generate_id("SCN")
    now = _now()

    scan_doc = {
        "scan_id": scan_id,
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "scan_date": now,
        "image_path": image_path,
        "heatmap_path": heatmap_path,
        "report_path": report_path,
        # nested result object
        "result": {
            "prediction": result.get("prediction", "Unknown"),
            "confidence": float(result.get("confidence", 0.0)),
            "ensemble_prob": float(result.get("ensemble_prob", 0.0)),
            "severity": result.get("severity", "None"),
            "subtype": result.get("subtype", "N/A"),
        },
        # nested model votes object
        "model_votes": {
            "densenet121": float(model_votes.get("densenet121", 0.0)),
            "resnet50": float(model_votes.get("resnet50", 0.0)),
            "efficientnet_b4": float(model_votes.get("efficientnet_b4", 0.0)),
            "vit_b16": float(model_votes.get("vit_b16", 0.0)),
            "mobilenetv3": float(model_votes.get("mobilenetv3", 0.0)),
            "inception_v3": float(model_votes.get("inception_v3", 0.0)),
            "attention_cnn": float(model_votes.get("attention_cnn", 0.0)),
        },
        "doctor_notes": doctor_notes.strip(),
        "processing_time_ms": int(processing_time_ms),
        "created_at": now,
    }

    diagnosis = result.get("prediction", "Unknown")

    with _client.start_session() as session:
        with session.start_transaction():
            db.scans.insert_one(scan_doc, session=session)

            # update patient summary
            db.patients.update_one(
                {"patient_id": patient_id},
                {
                    "$inc": {"total_scans": 1},
                    "$set": {
                        "last_scan_date": now,
                        "last_diagnosis": diagnosis,
                    },
                },
                session=session,
            )

            # update doctor scan count
            db.doctors.update_one(
                {"doctor_id": doctor_id},
                {"$inc": {"total_scans": 1}},
                session=session,
            )

            _write_audit(
                action="scan_created",
                doctor_id=doctor_id,
                patient_id=patient_id,
                scan_id=scan_id,
                details={
                    "prediction": diagnosis,
                    "confidence": result.get("confidence", 0.0),
                    "severity": result.get("severity", "None"),
                },
                session=session,
            )

    return {k: v for k, v in scan_doc.items() if k != "_id"}


def get_scans_for_patient(patient_id: str, limit: int = 100) -> list:
    """Return all active scans for a patient, newest first."""
    db = get_db()
    cursor = (
        db.scans.find({"patient_id": patient_id, "is_active": {"$ne": False}})
        .sort("scan_date", DESCENDING)
        .limit(limit)
    )
    return [{k: v for k, v in doc.items() if k != "_id"} for doc in cursor]


def get_scan_by_id(scan_id: str) -> Optional[dict]:
    db = get_db()
    record = db.scans.find_one({"scan_id": scan_id})
    return {k: v for k, v in record.items() if k != "_id"} if record else None


def update_scan_notes(scan_id: str, doctor_notes: str, doctor_id: str) -> bool:
    db = get_db()
    with _client.start_session() as session:
        with session.start_transaction():
            result = db.scans.update_one(
                {"scan_id": scan_id},
                {"$set": {"doctor_notes": doctor_notes.strip()}},
                session=session,
            )
            _write_audit(
                action="scan_notes_updated",
                doctor_id=doctor_id,
                scan_id=scan_id,
                details={},
                session=session,
            )
    return result.modified_count > 0


def update_scan_report_path(scan_id: str, report_path: str) -> bool:
    db = get_db()
    result = db.scans.update_one(
        {"scan_id": scan_id},
        {"$set": {"report_path": report_path}},
    )
    return result.modified_count > 0


def deactivate_scan(scan_id: str, doctor_id: str) -> bool:
    """Soft-delete a scan by setting is_active: False. Data is preserved."""
    db = get_db()
    with _client.start_session() as session:
        with session.start_transaction():
            result = db.scans.update_one(
                {"scan_id": scan_id},
                {"$set": {"is_active": False}},
                session=session,
            )
            _write_audit(
                action="scan_deactivated",
                doctor_id=doctor_id,
                scan_id=scan_id,
                details={},
                session=session,
            )
    return result.modified_count > 0


# ---------------------------------------------------------------------------
# Dashboard aggregations
# ---------------------------------------------------------------------------

def get_dashboard_stats(doctor_id: str) -> dict:
    """
    Returns aggregated stats for the dashboard.
    Uses MongoDB aggregation pipeline.
    """
    db = get_db()

    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    # total patients
    total_patients = db.patients.count_documents(
        {"assigned_doctor_id": doctor_id}
    )

    # scans today
    scans_today = db.scans.count_documents(
        {"doctor_id": doctor_id, "scan_date": {"$gte": today_start}}
    )

    # pneumonia rate (all time, this doctor)
    pipeline = [
        {"$match": {"doctor_id": doctor_id}},
        {
            "$group": {
                "_id": None,
                "total": {"$sum": 1},
                "pneumonia": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$result.prediction", "Pneumonia"]},
                            1,
                            0,
                        ]
                    }
                },
            }
        },
    ]
    agg = list(db.scans.aggregate(pipeline))
    if agg:
        total_scans = agg[0]["total"]
        pneumonia_count = agg[0]["pneumonia"]
        pneumonia_rate = round((pneumonia_count / total_scans) * 100, 1) if total_scans else 0.0
    else:
        total_scans = 0
        pneumonia_rate = 0.0

    # reports generated = scans with non-empty report_path
    reports_generated = db.scans.count_documents(
        {"doctor_id": doctor_id, "report_path": {"$ne": ""}}
    )

    return {
        "total_patients": total_patients,
        "scans_today": scans_today,
        "total_scans": total_scans,
        "pneumonia_rate": pneumonia_rate,
        "reports_generated": reports_generated,
    }


def get_recent_activity(doctor_id: str, limit: int = 5) -> list:
    """
    Returns last N scans with patient name joined.
    Each entry: scan_id, patient_id, patient_name, prediction,
                confidence, severity, scan_date, heatmap_path
    """
    db = get_db()

    pipeline = [
        {"$match": {"doctor_id": doctor_id}},
        {"$sort": {"scan_date": -1}},
        {"$limit": limit},
        {
            "$lookup": {
                "from": "patients",
                "localField": "patient_id",
                "foreignField": "patient_id",
                "as": "patient_info",
            }
        },
        {"$unwind": {"path": "$patient_info", "preserveNullAndEmptyArrays": True}},
        {
            "$project": {
                "_id": 0,
                "scan_id": 1,
                "patient_id": 1,
                "patient_name": {"$ifNull": ["$patient_info.name", "Unknown"]},
                "prediction": "$result.prediction",
                "confidence": "$result.confidence",
                "severity": "$result.severity",
                "scan_date": 1,
                "heatmap_path": 1,
            }
        },
    ]

    return list(db.scans.aggregate(pipeline))


# ---------------------------------------------------------------------------
# Audit log (internal, always called from within transactions)
# ---------------------------------------------------------------------------

def _write_audit(
    action: str,
    doctor_id: str = "",
    patient_id: str = "",
    scan_id: str = "",
    details: Optional[dict] = None,
    session=None,
):
    """
    Write one audit log entry. Pass session= when inside a transaction.
    Called internally only. Never called directly by GUI.
    """
    db = get_db()
    entry = {
        "timestamp": _now(),
        "action": action,
        "doctor_id": doctor_id,
        "patient_id": patient_id,
        "scan_id": scan_id,
        "details": details or {},
    }
    if session:
        db.audit_log.insert_one(entry, session=session)
    else:
        db.audit_log.insert_one(entry)


def get_audit_log(doctor_id: Optional[str] = None, limit: int = 100) -> list:
    """Retrieve recent audit entries. Useful for admin or debugging."""
    db = get_db()
    filt = {"doctor_id": doctor_id} if doctor_id else {}
    cursor = db.audit_log.find(filt, {"_id": 0}).sort("timestamp", DESCENDING).limit(limit)
    return list(cursor)
