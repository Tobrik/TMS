import os
import json
import base64
import logging
import uuid
import io
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, constr, field_validator
from typing import Optional, List, Dict, Any
import re
import sqlite3, hashlib, secrets
from PIL import Image, ImageFilter
import httpx

from auth import (
    create_access_token,
    get_current_user,
    require_patient,
    require_doctor,
    require_patient_or_doctor,
)
from crypto_utils import encrypt_field, decrypt_field

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

BEHIND_PROXY = os.environ.get("BEHIND_PROXY", "").lower() in ("1", "true", "yes")

def get_real_ip(request: Request) -> str:
    """Extract real client IP. Only trusts X-Forwarded-For when BEHIND_PROXY=true
    (i.e. Nginx/Cloudflare forcibly overwrites this header with real IP)."""
    if BEHIND_PROXY:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return get_remote_address(request)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tms")

# ─── Pydantic Request/Response Models ───

class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    city: str = Field("", max_length=100)
    password: str = Field(..., min_length=8, max_length=64)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not re.search(r'[A-Za-z]', v):
            raise ValueError("Пароль должен содержать хотя бы одну букву")
        if not re.search(r'\d', v):
            raise ValueError("Пароль должен содержать хотя бы одну цифру")
        return v

class RegisterResponse(BaseModel):
    patient_id: int
    access_token: str
    token_type: str = "bearer"

# Login accepts any password (backwards compat with old 5-digit PINs)
class LoginPatientRequest(BaseModel):
    patient_id: int = Field(..., gt=0)
    password: constr(min_length=1, max_length=64)

class LoginDoctorRequest(BaseModel):
    doctor_id: int = Field(..., gt=0)
    password: constr(min_length=1, max_length=64)

class LoginResponse(BaseModel):
    login: bool
    access_token: str
    token_type: str = "bearer"

class DoctorInfo(BaseModel):
    doctor_id: int
    full_name: str
    specialty: str
    created_at: Optional[str] = None

class DoctorLoginResponse(LoginResponse):
    doctor: Optional[DoctorInfo] = None

class HistoryRequest(BaseModel):
    patient_id: int = Field(..., gt=0)

class SymptomsRequest(BaseModel):
    patient_id: int = Field(..., gt=0)
    symptom_str: str = Field(..., min_length=1)

class AnalysRequest(BaseModel):
    symptoms: List[int] = Field(..., min_length=23, max_length=23)
    diagnose_setup: str = "Nothing"

class SaveExplanationRequest(BaseModel):
    day_id: int = Field(..., gt=0)
    patient_explanation: str = ""
    doctor_explanation: str = ""

class UpdateByDoctorRequest(BaseModel):
    patient_id: int = Field(..., gt=0)
    day_id: int = Field(..., gt=0)
    doctor_id: int = Field(..., gt=0)
    recept: Optional[str] = None
    disease_setup: Optional[str] = None

class PatientInfoRequest(BaseModel):
    patient_id: int = Field(..., gt=0)

class DaySymptomsRequest(BaseModel):
    day_id: int = Field(..., gt=0)

class LabResultItem(BaseModel):
    name: str
    value: str
    unit: str = ""
    reference_range: str = ""
    status: str = ""  # "normal", "high", "low"

class LabResultResponse(BaseModel):
    result_id: int
    test_type: str
    test_date: str
    results: List[LabResultItem]
    interpretation: str

class GetLabResultsRequest(BaseModel):
    patient_id: int = Field(..., gt=0)

# ─── Data & Config ───

model_dict = {
    "Gastroenteritis": ([0.0, 0.0, 0.0, 2.835492374437445, 15.358088877898277, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2150830135431683, 0.0, 0.0], 43.76321546894878),
    "Croup": ([0.0, 0.0, 11.445743094869425, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.87292660938857, 0.0, 0.0, 0.0, 11.52701610240259, 0.0, 0.0, 0.0], 32.031654663581826),
    "Scarlet Fever": ([0.0, 0.0, 0.0, 0.0, 0.0, 9.116404717554731, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.159872877966544, 0.0, 0.0, 0.0, 8.910748251927256, 0.0, 0.0, 0.0, 0.0], 23.817103444179114),
    "Eczema": ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.088135680193556, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.230981001282307, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 32.27669511833457),
    "Asthma": ([0.0, 0.0, 4.013460044880404, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.148492399703434, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.56929617979114], 42.23353377651567),
    "Type 1 Diabetes": ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.226327290578648, 4.3550777255458515, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.999718366968764, 0.0], 69.86173564127515),
    "Bronchiolitis": ([0.0, 0.0, 11.82227219023071, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2277262934489515, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.708398608051098], 30.368790732206605),
    "Meningitis": ([0.0, 0.0, 0.0, 0.0, 0.0, 8.995565067705446, 8.546131902138402, 0.0, 0.0, 0.0, 8.714651354612204, 2.2869737170763695, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 21.466857330755033),
    "Influenza": ([0.0, 0.0, 9.59320387475004, 0.0, 0.0, 8.640868953054095, 1.7856157416078982, 0.0, 8.959693076647547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 22.474301999408752),
    "Pneumonia": ([0.0, 2.187194382475494, 10.2438779340937, 0.0, 0.0, 9.201196328535795, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.879709650994538, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 22.039462525010897),
    "Chickenpox": ([0.0, 0.0, 0.0, 0.0, 0.0, 3.1380009798357853, 0.0, 12.145343392428853, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.88079352135918, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 30.971991926994647),
    "Appendicitis": ([14.733187425381898, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.657776271847671, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5721136493562833, 0.0, 0.0], 44.026175264172544),
    "Common Cold": ([0.0, 0.0, 3.321662031504646, 0.0, 0.0, 1.9510386014496788, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.608440261670545, 3.248914037624695, 0.0, 0.0, 0.0, 0.0, 0.0], 42.32083725409407),
}

disease_recommendations = {
    "Gastroenteritis": "Регидратация (оральные растворы типа Регидрон), дробное питьё, диета. При боли – спазмолитик по возрасту.",
    "Croup": "Увлажнённый прохладный воздух, ингаляции физраствором. При выраженном лающем кашле – консультация врача.",
    "Scarlet Fever": "Жаропонижающее (парацетамол/ибупрофен по возрасту), обильное питьё. Обязателен осмотр врача (часто требуется антибиотик).",
    "Eczema": "Увлажняющие кремы (эмоленты), антигистаминное при зуде, избегать аллергенов.",
    "Asthma": "Ингаляции короткодействующим бронхолитиком (сальбутамол), контроль дыхания, избегать триггеров.",
    "Type 1 Diabetes": "Контроль глюкозы, инсулинотерапия по назначению врача. Срочная консультация эндокринолога.",
    "Bronchiolitis": "Обильное питьё, промывание носа, контроль дыхания. При одышке – срочно к врачу.",
    "Meningitis": "Срочная госпитализация. Неотложное обращение за медицинской помощью.",
    "Influenza": "Покой, обильное питьё, жаропонижающее (парацетамол/ибупрофен), при кашле – ACC/муколитик по возрасту.",
    "Pneumonia": "Жаропонижающее при температуре, муколитики (ACC), обязательный осмотр врача (часто требуется антибиотик).",
    "Chickenpox": "Обработка сыпи антисептиком, антигистаминное при зуде, жаропонижающее при температуре.",
    "Appendicitis": "Срочно к хирургу. Не давать обезболивающие до осмотра врача.",
    "Common Cold": "Покой, тёплое питьё, промывание носа, жаропонижающее при необходимости."
}

DB = os.environ.get("DB_PATH", "/tmp/medical.db")

symptom_list = [
    'ABDOMINAL_PAIN','CHEST_PAIN','COUGH','DEHYDRATION','DIARRHEA','FEVER','HEADACHE','ITCHING',
    'MUSCLE_ACHES','NAUSEA','NECK_STIFFNESS','PHOTOPHOBIA','POLYDIPSIA','POLYURIA','RASH',
    'RESPIRATORY_DISTRESS','RUNNY_NOSE','SNEEZING','SORE_THROAT','STRIDOR','VOMITING',
    'WEIGHT_LOSS','WHEEZING'
]

# ─── Image PII Protection ───

def blur_pii_region(image_bytes: bytes, top_fraction: float = 0.18) -> bytes:
    """Blur the top portion of a medical document image where PII (name, address, etc.) is typically located."""
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    crop_h = int(h * top_fraction)
    if crop_h < 10:
        return image_bytes
    top_region = img.crop((0, 0, w, crop_h))
    blurred = top_region.filter(ImageFilter.GaussianBlur(radius=30))
    img.paste(blurred, (0, 0))
    buf = io.BytesIO()
    fmt = img.format or "JPEG"
    img.save(buf, format=fmt, quality=90)
    return buf.getvalue()

# ─── Password Hashing ───

def hash_password(pwd: str) -> str:
    if not isinstance(pwd, str) or len(pwd) < 1:
        raise ValueError("Пароль не может быть пустым")
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pwd.encode("utf-8"), salt, 600_000)
    return salt.hex() + ":" + dk.hex()

def verify_password(pwd: str, stored: str) -> bool:
    try:
        salt_hex, dk_hex = stored.split(":")
        salt = bytes.fromhex(salt_hex)
        dk = bytes.fromhex(dk_hex)
        test = hashlib.pbkdf2_hmac("sha256", pwd.encode("utf-8"), salt, 600_000)
        return secrets.compare_digest(test, dk)
    except Exception:
        return False

# ─── Database ───

def connect():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def create_tables():
    with connect() as conn:
        cur = conn.cursor()
        cur.executescript("""
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS doctors (
            doctor_id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            specialty TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS patients (
            patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            city TEXT,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS diary_days (
            day_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            doctor_id INTEGER,
            created_at TEXT DEFAULT (datetime('now')),

            disease_predict TEXT,
            score REAL,
            disease_setup TEXT,
            recept TEXT,
            patient_explanation TEXT,
            doctor_explanation TEXT,

            FOREIGN KEY(patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
            FOREIGN KEY(doctor_id) REFERENCES doctors(doctor_id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS diary_symptoms (
            day_id INTEGER NOT NULL,
            symptom_code TEXT NOT NULL,
            value INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(day_id, symptom_code),
            FOREIGN KEY(day_id) REFERENCES diary_days(day_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS lab_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            day_id INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            test_type TEXT,
            test_date TEXT,
            results_json TEXT,
            interpretation TEXT,
            image_filename TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
            FOREIGN KEY(day_id) REFERENCES diary_days(day_id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_days_patient ON diary_days(patient_id, day_id);
        CREATE INDEX IF NOT EXISTS idx_days_doctor  ON diary_days(doctor_id, day_id);
        CREATE INDEX IF NOT EXISTS idx_sym_code     ON diary_symptoms(symptom_code);
        CREATE INDEX IF NOT EXISTS idx_lab_patient  ON lab_results(patient_id);
        """)
        conn.commit()

        cur.execute("PRAGMA table_info(diary_days)")
        cols = {row[1] for row in cur.fetchall()}
        if "patient_explanation" not in cols:
            cur.execute("ALTER TABLE diary_days ADD COLUMN patient_explanation TEXT")
        if "doctor_explanation" not in cols:
            cur.execute("ALTER TABLE diary_days ADD COLUMN doctor_explanation TEXT")
        conn.commit()

def seed_doctors():
    """Seed doctors from DOCTOR_SEEDS env var. Format: name|specialty|pin;name|specialty|pin"""
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM doctors")
        if cur.fetchone()[0] > 0:
            return

    seeds_raw = os.environ.get("DOCTOR_SEEDS", "")
    if not seeds_raw:
        logger.warning("DOCTOR_SEEDS env variable not set — no doctors seeded")
        return

    with connect() as conn:
        cur = conn.cursor()
        for entry in seeds_raw.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split("|")
            if len(parts) != 3:
                logger.warning("Skipping malformed DOCTOR_SEEDS entry: %s", entry)
                continue
            full_name, spec, pwd = parts[0].strip(), parts[1].strip(), parts[2].strip()
            cur.execute("""
                INSERT INTO doctors(full_name, specialty, password_hash)
                VALUES (?,?,?)
            """, (encrypt_field(full_name), encrypt_field(spec), hash_password(pwd)))
        conn.commit()
        logger.info("Doctors seeded successfully")

# ─── Service Functions ───

def register_as_patient(full_name: str, city: str, password_5: str) -> int:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO patients(full_name, city, password_hash)
            VALUES (?,?,?)
        """, (encrypt_field(full_name), encrypt_field(city), hash_password(password_5)))
        conn.commit()
        return cur.lastrowid

def log_in_patient(patient_id: int, password_5: str) -> bool:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM patients WHERE patient_id = ?", (patient_id,))
        row = cur.fetchone()
        if not row:
            return False
        return verify_password(password_5, row[0])

def log_in_doctor(doctor_id: int, password_5: str) -> bool:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM doctors WHERE doctor_id = ?", (doctor_id,))
        row = cur.fetchone()
        if not row:
            return False
        return verify_password(password_5, row[0])

def get_doctor_info(doctor_id: int) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT doctor_id, full_name, specialty, created_at FROM doctors WHERE doctor_id = ?", (doctor_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {"doctor_id": row[0], "full_name": decrypt_field(row[1]), "specialty": decrypt_field(row[2]), "created_at": row[3]}

def select_patient(patient_id: int) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT patient_id, full_name, city, created_at
            FROM patients
            WHERE patient_id = ?
        """, (patient_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {"patient_id": row[0], "full_name": decrypt_field(row[1]), "city": decrypt_field(row[2]), "created_at": row[3]}

def last_day(patient_id: int) -> Optional[int]:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT MAX(day_id) FROM diary_days WHERE patient_id = ?", (patient_id,))
        return cur.fetchone()[0]

def insert_disease(
    patient_id: int,
    symptoms_23: List[int],
    disease_predict: str,
    score: float,
    disease_setup: Optional[str] = None,
    doctor_id: Optional[int] = None,
    recept: Optional[str] = None,
    patient_explanation: Optional[str] = None,
    doctor_explanation: Optional[str] = None
) -> int:

    if len(symptoms_23) != len(symptom_list):
        raise ValueError(f"symptoms_23 должен быть длиной {len(symptom_list)}")

    if recept is None:
        recept = disease_recommendations.get(
            disease_predict,
            "Консультация врача обязательна."
        )

    with connect() as conn:
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO diary_days(
                patient_id,
                doctor_id,
                disease_predict,
                score,
                disease_setup,
                recept,
                patient_explanation,
                doctor_explanation
            )
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            patient_id,
            doctor_id,
            encrypt_field(disease_predict),
            float(score),
            encrypt_field(disease_setup),
            encrypt_field(recept),
            encrypt_field(patient_explanation),
            encrypt_field(doctor_explanation),
        ))

        day_id = cur.lastrowid

        rows = [
            (day_id, symptom_list[i], int(symptoms_23[i]))
            for i in range(len(symptom_list))
        ]

        cur.executemany("""
            INSERT INTO diary_symptoms(day_id, symptom_code, value)
            VALUES (?,?,?)
        """, rows)

        conn.commit()
        return day_id

def update_recept(day_id: int, recept_text: str) -> None:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE diary_days SET recept = ? WHERE day_id = ?", (encrypt_field(recept_text), day_id))
        conn.commit()

def update_recept_last_day(patient_id: int, recept_text: str) -> bool:
    d = last_day(patient_id)
    if d is None:
        return False
    update_recept(d, recept_text)
    return True

def list_doctors_db() -> List[Dict[str, Any]]:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT doctor_id, full_name, specialty, created_at
            FROM doctors
            ORDER BY doctor_id
        """)
        rows = cur.fetchall()
        return [
            {"doctor_id": r[0], "full_name": decrypt_field(r[1]), "specialty": decrypt_field(r[2]), "created_at": r[3]}
            for r in rows
        ]

def doctor_update_day(
    patient_id: int,
    day_id: int,
    doctor_id: int,
    new_recept: Optional[str] = None,
    new_disease_setup: Optional[str] = None,
    new_doctor_id: Optional[int] = None
) -> bool:
    if new_recept is None and new_disease_setup is None and new_doctor_id is None:
        return False

    set_parts = []
    params = []

    if new_recept is not None:
        set_parts.append("recept = ?")
        params.append(encrypt_field(new_recept))

    if new_disease_setup is not None:
        set_parts.append("disease_setup = ?")
        params.append(encrypt_field(new_disease_setup))

    if new_doctor_id is not None:
        set_parts.append("doctor_id = ?")
        params.append(int(new_doctor_id))

    set_sql = ", ".join(set_parts)

    with connect() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            UPDATE diary_days
            SET {set_sql}
            WHERE day_id = ? AND patient_id = ?
        """, (*params, int(day_id), int(patient_id)))

        conn.commit()
        return cur.rowcount > 0

def get_patient_history(patient_id: int, limit: int = 30) -> List[Dict[str, Any]]:
    if limit <= 0:
        limit = 30

    with connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                d.day_id,
                d.created_at,
                d.disease_predict,
                d.score,
                d.disease_setup,
                d.recept,
                d.doctor_id,
                doc.full_name,
                d.patient_explanation,
                d.doctor_explanation
            FROM diary_days d
            LEFT JOIN doctors doc ON doc.doctor_id = d.doctor_id
            WHERE d.patient_id = ?
            ORDER BY d.day_id DESC
            LIMIT ?
        """, (patient_id, limit))
        rows = cur.fetchall()

        return [{
            "day_id": r[0],
            "created_at": r[1],
            "disease_predict": decrypt_field(r[2]),
            "score": r[3],
            "disease_setup": decrypt_field(r[4]),
            "recept": decrypt_field(r[5]),
            "doctor_id": r[6],
            "doctor_name": r[7],
            "patient_explanation": decrypt_field(r[8]),
            "doctor_explanation": decrypt_field(r[9]),
        } for r in rows]

def get_symptom_graph(patient_id: int, symptom_code: str) -> List[Dict[str, Any]]:
    if symptom_code not in symptom_list:
        raise ValueError("Неверный symptom_code (нет в symptom_list)")

    with connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT d.day_id, d.created_at, s.value
            FROM diary_days d
            JOIN diary_symptoms s ON s.day_id = d.day_id
            WHERE d.patient_id = ? AND s.symptom_code = ?
            ORDER BY d.day_id
        """, (patient_id, symptom_code))
        rows = cur.fetchall()

        return [{"day_id": r[0], "created_at": r[1], "value": r[2]} for r in rows]


def model_predict(symptoms):
    score_dict = dict()
    if not len(symptoms) > len(model_dict["Croup"][0]):
        symptoms = symptoms[:len(model_dict["Croup"][0])]
    for dis,model in model_dict.items():
        f = model[1]
        f_max = model[1]
        relevant = 0
        matched = 0
        for i in range(len(symptoms)):
            f += model[0][i] * symptoms[i]
            f_max += model[0][i] * 3
            if model[0][i] > 0:
                relevant += 1
                if symptoms[i] > 0:
                    matched += 1
        score = f / f_max
        if relevant > 0 and matched == 0:
            score *= 0.1
        score_dict[dis] = score
    pre_diagnose = sorted(score_dict.items(),key = lambda x: x[1] ,reverse = True)[:3]
    return pre_diagnose

RED_ZONE_DISEASES = {"Meningitis", "Appendicitis", "Type 1 Diabetes"}
YELLOW_ZONE_DISEASES = {"Pneumonia", "Scarlet Fever", "Influenza"}

def classify_zone(disease: str, score: float) -> str:
    if disease in RED_ZONE_DISEASES or score > 0.6:
        return "red"
    if disease in YELLOW_ZONE_DISEASES or score > 0.4:
        return "yellow"
    return "green"

def get_all_patients_triage() -> list:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT p.patient_id, p.full_name, p.city, p.created_at,
                   d.disease_predict, d.score, d.created_at as diag_date
            FROM patients p
            LEFT JOIN diary_days d ON d.day_id = (
                SELECT MAX(d2.day_id) FROM diary_days d2 WHERE d2.patient_id = p.patient_id
            )
            ORDER BY p.patient_id
        """)
        rows = cur.fetchall()
        result = []
        for r in rows:
            disease = decrypt_field(r[4]) or ""
            score = r[5] or 0.0
            first_disease = disease.split(" ")[0] if disease else ""
            zone = classify_zone(first_disease, score)
            result.append({
                "patient_id": r[0],
                "full_name": decrypt_field(r[1]),
                "city": decrypt_field(r[2]),
                "created_at": r[3],
                "last_disease": disease,
                "last_score": score,
                "diag_date": r[6],
                "zone": zone,
            })
        return result

def day_belongs_to_patient(day_id: int, patient_id: int) -> bool:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM diary_days WHERE day_id = ? AND patient_id = ?", (day_id, patient_id))
        return cur.fetchone() is not None

# ─── Startup ───

create_tables()
seed_doctors()

app = FastAPI(
    title="TMS API",
    description="Therapist Machine Support — AI-powered pediatric symptom analysis and triage system",
    version="1.0.0",
)

router = APIRouter(prefix="/api")

# Rate limiter
limiter = Limiter(key_func=get_real_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Global exception handler — no stack traces in production
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# CORS — restricted origins
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# ─── Health Check ───

@router.get("/health")
async def health():
    try:
        with connect() as conn:
            conn.execute("SELECT 1")
        return {"status": "healthy", "version": "1.0.0"}
    except Exception:
        return JSONResponse(status_code=503, content={"status": "unhealthy"})

# ─── Auth endpoints (public, rate-limited) ───

@router.post("/regist_as_patient", response_model=RegisterResponse)
@limiter.limit("10/minute")
async def regist_as_patient(request: Request, body: RegisterRequest):
    patient_id = register_as_patient(body.name, body.city, body.password)
    logger.info("Patient %d registered", patient_id)
    token = create_access_token({"role": "patient", "patient_id": patient_id})
    return RegisterResponse(patient_id=patient_id, access_token=token)

@router.post("/login_patient", response_model=LoginResponse)
@limiter.limit("5/minute")
async def login_patient_endpoint(request: Request, body: LoginPatientRequest):
    flag = log_in_patient(body.patient_id, body.password)
    if not flag:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    logger.info("Patient %d logged in", body.patient_id)
    token = create_access_token({"role": "patient", "patient_id": body.patient_id})
    return LoginResponse(login=True, access_token=token)

@router.post("/login_doctor", response_model=DoctorLoginResponse)
@limiter.limit("5/minute")
async def login_doctor_endpoint(request: Request, body: LoginDoctorRequest):
    flag = log_in_doctor(body.doctor_id, body.password)
    if not flag:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    logger.info("Doctor %d logged in", body.doctor_id)
    info = get_doctor_info(body.doctor_id)
    token = create_access_token({"role": "doctor", "doctor_id": body.doctor_id})
    return DoctorLoginResponse(login=True, access_token=token, doctor=info)

@router.post("/list_doctor")
@limiter.limit("20/minute")
async def list_doctor_endpoint(request: Request):
    docs = list_doctors_db()
    return {"doctors": docs}

# ─── Patient endpoints (require patient or doctor token) ───

@router.post("/get_history")
async def get_history_endpoint(body: HistoryRequest, user: dict = Depends(require_patient_or_doctor)):
    if user.get("role") == "patient" and user.get("patient_id") != body.patient_id:
        raise HTTPException(status_code=403, detail="Access denied")
    history = get_patient_history(body.patient_id)
    return {"history": history}

@router.post("/get_symptoms")
async def get_symptoms_endpoint(body: SymptomsRequest, user: dict = Depends(require_patient_or_doctor)):
    if user.get("role") == "patient" and user.get("patient_id") != body.patient_id:
        raise HTTPException(status_code=403, detail="Access denied")
    graph = get_symptom_graph(body.patient_id, body.symptom_str)
    return {"symptoms_arr": graph}

@router.post("/analys")
async def analys_endpoint(body: AnalysRequest, user: dict = Depends(require_patient)):
    patient_id = user["patient_id"]

    top3 = model_predict(body.symptoms)
    score = (top3[0][1]+top3[1][1]+top3[2][1])/3
    preliminary_diagnose = top3[0][0] + " " + top3[1][0] + " " + top3[2][0]
    if score < 0.2 and body.diagnose_setup == "Nothing":
        preliminary_diagnose = "Nothing"

    recept = disease_recommendations.get(top3[0][0], "Nothing")
    day = insert_disease(patient_id, body.symptoms, preliminary_diagnose, score, body.diagnose_setup, None, recept)
    return {
        "day": day,
        "preliminary_diagnose": preliminary_diagnose,
        "recept": recept,
        "score": score,
        "top1": top3[0][0],
        "top2": top3[1][0],
        "top3": top3[2][0],
        "top1_score": top3[0][1],
        "top2_score": top3[1][1],
        "top3_score": top3[2][1],
    }

@router.post("/save_explanation")
async def save_explanation_endpoint(body: SaveExplanationRequest, user: dict = Depends(require_patient)):
    patient_id = user.get("patient_id")
    if not day_belongs_to_patient(body.day_id, patient_id):
        raise HTTPException(status_code=403, detail="Access denied")
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE diary_days
            SET patient_explanation = ?, doctor_explanation = ?
            WHERE day_id = ?
        """, (encrypt_field(body.patient_explanation), encrypt_field(body.doctor_explanation), body.day_id))
        conn.commit()
        return {"ok": cur.rowcount > 0}

# ─── Doctor-only endpoints ───

@router.post("/update_by_doctor")
async def update_by_doctor_endpoint(body: UpdateByDoctorRequest, user: dict = Depends(require_doctor)):
    doctor_id = user.get("doctor_id")
    ok = doctor_update_day(
        body.patient_id,
        body.day_id,
        doctor_id,
        body.recept,
        body.disease_setup,
        doctor_id,
    )
    return {"ok": ok}

@router.post("/list_patients_triage")
async def list_patients_triage_endpoint(user: dict = Depends(require_doctor)):
    patients = get_all_patients_triage()
    return {"patients": patients}

@router.post("/get_patient_info")
async def get_patient_info_endpoint(body: PatientInfoRequest, user: dict = Depends(require_patient_or_doctor)):
    if user.get("role") == "patient" and user.get("patient_id") != body.patient_id:
        raise HTTPException(status_code=403, detail="Access denied")
    info = select_patient(body.patient_id)
    if info is None:
        return {"found": False, "patient": None}
    return {"found": True, "patient": info}

@router.post("/get_day_symptoms")
async def get_day_symptoms_endpoint(body: DaySymptomsRequest, user: dict = Depends(require_patient_or_doctor)):
    if user.get("role") == "patient":
        if not day_belongs_to_patient(body.day_id, user["patient_id"]):
            raise HTTPException(status_code=403, detail="Access denied")
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT symptom_code, value
            FROM diary_symptoms
            WHERE day_id = ?
            ORDER BY symptom_code
        """, (body.day_id,))
        rows = cur.fetchall()
        symptoms = {r[0]: r[1] for r in rows}
    return {"symptoms": symptoms}

# ─── Lab Results endpoints ───

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

@router.post("/upload_lab_result")
async def upload_lab_result_endpoint(
    image: UploadFile = File(...),
    user: dict = Depends(require_patient),
):
    patient_id = user["patient_id"]

    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPG, PNG, or WebP.")

    raw_bytes = await image.read()
    if len(raw_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB)")

    # Blur top 18% of image to remove PII (name, address, clinic header)
    image_bytes = blur_pii_region(raw_bytes)

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = image.content_type

    # Sanitize filename — never trust user-supplied names
    safe_filename = f"{uuid.uuid4().hex}.jpg"

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured on server")

    vision_model = os.environ.get("GROQ_VISION_MODEL", "llama-4-scout-17b-16e-instruct")

    system_prompt = (
        "Ты медицинский OCR-ассистент. Извлеки данные из фотографии медицинского анализа.\n"
        "\n"
        "КРИТИЧЕСКОЕ ПРАВИЛО БЕЗОПАСНОСТИ: Полностью игнорируй любые персональные данные на изображении "
        "(ФИО, адреса, даты рождения, телефоны, номера полисов, названия клиник). "
        "НИКОГДА не включай их в JSON-ответ. Извлекай ТОЛЬКО медицинские показатели из таблицы результатов.\n"
        "\n"
        "Верни ТОЛЬКО валидный JSON без markdown:\n"
        "{\n"
        '  "test_type": "Тип анализа",\n'
        '  "test_date": "YYYY-MM-DD или пусто",\n'
        '  "interpretation": "Краткая интерпретация отклонений",\n'
        '  "results": [\n'
        '    {"name": "Показатель", "value": "значение", "unit": "ед.", "reference_range": "норма", "status": "normal|high|low"}\n'
        "  ]\n"
        "}\n"
        "status: 'normal' если в пределах нормы, 'high' если выше, 'low' если ниже.\n"
        "Если не можешь разобрать изображение — верни: {\"error\": \"описание проблемы\"}"
    )

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={
                    "model": vision_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                            {"type": "text", "text": "Извлеки все данные из этого анализа."},
                        ]},
                    ],
                    "temperature": 0,
                    "max_tokens": 4096,
                },
            )
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error("Groq Vision API error: %s %s", e.response.status_code, e.response.text)
        raise HTTPException(status_code=502, detail="AI service error")
    except Exception as e:
        logger.error("Groq Vision request failed: %s", e)
        raise HTTPException(status_code=502, detail="AI service unavailable")

    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        logger.error("Failed to parse Groq Vision response: %s", content[:500])
        raise HTTPException(status_code=502, detail="AI returned invalid JSON")

    if "error" in parsed:
        raise HTTPException(status_code=422, detail=parsed["error"])

    test_type = parsed.get("test_type", "Неизвестный анализ")
    test_date = parsed.get("test_date", "")
    interpretation = parsed.get("interpretation", "")
    results = parsed.get("results", [])

    with connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO lab_results(patient_id, test_type, test_date, results_json, interpretation, image_filename)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (patient_id, test_type, test_date, encrypt_field(json.dumps(results, ensure_ascii=False)), encrypt_field(interpretation), safe_filename))
        conn.commit()
        result_id = cur.lastrowid

    return {
        "result_id": result_id,
        "test_type": test_type,
        "test_date": test_date,
        "results": results,
        "interpretation": interpretation,
    }


@router.post("/get_lab_results")
async def get_lab_results_endpoint(body: GetLabResultsRequest = None, user: dict = Depends(require_patient_or_doctor)):
    if user.get("role") == "patient":
        patient_id = user["patient_id"]
    else:
        if not body or not body.patient_id:
            raise HTTPException(status_code=400, detail="patient_id required for doctors")
        patient_id = body.patient_id

    with connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT result_id, test_type, test_date, results_json, interpretation, created_at
            FROM lab_results
            WHERE patient_id = ?
            ORDER BY result_id DESC
            LIMIT 50
        """, (patient_id,))
        rows = cur.fetchall()

    results = []
    for r in rows:
        decrypted_json = decrypt_field(r[3])
        try:
            items = json.loads(decrypted_json) if decrypted_json else []
        except json.JSONDecodeError:
            items = []
        results.append({
            "result_id": r[0],
            "test_type": r[1],
            "test_date": r[2],
            "results": items,
            "interpretation": decrypt_field(r[4]),
            "created_at": r[5],
        })

    return {"results": results}

# Include router with /api prefix
app.include_router(router)
