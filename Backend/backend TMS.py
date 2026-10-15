import os
import json
import asyncio
import base64
import logging
import uuid
import io
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, constr, field_validator
from typing import Optional, List, Dict, Any
import re
import psycopg2, psycopg2.extras, hashlib, secrets
try:
    from PIL import Image, ImageFilter
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

from auth import (
    create_access_token,
    get_current_user,
    require_patient,
    require_doctor,
    require_patient_or_doctor,
)
from crypto_utils import encrypt_field, decrypt_field
from ml_model import (
    LABEL_CLASSES, explain_prediction, predict_distribution, qa_engine, validate_evidences,
)
import extraction
import llm
import triage

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
    evidences: List[str] = Field(..., min_length=1, max_length=300)
    absent: List[str] = Field(default_factory=list, max_length=300)  # explicitly denied
    age: Optional[int] = Field(default=None, ge=0, le=120)          # None = not stated
    sex: Optional[str] = Field(default=None, pattern="^[MF]$")
    intake_id: Optional[int] = Field(default=None, gt=0)            # from /extract_symptoms
    diagnose_setup: str = "Nothing"

class ExtractSymptomsRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000)

class NextQuestionsRequest(BaseModel):
    evidences: List[str] = Field(..., min_length=1, max_length=300)
    absent: List[str] = Field(default_factory=list, max_length=300)
    age: Optional[int] = Field(default=None, ge=0, le=120)
    sex: Optional[str] = Field(default=None, pattern="^[MF]$")
    n: int = Field(default=3, ge=1, le=5)
    round_num: int = Field(default=1, ge=1, le=3)

class ExplainRequest(BaseModel):
    day_id: int = Field(..., gt=0)
    lang: str = Field(default="ru", pattern="^(ru|en|kk)$")

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
    status: str = ""

class LabResultResponse(BaseModel):
    result_id: int
    test_type: str
    test_date: str
    results: List[LabResultItem]
    interpretation: str

class GetLabResultsRequest(BaseModel):
    patient_id: int = Field(..., gt=0)

# ─── Data & Config ───

# model_dict removed — predictions now use sklearn RandomForest via ml_model.py

disease_recommendations = {
    "Acute COPD exacerbation / infection": "Бронхолитики (сальбутамол), системные кортикостероиды, антибиотики при инфекции. Срочная консультация врача.",
    "Acute dystonic reactions": "Немедленная отмена вызвавшего препарата, антихолинергические средства (дифенгидрамин). Срочно к врачу.",
    "Acute laryngitis": "Голосовой покой, увлажнение воздуха, тёплое питьё, при необходимости — противовоспалительные.",
    "Acute otitis media": "Жаропонижающее, обезболивающее. Консультация ЛОР-врача, возможна антибиотикотерапия.",
    "Acute pulmonary edema": "СКОРАЯ ПОМОЩЬ НЕМЕДЛЕННО. Положение сидя, кислород, диуретики по назначению врача.",
    "Acute rhinosinusitis": "Промывание носа солевым раствором, сосудосуживающие при необходимости. Консультация ЛОР при затяжном течении.",
    "Allergic sinusitis": "Антигистаминные препараты, назальные кортикостероиды, избегать аллергенов.",
    "Anaphylaxis": "СКОРАЯ ПОМОЩЬ НЕМЕДЛЕННО. Адреналин (эпинефрин) внутримышечно, антигистаминные, кортикостероиды.",
    "Anemia": "Препараты железа / витамин B12 по назначению врача. Консультация гематолога для выяснения причины.",
    "Atrial fibrillation": "СРОЧНАЯ консультация кардиолога. Антиаритмические препараты, контроль частоты сердечных сокращений.",
    "Boerhaave": "СКОРАЯ ПОМОЩЬ НЕМЕДЛЕННО. Хирургическое вмешательство. Разрыв пищевода — жизнеугрожающее состояние.",
    "Bronchiectasis": "Дренажный массаж, муколитики, антибиотики при обострении. Наблюдение пульмонолога.",
    "Bronchiolitis": "Обильное питьё, промывание носа, контроль дыхания. При одышке — срочно к врачу.",
    "Bronchitis": "Покой, обильное питьё, муколитики. При высокой температуре или затяжном течении — консультация врача.",
    "Bronchospasm / acute asthma exacerbation": "Ингаляции сальбутамола немедленно. При тяжёлом приступе — СКОРАЯ ПОМОЩЬ.",
    "Chagas": "Антипаразитарные препараты (бензнидазол/нифуртимокс). Консультация инфекциониста обязательна.",
    "Chronic rhinosinusitis": "Назальные кортикостероиды, промывание носа, консультация ЛОР-врача.",
    "Cluster headache": "Кислородная терапия, триптаны. Консультация невролога для подбора профилактического лечения.",
    "Croup": "Увлажнённый прохладный воздух, ингаляции физраствором. При выраженном лающем кашле — консультация врача.",
    "Ebola": "СКОРАЯ ПОМОЩЬ НЕМЕДЛЕННО. Изоляция, симптоматическое лечение в специализированном стационаре.",
    "Epiglottitis": "СКОРАЯ ПОМОЩЬ НЕМЕДЛЕННО. Жизнеугрожающее состояние. Госпитализация, антибиотики.",
    "GERD": "Изменение образа жизни (диета, позиционная терапия), ингибиторы протонной помпы. Консультация гастроэнтеролога.",
    "Guillain-Barré syndrome": "СРОЧНАЯ госпитализация. Иммунотерапия (ВВИГ или плазмаферез), наблюдение невролога.",
    "HIV (initial infection)": "СРОЧНАЯ консультация инфекциониста. Антиретровирусная терапия, мониторинг иммунного статуса.",
    "Influenza": "Покой, обильное питьё, жаропонижающее (парацетамол/ибупрофен). При тяжёлом течении — осельтамивир.",
    "Inguinal hernia": "Консультация хирурга. При ущемлении — СКОРАЯ ПОМОЩЬ НЕМЕДЛЕННО.",
    "Larygospasm": "Успокоить пациента, дышать в пакет. При повторных эпизодах — консультация ЛОР.",
    "Localized edema": "Выяснить причину (травма, аллергия, венозная недостаточность). Консультация врача.",
    "Myasthenia gravis": "Ингибиторы холинэстеразы, консультация невролога. При миастеническом кризе — СКОРАЯ ПОМОЩЬ.",
    "Myocarditis": "СРОЧНАЯ госпитализация. Постельный режим, противовоспалительное лечение под контролем кардиолога.",
    "PSVT": "Вагусные пробы (Вальсальва). При неэффективности — СРОЧНАЯ консультация кардиолога / СКОРАЯ.",
    "Pancreatic neoplasm": "СРОЧНАЯ консультация онколога и гастроэнтеролога. Дообследование (КТ, МРТ, биопсия).",
    "Panic attack": "Дыхательные упражнения, успокоительные. Консультация психиатра/психолога для профилактики.",
    "Pericarditis": "Противовоспалительные (НПВП, колхицин), покой. Консультация кардиолога обязательна.",
    "Pneumonia": "Жаропонижающее при температуре, муколитики. Обязательный осмотр врача (часто требуется антибиотик).",
    "Possible NSTEMI / STEMI": "СКОРАЯ ПОМОЩЬ НЕМЕДЛЕННО. Аспирин, нитроглицерин сублингвально. Инфаркт миокарда.",
    "Pulmonary embolism": "СКОРАЯ ПОМОЩЬ НЕМЕДЛЕННО. Антикоагулянты, тромболизис по показаниям. Жизнеугрожающее состояние.",
    "Pulmonary neoplasm": "СРОЧНАЯ консультация пульмонолога и онколога. Дообследование (КТ грудной клетки, биопсия).",
    "SLE": "Консультация ревматолога. Гидроксихлорохин, кортикостероиды по назначению. Защита от солнца.",
    "Sarcoidosis": "Консультация пульмонолога. Кортикостероиды при симптоматическом течении.",
    "Scombroid food poisoning": "Антигистаминные препараты, обильное питьё. Симптомы обычно проходят за 6-12 часов.",
    "Spontaneous pneumothorax": "СКОРАЯ ПОМОЩЬ НЕМЕДЛЕННО. Дренирование плевральной полости в условиях стационара.",
    "Spontaneous rib fracture": "Обезболивание, консультация травматолога. Исключить патологические причины (остеопороз, метастазы).",
    "Stable angina": "Нитраты при приступе, антиагреганты, бета-блокаторы. Наблюдение кардиолога.",
    "Tuberculosis": "СРОЧНАЯ консультация фтизиатра. Длительная противотуберкулёзная терапия (минимум 6 месяцев).",
    "URTI": "Покой, тёплое питьё, промывание носа, жаропонижающее при необходимости.",
    "Unstable angina": "СКОРАЯ ПОМОЩЬ НЕМЕДЛЕННО. Аспирин, нитроглицерин. Требует срочной госпитализации.",
    "Viral pharyngitis": "Полоскание горла, тёплое питьё, жаропонижающее при необходимости.",
    "Whooping cough": "Антибиотики (азитромицин) на ранних стадиях. Изоляция пациента. Консультация врача обязательна.",
}

disease_labels = {
    "Acute COPD exacerbation / infection": "Обострение ХОБЛ / инфекция",
    "Acute dystonic reactions": "Острые дистонические реакции",
    "Acute laryngitis": "Острый ларингит",
    "Acute otitis media": "Острый средний отит",
    "Acute pulmonary edema": "Острый отёк лёгких",
    "Acute rhinosinusitis": "Острый риносинусит",
    "Allergic sinusitis": "Аллергический синусит",
    "Anaphylaxis": "Анафилаксия",
    "Anemia": "Анемия",
    "Atrial fibrillation": "Мерцательная аритмия",
    "Boerhaave": "Синдром Бурхаве (разрыв пищевода)",
    "Bronchiectasis": "Бронхоэктатическая болезнь",
    "Bronchiolitis": "Бронхиолит",
    "Bronchitis": "Бронхит",
    "Bronchospasm / acute asthma exacerbation": "Бронхоспазм / Обострение астмы",
    "Chagas": "Болезнь Шагаса",
    "Chronic rhinosinusitis": "Хронический риносинусит",
    "Cluster headache": "Кластерная головная боль",
    "Croup": "Круп (Острый ларинготрахеит)",
    "Ebola": "Лихорадка Эбола",
    "Epiglottitis": "Эпиглоттит",
    "GERD": "ГЭРБ (Гастроэзофагеальный рефлюкс)",
    "Guillain-Barré syndrome": "Синдром Гийена-Барре",
    "HIV (initial infection)": "ВИЧ (первичная инфекция)",
    "Influenza": "Грипп / ОРВИ",
    "Inguinal hernia": "Паховая грыжа",
    "Larygospasm": "Ларингоспазм",
    "Localized edema": "Локализованный отёк",
    "Myasthenia gravis": "Миастения гравис",
    "Myocarditis": "Миокардит",
    "PSVT": "Пароксизмальная суправентрикулярная тахикардия",
    "Pancreatic neoplasm": "Новообразование поджелудочной железы",
    "Panic attack": "Паническая атака",
    "Pericarditis": "Перикардит",
    "Pneumonia": "Пневмония",
    "Possible NSTEMI / STEMI": "Возможный инфаркт миокарда (НСТМИ/СТМИ)",
    "Pulmonary embolism": "Тромбоэмболия лёгочной артерии (ТЭЛА)",
    "Pulmonary neoplasm": "Новообразование лёгкого",
    "SLE": "Системная красная волчанка",
    "Sarcoidosis": "Саркоидоз",
    "Scombroid food poisoning": "Скомброидное пищевое отравление",
    "Spontaneous pneumothorax": "Спонтанный пневмоторакс",
    "Spontaneous rib fracture": "Спонтанный перелом ребра",
    "Stable angina": "Стабильная стенокардия",
    "Tuberculosis": "Туберкулёз",
    "URTI": "ОРВИ (инфекция верхних дыхательных путей)",
    "Unstable angina": "Нестабильная стенокардия",
    "Viral pharyngitis": "Вирусный фарингит",
    "Whooping cough": "Коклюш",
}

disease_doctors = {
    "Acute COPD exacerbation / infection": "Пульмонолог / Терапевт",
    "Acute dystonic reactions": "Невролог / Скорая помощь",
    "Acute laryngitis": "ЛОР-врач",
    "Acute otitis media": "ЛОР-врач / Педиатр",
    "Acute pulmonary edema": "СКОРАЯ ПОМОЩЬ (103)",
    "Acute rhinosinusitis": "ЛОР-врач",
    "Allergic sinusitis": "Аллерголог / ЛОР-врач",
    "Anaphylaxis": "СКОРАЯ ПОМОЩЬ (103)",
    "Anemia": "Гематолог / Терапевт",
    "Atrial fibrillation": "Кардиолог",
    "Boerhaave": "СКОРАЯ ПОМОЩЬ (103) / Хирург",
    "Bronchiectasis": "Пульмонолог",
    "Bronchiolitis": "Педиатр / Пульмонолог",
    "Bronchitis": "Терапевт",
    "Bronchospasm / acute asthma exacerbation": "Аллерголог / СКОРАЯ при тяжёлом приступе",
    "Chagas": "Инфекционист",
    "Chronic rhinosinusitis": "ЛОР-врач",
    "Cluster headache": "Невролог",
    "Croup": "Педиатр / СКОРАЯ (если задыхается)",
    "Ebola": "СКОРАЯ ПОМОЩЬ (103) / Инфекционист",
    "Epiglottitis": "СКОРАЯ ПОМОЩЬ (103) / ЛОР-врач",
    "GERD": "Гастроэнтеролог",
    "Guillain-Barré syndrome": "Невролог / СКОРАЯ ПОМОЩЬ",
    "HIV (initial infection)": "Инфекционист",
    "Influenza": "Терапевт / Педиатр",
    "Inguinal hernia": "Хирург",
    "Larygospasm": "ЛОР-врач",
    "Localized edema": "Терапевт / Хирург",
    "Myasthenia gravis": "Невролог",
    "Myocarditis": "Кардиолог / СКОРАЯ ПОМОЩЬ",
    "PSVT": "Кардиолог",
    "Pancreatic neoplasm": "Онколог / Гастроэнтеролог",
    "Panic attack": "Психиатр / Психолог",
    "Pericarditis": "Кардиолог",
    "Pneumonia": "Терапевт / Пульмонолог",
    "Possible NSTEMI / STEMI": "СКОРАЯ ПОМОЩЬ (103)",
    "Pulmonary embolism": "СКОРАЯ ПОМОЩЬ (103)",
    "Pulmonary neoplasm": "Онколог / Пульмонолог",
    "SLE": "Ревматолог",
    "Sarcoidosis": "Пульмонолог / Ревматолог",
    "Scombroid food poisoning": "Терапевт / Аллерголог",
    "Spontaneous pneumothorax": "СКОРАЯ ПОМОЩЬ (103) / Хирург",
    "Spontaneous rib fracture": "Травматолог",
    "Stable angina": "Кардиолог",
    "Tuberculosis": "Фтизиатр",
    "URTI": "Терапевт / Педиатр",
    "Unstable angina": "СКОРАЯ ПОМОЩЬ (103) / Кардиолог",
    "Viral pharyngitis": "Терапевт / ЛОР-врач",
    "Whooping cough": "Инфекционист / Педиатр",
}

DATABASE_URL = os.environ.get("DATABASE_URL", "")

symptom_list =[
    'ABDOMINAL_PAIN','CHEST_PAIN','COUGH','DEHYDRATION','DIARRHEA','FEVER','HEADACHE','ITCHING',
    'MUSCLE_ACHES','NAUSEA','NECK_STIFFNESS','PHOTOPHOBIA','POLYDIPSIA','POLYURIA','RASH',
    'RESPIRATORY_DISTRESS','RUNNY_NOSE','SNEEZING','SORE_THROAT','STRIDOR','VOMITING',
    'WEIGHT_LOSS','WHEEZING'
]

# ─── Image PII Protection ───

def blur_pii_region(image_bytes: bytes, top_fraction: float = 0.18) -> bytes:
    if not HAS_PILLOW:
        logger.warning("Pillow not installed — skipping PII blur")
        return image_bytes
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
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    return conn

def create_tables():
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            doctor_id SERIAL PRIMARY KEY,
            full_name TEXT NOT NULL,
            specialty TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS patients (
            patient_id SERIAL PRIMARY KEY,
            full_name TEXT NOT NULL,
            city TEXT,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS diary_days (
            day_id SERIAL PRIMARY KEY,
            patient_id INTEGER NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
            doctor_id INTEGER REFERENCES doctors(doctor_id) ON DELETE SET NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            disease_predict TEXT,
            score REAL,
            disease_setup TEXT,
            recept TEXT,
            patient_explanation TEXT,
            doctor_explanation TEXT
        );

        CREATE TABLE IF NOT EXISTS diary_symptoms (
            day_id INTEGER NOT NULL REFERENCES diary_days(day_id) ON DELETE CASCADE,
            symptom_code TEXT NOT NULL,
            value INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(day_id, symptom_code)
        );

        CREATE TABLE IF NOT EXISTS lab_results (
            result_id SERIAL PRIMARY KEY,
            patient_id INTEGER NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
            day_id INTEGER REFERENCES diary_days(day_id) ON DELETE SET NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            test_type TEXT,
            test_date TEXT,
            results_json TEXT,
            interpretation TEXT,
            image_filename TEXT
        );

        -- v2 pipeline: one intake per free-text complaint (text + extraction +
        -- safety review), linked to the diary day it produced.
        CREATE TABLE IF NOT EXISTS intakes (
            intake_id SERIAL PRIMARY KEY,
            patient_id INTEGER NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
            created_at TIMESTAMP DEFAULT NOW(),
            text TEXT,
            result_json TEXT
        );
        ALTER TABLE diary_days ADD COLUMN IF NOT EXISTS zone TEXT;
        ALTER TABLE diary_days ADD COLUMN IF NOT EXISTS analysis_json TEXT;
        ALTER TABLE diary_days ADD COLUMN IF NOT EXISTS intake_id INTEGER
            REFERENCES intakes(intake_id) ON DELETE SET NULL;

        CREATE INDEX IF NOT EXISTS idx_days_patient ON diary_days(patient_id, day_id);
        CREATE INDEX IF NOT EXISTS idx_days_doctor  ON diary_days(doctor_id, day_id);
        CREATE INDEX IF NOT EXISTS idx_sym_code     ON diary_symptoms(symptom_code);
        CREATE INDEX IF NOT EXISTS idx_lab_patient  ON lab_results(patient_id);
        CREATE INDEX IF NOT EXISTS idx_intake_patient ON intakes(patient_id);
        """)
        conn.commit()
    finally:
        conn.close()

def seed_doctors():
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM doctors")
        if cur.fetchone()[0] > 0:
            conn.close()
            return

        seeds_raw = os.environ.get("DOCTOR_SEEDS", "")
        if not seeds_raw:
            logger.warning("DOCTOR_SEEDS env variable not set — no doctors seeded")
            conn.close()
            return

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
                VALUES (%s,%s,%s)
            """, (encrypt_field(full_name), encrypt_field(spec), hash_password(pwd)))
        conn.commit()
        logger.info("Doctors seeded successfully")
    finally:
        conn.close()

# ─── Service Functions ───

def register_as_patient(full_name: str, city: str, password_5: str) -> int:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO patients(full_name, city, password_hash)
            VALUES (%s,%s,%s) RETURNING patient_id
        """, (encrypt_field(full_name), encrypt_field(city), hash_password(password_5)))
        patient_id = cur.fetchone()[0]
        conn.commit()
        return patient_id
    finally:
        conn.close()

def log_in_patient(patient_id: int, password_5: str) -> bool:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM patients WHERE patient_id = %s", (patient_id,))
        row = cur.fetchone()
        if not row:
            return False
        return verify_password(password_5, row[0])
    finally:
        conn.close()

def log_in_doctor(doctor_id: int, password_5: str) -> bool:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM doctors WHERE doctor_id = %s", (doctor_id,))
        row = cur.fetchone()
        if not row:
            return False
        return verify_password(password_5, row[0])
    finally:
        conn.close()

def get_doctor_info(doctor_id: int) -> Optional[Dict[str, Any]]:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT doctor_id, full_name, specialty, created_at FROM doctors WHERE doctor_id = %s", (doctor_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {"doctor_id": row[0], "full_name": decrypt_field(row[1]), "specialty": decrypt_field(row[2]), "created_at": str(row[3])}
    finally:
        conn.close()

def select_patient(patient_id: int) -> Optional[Dict[str, Any]]:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT patient_id, full_name, city, created_at
            FROM patients
            WHERE patient_id = %s
        """, (patient_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {"patient_id": row[0], "full_name": decrypt_field(row[1]), "city": decrypt_field(row[2]), "created_at": str(row[3])}
    finally:
        conn.close()

def last_day(patient_id: int) -> Optional[int]:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT MAX(day_id) FROM diary_days WHERE patient_id = %s", (patient_id,))
        return cur.fetchone()[0]
    finally:
        conn.close()

def insert_disease(
    patient_id: int,
    evidences: List[str],
    disease_predict: str,
    score: float,
    disease_setup: Optional[str] = None,
    doctor_id: Optional[int] = None,
    recept: Optional[str] = None,
    patient_explanation: Optional[str] = None,
    doctor_explanation: Optional[str] = None,
    absent: Optional[List[str]] = None,
    zone: Optional[str] = None,
    analysis: Optional[dict] = None,
    intake_id: Optional[int] = None,
) -> int:
    """disease_predict is the top-1 disease name and score its probability
    (v1 stored the space-joined top-3 names and their average)."""

    if recept is None:
        recept = disease_recommendations.get(
            disease_predict,
            "Консультация врача обязательна."
        )

    conn = connect()
    try:
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
                doctor_explanation,
                zone,
                analysis_json,
                intake_id
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING day_id
        """, (
            patient_id,
            doctor_id,
            encrypt_field(disease_predict),
            float(score),
            encrypt_field(disease_setup),
            encrypt_field(recept),
            encrypt_field(patient_explanation),
            encrypt_field(doctor_explanation),
            zone,
            encrypt_field(json.dumps(analysis, ensure_ascii=False)) if analysis is not None else None,
            intake_id,
        ))

        day_id = cur.fetchone()[0]

        # One row per evidence token: value=1 reported present, value=0 denied
        rows = [(day_id, ev, 1) for ev in evidences] + [(day_id, ev, 0) for ev in (absent or [])]
        if rows:
            cur.executemany("""
                INSERT INTO diary_symptoms(day_id, symptom_code, value)
                VALUES (%s,%s,%s)
                ON CONFLICT (day_id, symptom_code) DO NOTHING
            """, rows)

        conn.commit()
        return day_id
    finally:
        conn.close()


def insert_intake(patient_id: int, text: str, result: dict) -> int:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO intakes(patient_id, text, result_json) VALUES (%s,%s,%s) RETURNING intake_id",
            (patient_id, encrypt_field(text), encrypt_field(json.dumps(result, ensure_ascii=False))),
        )
        intake_id = cur.fetchone()[0]
        conn.commit()
        return intake_id
    finally:
        conn.close()


def get_intake(intake_id: int, patient_id: int) -> Optional[Dict[str, Any]]:
    """The intake if it belongs to this patient, else None."""
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT text, result_json FROM intakes WHERE intake_id = %s AND patient_id = %s",
            (intake_id, patient_id),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"text": decrypt_field(row[0]) or "", "result": json.loads(decrypt_field(row[1]) or "{}")}
    finally:
        conn.close()


def top1_name(disease_predict: Optional[str]) -> str:
    """Top-1 disease from a stored disease_predict. v1 rows hold the space-joined
    top-3 ("Possible NSTEMI / STEMI URTI Bronchitis"), so splitting on spaces is
    wrong; match the longest known disease name at the start instead."""
    if not disease_predict:
        return ""
    for name in sorted(LABEL_CLASSES, key=len, reverse=True):
        if disease_predict == name or disease_predict.startswith(name + " "):
            return name
    return disease_predict

def update_recept(day_id: int, recept_text: str) -> None:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE diary_days SET recept = %s WHERE day_id = %s", (encrypt_field(recept_text), day_id))
        conn.commit()
    finally:
        conn.close()

def update_recept_last_day(patient_id: int, recept_text: str) -> bool:
    d = last_day(patient_id)
    if d is None:
        return False
    update_recept(d, recept_text)
    return True

def list_doctors_db() -> List[Dict[str, Any]]:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT doctor_id, full_name, specialty, created_at
            FROM doctors
            ORDER BY doctor_id
        """)
        rows = cur.fetchall()
        return [
            {"doctor_id": r[0], "full_name": decrypt_field(r[1]), "specialty": decrypt_field(r[2]), "created_at": str(r[3])}
            for r in rows
        ]
    finally:
        conn.close()

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
        set_parts.append("recept = %s")
        params.append(encrypt_field(new_recept))

    if new_disease_setup is not None:
        set_parts.append("disease_setup = %s")
        params.append(encrypt_field(new_disease_setup))

    if new_doctor_id is not None:
        set_parts.append("doctor_id = %s")
        params.append(int(new_doctor_id))

    set_sql = ", ".join(set_parts)

    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(f"""
            UPDATE diary_days
            SET {set_sql}
            WHERE day_id = %s AND patient_id = %s
        """, (*params, int(day_id), int(patient_id)))

        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()

def get_patient_history(patient_id: int, limit: int = 30) -> List[Dict[str, Any]]:
    if limit <= 0:
        limit = 30

    conn = connect()
    try:
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
                d.doctor_explanation,
                d.zone
            FROM diary_days d
            LEFT JOIN doctors doc ON doc.doctor_id = d.doctor_id
            WHERE d.patient_id = %s
            ORDER BY d.day_id DESC
            LIMIT %s
        """, (patient_id, limit))
        rows = cur.fetchall()

        return [{
            "day_id": r[0],
            "created_at": str(r[1]),
            "disease_predict": top1_name(decrypt_field(r[2])),
            "score": r[3],
            "disease_setup": decrypt_field(r[4]),
            "recept": decrypt_field(r[5]),
            "doctor_id": r[6],
            "doctor_name": decrypt_field(r[7]),
            "patient_explanation": decrypt_field(r[8]),
            "doctor_explanation": decrypt_field(r[9]),
            "zone": r[10],
        } for r in rows]
    finally:
        conn.close()

def get_symptom_graph(patient_id: int, symptom_code: str) -> List[Dict[str, Any]]:
    if symptom_code not in symptom_list:
        raise ValueError("Неверный symptom_code (нет в symptom_list)")

    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT d.day_id, d.created_at, s.value
            FROM diary_days d
            JOIN diary_symptoms s ON s.day_id = d.day_id
            WHERE d.patient_id = %s AND s.symptom_code = %s
            ORDER BY d.day_id
        """, (patient_id, symptom_code))
        rows = cur.fetchall()

        return [{"day_id": r[0], "created_at": str(r[1]), "value": r[2]} for r in rows]
    finally:
        conn.close()

# Triage lives in triage.py. Zones are computed once in /analys and stored with
# the diary day, so the doctor's dashboard shows exactly what the patient saw.

def get_all_patients_triage() -> list:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT p.patient_id, p.full_name, p.city, p.created_at,
                   d.disease_predict, d.score, d.created_at as diag_date,
                   d.day_id, d.zone
            FROM patients p
            LEFT JOIN diary_days d ON d.day_id = (
                SELECT MAX(d2.day_id) FROM diary_days d2 WHERE d2.patient_id = p.patient_id
            )
            ORDER BY p.patient_id
        """)
        rows = cur.fetchall()
        result = []
        for r in rows:
            disease = top1_name(decrypt_field(r[4]))
            score = r[5] or 0.0
            day_id = r[7]
            zone = r[8]

            if zone is None:
                # v1 row: no stored zone. Re-assess from the top-1 disease and the
                # reported symptoms (v1 stored the top-3 average as score, so it
                # can't serve as a probability).
                evidences = []
                if day_id:
                    cur.execute(
                        "SELECT symptom_code FROM diary_symptoms WHERE day_id = %s AND value > 0",
                        (day_id,),
                    )
                    evidences = [row[0] for row in cur.fetchall()]
                zone = triage.assess({disease: 1.0} if disease else {}, evidences).zone

            result.append({
                "patient_id": r[0],
                "full_name": decrypt_field(r[1]),
                "city": decrypt_field(r[2]),
                "created_at": str(r[3]),
                "last_disease": disease,
                "last_score": score,
                "diag_date": str(r[6]) if r[6] else None,
                "zone": zone,
            })
        return result
    finally:
        conn.close()

def day_belongs_to_patient(day_id: int, patient_id: int) -> bool:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM diary_days WHERE day_id = %s AND patient_id = %s", (day_id, patient_id))
        return cur.fetchone() is not None
    finally:
        conn.close()

def get_day_for_explanation(day_id: int, patient_id: int) -> Optional[Dict[str, Any]]:
    """Everything /explain needs about one of the patient's diary days."""
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT d.zone, d.analysis_json, i.text
            FROM diary_days d
            LEFT JOIN intakes i ON i.intake_id = d.intake_id
            WHERE d.day_id = %s AND d.patient_id = %s
        """, (day_id, patient_id))
        row = cur.fetchone()
        if not row:
            return None
        cur.execute("SELECT symptom_code, value FROM diary_symptoms WHERE day_id = %s", (day_id,))
        symptoms = cur.fetchall()
        return {
            "zone": row[0],
            "analysis": json.loads(decrypt_field(row[1]) or "{}") if row[1] else {},
            "text": decrypt_field(row[2]) or "",
            "evidences": [s[0] for s in symptoms if s[1] > 0],
            "absent": [s[0] for s in symptoms if s[1] == 0],
        }
    finally:
        conn.close()

def save_explanations(day_id: int, patient_id: int, patient_explanation: str, doctor_explanation: str) -> None:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE diary_days SET patient_explanation = %s, doctor_explanation = %s
            WHERE day_id = %s AND patient_id = %s
        """, (encrypt_field(patient_explanation), encrypt_field(doctor_explanation), day_id, patient_id))
        conn.commit()
    finally:
        conn.close()

# ─── Startup ───

create_tables()
seed_doctors()

app = FastAPI(
    title="TMS API",
    description="Therapist Machine Support — AI-assisted symptom analysis and triage system",
    version="1.0.0",
)

# Rate limiter
limiter = Limiter(key_func=get_real_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# CORS
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Для хакатона лучше оставить "*", чтобы не было проблем с фронтом
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ─── Health Check ───

@app.get("/health")
async def health():
    try:
        conn = connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
        finally:
            conn.close()
        return {"status": "healthy", "version": "1.0.0"}
    except Exception:
        return JSONResponse(status_code=503, content={"status": "unhealthy"})

@app.get("/")
async def root():
    return {"message": "TMS API is running"}

@app.get("/healthz")
async def healthz():
    from ml_model import _clf, N_FEATURES, LABEL_CLASSES, _MODEL_PATH
    import os
    model_file_exists = os.path.exists(_MODEL_PATH)
    return {
        "model_loaded": _clf is not None,
        "model_path": str(_MODEL_PATH),
        "model_file_exists": model_file_exists,
        "n_features": N_FEATURES,
        "n_classes": len(LABEL_CLASSES),
    }

# ─── Auth endpoints (public, rate-limited) ───

@app.post("/regist_as_patient", response_model=RegisterResponse)
@limiter.limit("10/minute")
async def regist_as_patient(request: Request, body: RegisterRequest):
    patient_id = register_as_patient(body.name, body.city, body.password)
    logger.info("Patient %d registered", patient_id)
    token = create_access_token({"role": "patient", "patient_id": patient_id})
    return RegisterResponse(patient_id=patient_id, access_token=token)

@app.post("/login_patient", response_model=LoginResponse)
@limiter.limit("5/minute")
async def login_patient_endpoint(request: Request, body: LoginPatientRequest):
    flag = log_in_patient(body.patient_id, body.password)
    if not flag:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    logger.info("Patient %d logged in", body.patient_id)
    token = create_access_token({"role": "patient", "patient_id": body.patient_id})
    return LoginResponse(login=True, access_token=token)

@app.post("/login_doctor", response_model=DoctorLoginResponse)
@limiter.limit("5/minute")
async def login_doctor_endpoint(request: Request, body: LoginDoctorRequest):
    flag = log_in_doctor(body.doctor_id, body.password)
    if not flag:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    logger.info("Doctor %d logged in", body.doctor_id)
    info = get_doctor_info(body.doctor_id)
    token = create_access_token({"role": "doctor", "doctor_id": body.doctor_id})
    return DoctorLoginResponse(login=True, access_token=token, doctor=info)

@app.post("/list_doctor")
@limiter.limit("20/minute")
async def list_doctor_endpoint(request: Request):
    docs = list_doctors_db()
    return {"doctors": docs}

# ─── Patient endpoints (require patient or doctor token) ───

@app.post("/get_history")
async def get_history_endpoint(body: HistoryRequest, user: dict = Depends(require_patient_or_doctor)):
    if user.get("role") == "patient" and user.get("patient_id") != body.patient_id:
        raise HTTPException(status_code=403, detail="Access denied")
    history = get_patient_history(body.patient_id)
    return {"history": history}

@app.post("/get_symptoms")
async def get_symptoms_endpoint(body: SymptomsRequest, user: dict = Depends(require_patient_or_doctor)):
    if user.get("role") == "patient" and user.get("patient_id") != body.patient_id:
        raise HTTPException(status_code=403, detail="Access denied")
    graph = get_symptom_graph(body.patient_id, body.symptom_str)
    return {"symptoms_arr": graph}

def _clean(evidences: List[str], where: str) -> List[str]:
    valid, invalid = validate_evidences(list(dict.fromkeys(evidences)))
    if invalid:
        logger.warning("%s: dropped %d unknown evidence tokens", where, len(invalid))
    return valid


def _recommendation(disease: str, zone: str, review: Optional[dict]) -> str:
    text = disease_recommendations.get(disease, "Консультация врача обязательна.")
    if review and review.get("out_of_scope"):
        likely = review.get("likely_condition")
        text = ("Ваши симптомы могут указывать на состояние, которое система не умеет оценивать"
                + (f" (возможно: {likely})" if likely else "")
                + ". Результат ниже может быть неточным — обязательно обратитесь к врачу. " + text)
    if zone == "red":
        text = "⚠️ СРОЧНО обратитесь к врачу или вызовите скорую помощь! " + text
    return text


@app.post("/analys")
async def analys_endpoint(body: AnalysRequest, user: dict = Depends(require_patient)):
    patient_id = user["patient_id"]
    evidences = _clean(body.evidences, "analys")
    absent = [e for e in _clean(body.absent, "analys") if e not in set(evidences)]
    if not evidences:
        raise HTTPException(status_code=422, detail="No valid evidences")

    review = None
    if body.intake_id is not None:
        intake = get_intake(body.intake_id, patient_id)
        if intake is None:
            raise HTTPException(status_code=403, detail="Access denied")
        review = intake["result"].get("review")

    distribution = predict_distribution(evidences, body.age, body.sex, absent)
    top3 = list(distribution.items())[:3]
    top1, top1_score = top3[0]
    assessed = triage.assess(distribution, evidences, review)
    out_of_scope = bool(review and review.get("out_of_scope"))
    likely_condition = review.get("likely_condition") if out_of_scope else None

    day = insert_disease(
        patient_id,
        evidences,
        top1,
        top1_score,
        body.diagnose_setup,
        absent=absent,
        zone=assessed.zone,
        analysis={
            "top3": [[name, round(p, 4)] for name, p in top3],
            "reasons": assessed.reasons,
            "red_mass": round(assessed.red_mass, 4),
            "yellow_mass": round(assessed.yellow_mass, 4),
            "age": body.age,
            "sex": body.sex,
            "out_of_scope": out_of_scope,
            "likely_condition": likely_condition,
        },
        intake_id=body.intake_id,
    )

    return {
        "day": day,
        "diseaseName": top1,
        "diseaseLabel": disease_labels.get(top1, top1),
        "doctor": disease_doctors.get(top1, "Терапевт"),
        "recommendation": _recommendation(top1, assessed.zone, review),
        "slices": [
            {"name": name, "label": disease_labels.get(name, name), "score": sc}
            for name, sc in top3
        ],
        "score": top1_score,
        "zone": assessed.zone,
        "reasons": assessed.reasons,
        "out_of_scope": {"likely_condition": likely_condition} if out_of_scope else None,
    }


@app.post("/extract_symptoms")
@limiter.limit("20/minute")
async def extract_symptoms_endpoint(
    request: Request,
    body: ExtractSymptomsRequest,
    user: dict = Depends(require_patient),
):
    """
    Free text → evidence tokens, with status (present / absent / uncertain) and
    the patient's own words (Groq) or a confidence (Clef) for each — see
    extraction.extract_complaint. Stores an intake (text, findings, safety
    review) that /analys reads by intake_id.
    """
    result = (await extraction.extract_complaint(body.text)).to_dict()
    intake_id = insert_intake(user["patient_id"], body.text, result)

    review = result.get("review") or {}
    alert = triage.critical_alert(result["evidences"])
    if alert is None and review.get("emergency"):
        alert = {
            "level": "red" if review.get("emergency_verified") else "yellow",
            "message": "⚠️ Ваше описание похоже на неотложное состояние. "
                       "Если вам становится хуже — немедленно вызовите скорую помощь (103).",
        }

    return {
        "intake_id": intake_id,
        "evidences": result["evidences"],
        "absent": result["absent"],
        "uncertain": result["uncertain"],
        "findings": [{**f, "label": extraction.evidence_label(f["code"])} for f in result["findings"]],
        "age": result["age"],
        "sex": result["sex"],
        "noSymptoms": not result["evidences"] and not result["uncertain"],
        "critical_alert": alert,
        "mode": result["mode"],
    }


@app.post("/next_questions")
@limiter.limit("30/minute")
async def next_questions_endpoint(
    request: Request,  # noqa: used by slowapi limiter
    body: NextQuestionsRequest,
    _user: dict = Depends(require_patient),
):
    """
    Up to `n` yes/no follow-up questions, chosen by expected information gain
    over diseases and triage tiers (qa_engine.py). Two rounds; stops early once
    the top diagnosis reaches 70%. Pass every "no" back in `absent`.
    """
    evidences = _clean(body.evidences, "next_questions")
    absent = [e for e in _clean(body.absent, "next_questions") if e not in set(evidences)]
    if not evidences:
        raise HTTPException(status_code=422, detail="No valid evidences")

    result = qa_engine.get_questions(
        evidences=evidences,
        age=body.age,
        sex=body.sex,
        round_num=body.round_num,
        n_questions=body.n,
        absent=absent,
    )
    distribution = predict_distribution(evidences, body.age, body.sex, absent)

    return {
        "questions": [
            {
                "evidence_id": q.evidence_id,
                "question_en": q.question_en,
                "question_ru": q.question_ru,
                "question_kk": q.question_kk,
                "score": q.score,
                "reason": q.reason,
            }
            for q in result.questions
        ],
        "current_top3": [
            {"disease": name, "label": disease_labels.get(name, name), "score": round(sc, 3)}
            for name, sc in result.top3
        ],
        "zone": triage.assess(distribution, evidences).zone,
        "critical_alert": triage.critical_alert(evidences),
        "should_stop": result.should_stop,
        "stop_reason": result.stop_reason,
        "confidence": round(result.confidence, 4),
        "round_type": result.round_type,
    }


_LANG_NAMES = {"ru": "Russian", "en": "English", "kk": "Kazakh"}

_FALLBACK_EXPLANATION = {
    "ru": ("Анализ симптомов завершён. Это предварительная оценка — обратитесь к врачу.",
           "Автоматическое обоснование недоступно."),
    "en": ("Symptom analysis complete. This is a preliminary assessment — please see a doctor.",
           "Automatic clinical reasoning is unavailable."),
    "kk": ("Симптомдар талдауы аяқталды. Бұл алдын ала баға — дәрігерге жүгініңіз.",
           "Автоматты клиникалық негіздеме қол жетімсіз."),
}

_PATIENT_EXPLAIN_PROMPT = """\
You explain a preliminary triage result to a PATIENT. Write 2-4 short sentences in {lang}, plain words, calm tone.
- Say which of the patient's symptoms point to the result. Use ONLY symptoms listed under "Reported" or "Denied"; never add others.
- Say it is a preliminary assessment and a doctor must confirm it. If the zone is red, tell them to seek urgent care now.
- No drug names or doses. No JSON, no markdown. Entirely in {lang}."""

_DOCTOR_EXPLAIN_PROMPT = """\
You are a clinical decision-support assistant writing for a DOCTOR, in {lang}. Use markdown with these sections:
**Clinical reasoning** · **Differential** · **Key findings and red flags** · **Suggested work-up** · **Watch for**
Rules:
- Base the reasoning on the facts given. "Influence" values are measured effects inside the model (how much each answer moved each diagnosis), not clinical likelihood ratios; you may cite them as such.
- Mention only the reported and denied findings listed; do not invent symptoms, vitals or history.
- If the case may be outside the model's 49 conditions, say so first.
- End with one line stating this is decision support, not a diagnosis. 8-15 sentences total."""


def _explanation_facts(day: Dict[str, Any]) -> str:
    analysis = day["analysis"]
    top3 = analysis.get("top3") or []
    attributions = explain_prediction(
        day["evidences"], analysis.get("age"), analysis.get("sex"), day["absent"],
        diseases=[name for name, _ in top3],
    )

    def label(token: str) -> str:
        if token.startswith("NOT "):
            return "denies: " + extraction.evidence_label(token[4:])["en"]
        return extraction.evidence_label(token)["en"]

    lines = []
    if day["text"]:
        lines.append(f'Patient\'s own words: "{day["text"]}"')
    lines.append("Reported: " + ("; ".join(label(e) for e in day["evidences"]) or "—"))
    lines.append("Denied: " + ("; ".join(label(e) for e in day["absent"]) or "—"))
    lines.append("Model differential (probability, then the answers that moved it most):")
    for name, p in top3:
        influence = ", ".join(f"{label(ev)} {'+' if d >= 0 else '−'}{abs(d):.0%}" for ev, d in attributions.get(name, []))
        lines.append(f"- {name}: {p:.0%}. Influence: {influence or '—'}")
    reasons = ", ".join(r["code"] + (f" {r['evidence']}" if "evidence" in r else "") for r in analysis.get("reasons", []))
    lines.append(f"Triage zone: {day['zone']}" + (f" (because: {reasons})" if reasons else ""))
    if analysis.get("out_of_scope"):
        lines.append("Safety reviewer: the complaint may be outside the model's 49 conditions"
                     + (f"; most likely {analysis.get('likely_condition')}" if analysis.get("likely_condition") else ""))
    return "\n".join(lines)


@app.post("/explain")
@limiter.limit("20/minute")
async def explain_endpoint(request: Request, body: ExplainRequest, user: dict = Depends(require_patient)):
    """
    Patient and doctor explanations for one diary day, generated on the server
    and saved with it. The LLM verbalises measured facts (leave-one-out effects
    of each answer on the model) instead of inventing symptom weights.
    Replaces /save_explanation, which stored whatever text the client sent —
    letting a patient write the "AI clinical analysis" their doctor reads.
    """
    patient_id = user["patient_id"]
    day = get_day_for_explanation(body.day_id, patient_id)
    if day is None:
        raise HTTPException(status_code=403, detail="Access denied")

    lang = _LANG_NAMES[body.lang]
    facts = _explanation_facts(day)
    patient_result, doctor_result = await asyncio.gather(
        llm.chat_text(
            [{"role": "system", "content": _PATIENT_EXPLAIN_PROMPT.format(lang=lang)},
             {"role": "user", "content": facts}],
            name="explain_patient", reasoning="low", max_tokens=1500,
        ),
        llm.chat_text(
            [{"role": "system", "content": _DOCTOR_EXPLAIN_PROMPT.format(lang=lang)},
             {"role": "user", "content": facts}],
            name="explain_doctor", reasoning="medium", max_tokens=4000,
        ),
    )
    fallback_patient, fallback_doctor = _FALLBACK_EXPLANATION[body.lang]
    patient_text = patient_result.data if patient_result else fallback_patient
    doctor_text = doctor_result.data if doctor_result else fallback_doctor
    save_explanations(body.day_id, patient_id, patient_text, doctor_text)
    return {"patientExplanation": patient_text, "doctorExplanation": doctor_text}

# ─── Doctor-only endpoints ───

@app.post("/update_by_doctor")
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

@app.post("/list_patients_triage")
async def list_patients_triage_endpoint(user: dict = Depends(require_doctor)):
    patients = get_all_patients_triage()
    return {"patients": patients}

@app.post("/get_patient_info")
async def get_patient_info_endpoint(body: PatientInfoRequest, user: dict = Depends(require_patient_or_doctor)):
    if user.get("role") == "patient" and user.get("patient_id") != body.patient_id:
        raise HTTPException(status_code=403, detail="Access denied")
    info = select_patient(body.patient_id)
    if info is None:
        return {"found": False, "patient": None}
    return {"found": True, "patient": info}

@app.post("/get_day_symptoms")
async def get_day_symptoms_endpoint(body: DaySymptomsRequest, user: dict = Depends(require_patient_or_doctor)):
    if user.get("role") == "patient":
        if not day_belongs_to_patient(body.day_id, user["patient_id"]):
            raise HTTPException(status_code=403, detail="Access denied")
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT symptom_code, value
            FROM diary_symptoms
            WHERE day_id = %s
            ORDER BY symptom_code
        """, (body.day_id,))
        rows = cur.fetchall()
        symptoms = {r[0]: r[1] for r in rows}
    finally:
        conn.close()
    # value: 1 reported, 0 denied. labels: {en, ru, kk} per evidence token.
    return {"symptoms": symptoms, "labels": {code: extraction.evidence_label(code) for code in symptoms}}

# ─── Lab Results endpoints ───

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

@app.post("/upload_lab_result")
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

    image_bytes = blur_pii_region(raw_bytes)

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = image.content_type

    safe_filename = f"{uuid.uuid4().hex}.jpg"

    if not llm.available():
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured on server")

    system_prompt = (
        "Ты медицинский OCR-ассистент. Извлеки данные из фотографии медицинского анализа.\n"
        "КРИТИЧЕСКОЕ ПРАВИЛО БЕЗОПАСНОСТИ: Полностью игнорируй любые персональные данные на изображении.\n"
        "Верни ТОЛЬКО валидный JSON без markdown:\n"
        "{\n"
        '  "test_type": "Тип анализа",\n'
        '  "test_date": "YYYY-MM-DD или пусто",\n'
        '  "interpretation": "Краткая интерпретация отклонений",\n'
        '  "results":[\n'
        '    {"name": "Показатель", "value": "значение", "unit": "ед.", "reference_range": "норма", "status": "normal|high|low"}\n'
        "  ]\n"
        "}\n"
    )

    # Groq documents JSON mode, not strict schemas, for image input (strict=False);
    # reasoning off — this is transcription, not judgement.
    result = await llm.chat_json(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                {"type": "text", "text": "Извлеки все данные из этого анализа."},
            ]},
        ],
        {}, name="lab_ocr", models=llm.VISION_MODELS, strict=False, reasoning="none",
        temperature=0.2, max_tokens=4096, timeout=60,
    )
    if result is None:
        raise HTTPException(status_code=502, detail="AI service unavailable")
    parsed = result.data if isinstance(result.data, dict) else {}

    if "error" in parsed:
        raise HTTPException(status_code=422, detail=str(parsed["error"]))

    test_type = str(parsed.get("test_type") or "Неизвестный анализ")
    test_date = str(parsed.get("test_date") or "")
    interpretation = str(parsed.get("interpretation") or "")
    results = [
        {key: str(item.get(key) or "") for key in ("name", "value", "unit", "reference_range", "status")}
        for item in (parsed.get("results") or []) if isinstance(item, dict)
    ]

    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO lab_results(patient_id, test_type, test_date, results_json, interpretation, image_filename)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING result_id
        """, (patient_id, test_type, test_date, encrypt_field(json.dumps(results, ensure_ascii=False)), encrypt_field(interpretation), safe_filename))
        result_id = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()

    return {
        "result_id": result_id,
        "test_type": test_type,
        "test_date": test_date,
        "results": results,
        "interpretation": interpretation,
    }


@app.post("/get_lab_results")
async def get_lab_results_endpoint(body: GetLabResultsRequest = None, user: dict = Depends(require_patient_or_doctor)):
    if user.get("role") == "patient":
        patient_id = user["patient_id"]
    else:
        if not body or not body.patient_id:
            raise HTTPException(status_code=400, detail="patient_id required for doctors")
        patient_id = body.patient_id

    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT result_id, test_type, test_date, results_json, interpretation, created_at
            FROM lab_results
            WHERE patient_id = %s
            ORDER BY result_id DESC
            LIMIT 50
        """, (patient_id,))
        rows = cur.fetchall()
    finally:
        conn.close()

    results = []
    for r in rows:
        decrypted_json = decrypt_field(r[3])
        try:
            items = json.loads(decrypted_json) if decrypted_json else[]
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