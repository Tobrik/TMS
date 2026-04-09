import os
import json
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
from ml_model import sklearn_predict, validate_evidences, qa_engine

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
    age: int = Field(default=25, ge=0, le=120)
    sex: str = Field(default="M", pattern="^[MF]$")
    diagnose_setup: str = "Nothing"

class ExtractSymptomsRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000)

class NextQuestionsRequest(BaseModel):
    evidences: List[str] = Field(..., min_length=1, max_length=300)
    age: int = Field(default=25, ge=0, le=120)
    sex: str = Field(default="M", pattern="^[MF]$")
    n: int = Field(default=3, ge=1, le=5)
    round_num: int = Field(default=1, ge=1, le=3)

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

        CREATE INDEX IF NOT EXISTS idx_days_patient ON diary_days(patient_id, day_id);
        CREATE INDEX IF NOT EXISTS idx_days_doctor  ON diary_days(doctor_id, day_id);
        CREATE INDEX IF NOT EXISTS idx_sym_code     ON diary_symptoms(symptom_code);
        CREATE INDEX IF NOT EXISTS idx_lab_patient  ON lab_results(patient_id);
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
    doctor_explanation: Optional[str] = None
) -> int:

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
                doctor_explanation
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING day_id
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

        day_id = cur.fetchone()[0]

        # Store each evidence token as a symptom row (value=1 means present)
        if evidences:
            rows = [(day_id, ev, 1) for ev in evidences]
            cur.executemany("""
                INSERT INTO diary_symptoms(day_id, symptom_code, value)
                VALUES (%s,%s,%s)
                ON CONFLICT (day_id, symptom_code) DO NOTHING
            """, rows)

        conn.commit()
        return day_id
    finally:
        conn.close()

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
                d.doctor_explanation
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
            "disease_predict": decrypt_field(r[2]),
            "score": r[3],
            "disease_setup": decrypt_field(r[4]),
            "recept": decrypt_field(r[5]),
            "doctor_id": r[6],
            "doctor_name": r[7],
            "patient_explanation": decrypt_field(r[8]),
            "doctor_explanation": decrypt_field(r[9]),
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

# model_predict() removed — use sklearn_predict() from ml_model.py

RED_ZONE_DISEASES = {
    "Anaphylaxis", "Boerhaave", "Ebola", "Epiglottitis",
    "Acute pulmonary edema", "Possible NSTEMI / STEMI", "Pulmonary embolism",
    "Spontaneous pneumothorax", "Unstable angina",
}
YELLOW_ZONE_DISEASES = {
    "Atrial fibrillation", "Bronchospasm / acute asthma exacerbation",
    "Guillain-Barré syndrome", "HIV (initial infection)", "Myocarditis",
    "Pancreatic neoplasm", "PSVT", "Pneumonia", "Pulmonary neoplasm",
    "Stable angina", "Tuberculosis",
}

# ─── Symptom-based red flags (safety layer) ──────────────────────────────────
# These override the disease-based classification.
# Medical triage depends on SYMPTOMS, not on what the model guessed.
# If ANY of these evidences are present → force that zone regardless of diagnosis.

RED_FLAG_EVIDENCES = {
    # Stridor = airway obstruction → minutes to death
    "E_112",
    # Chest pain at rest → possible ACS
    "E_14",
    # Syncope / loss of consciousness
    # "E_82" is dizziness which is yellow, not red
}

YELLOW_FLAG_EVIDENCES = {
    # Fever (T≥38) — needs medical evaluation
    "E_91",
    # Pleuritic pain (pain on deep breathing) — pneumonia/PE marker
    "E_220",
    # Shortness of breath
    "E_66",
    # Dizziness / near-syncope
    "E_82",
    # Palpitations / tachycardia
    "E_155",
    # Repeated vomiting — dehydration risk
    "E_211",
}

# Combinations that escalate to RED even if individual symptoms are only yellow
RED_FLAG_COMBOS = [
    # Fever + pleuritic pain = pneumonia until proven otherwise
    {"E_91", "E_220"},
    # Fever + shortness of breath = possible pneumonia/sepsis
    {"E_91", "E_66"},
    # Chest pain + sweating = ACS
    {"E_14", "E_50"},
    # Chest pain + radiation to left arm
    {"E_14", "E_57_@_V_195"},
    # Chest pain + radiation to jaw
    {"E_14", "E_57_@_V_121"},
    # Chest pain + nausea = ACS
    {"E_14", "E_148"},
]


def classify_zone(disease: str, score: float, evidences: list = None) -> str:
    """
    Triage classification with symptom-based safety layer.

    Priority order:
      1. Red flag evidences / combos → "red" (regardless of diagnosis)
      2. Yellow flag evidences → at least "yellow"
      3. Disease name in RED_ZONE_DISEASES → "red"
      4. Disease name in YELLOW_ZONE_DISEASES → "yellow"
      5. Score thresholds → fallback
    """
    ev_set = set(evidences) if evidences else set()

    # --- Safety layer: symptom-based flags OVERRIDE model output ---

    # 1) Single red-flag symptoms
    if ev_set & RED_FLAG_EVIDENCES:
        return "red"

    # 2) Red-flag COMBINATIONS (e.g. fever + pleuritic pain)
    for combo in RED_FLAG_COMBOS:
        if combo <= ev_set:  # combo is subset of patient evidences
            return "red"

    # 3) Any yellow-flag symptom → at least yellow
    has_yellow_flag = bool(ev_set & YELLOW_FLAG_EVIDENCES)

    # --- Disease-based classification ---
    if disease in RED_ZONE_DISEASES or score > 0.6:
        return "red"
    if disease in YELLOW_ZONE_DISEASES or score > 0.5 or has_yellow_flag:
        return "yellow"

    return "green"

def get_all_patients_triage() -> list:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT p.patient_id, p.full_name, p.city, p.created_at,
                   d.disease_predict, d.score, d.created_at as diag_date,
                   d.day_id
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
            day_id = r[7]

            # Fetch stored evidences for symptom-based triage
            evidences = []
            if day_id:
                cur.execute(
                    "SELECT symptom_code FROM diary_symptoms WHERE day_id = %s",
                    (day_id,),
                )
                evidences = [row[0] for row in cur.fetchall()]

            first_disease = disease.split(" ")[0] if disease else ""
            zone = classify_zone(first_disease, score, evidences=evidences)
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

# ─── Startup ───

create_tables()
seed_doctors()

# ─── LLM extraction prompt (Groq LLaMA 3.3 70B) ────────────────────────────
_LLM_EXTRACTION_PROMPT = """\
You are a clinical NLP engine for a medical triage system operating in Central Asia (Kazakhstan/CIS).
Your sole task: extract structured diagnostic evidence from a patient's free-text complaint.

## OUTPUT CONTRACT
Return ONLY valid JSON. No markdown, no preamble, no explanation, no code fences.
Schema:
{
  "evidences": ["E_91", "E_55_@_V_89", ...],
  "age": 32,
  "sex": "M",
  "negated": ["E_201"],
  "uncertain": ["E_14"],
  "implicit": ["E_175"],
  "onset_days": 3,
  "severity": "medium",
  "extraction_notes": "brief internal reasoning, 1-2 sentences max"
}

- evidences: confirmed positive findings → DDXPlus binary evidence IDs
- negated: explicitly denied symptoms (DO NOT include in evidences)
- uncertain: mentioned with doubt ("возможно", "кажется", "might have")
- implicit: inferred from context but not stated directly (e.g. "лежу 3 дня" → E_88)
- age: integer or null
- sex: "M", "F", or null
- onset_days: how many days since symptoms started (integer or null)
- severity: "mild" | "medium" | "severe" | null
- extraction_notes: 1 sentence explaining your main extraction decisions

## EVIDENCE ID REFERENCE
Map patient symptoms to these DDXPlus binary evidence IDs:

### Systemic / Fever
E_91 — fever / temperature (температура, қызба)
E_94 — chills / shivers (озноб, қалтырау)
E_50 — increased sweating (потеет, терлеу)
E_175 — fatigue / malaise (слабость, усталость, әлсіздік)
E_88 — extreme fatigue / bedridden (не может встать, лежит весь день)
E_89 — non-restful sleep / insomnia (плохой сон, бессонница)
E_144 — diffuse muscle pain / myalgia (ломота, мышечные боли, бұлшықет ауру)
E_162 — unintentional weight loss (похудел, теряет вес)
E_161 — decreased appetite (нет аппетита, жоқ тәбет)
E_96 — weight gain (набрал вес)

### Pain (requires E_53 = pain present)
E_53 — pain present (есть боль, any pain)
E_55_@_V_89 — pain location: forehead/head (голова, бас)
E_55_@_V_166 — pain location: temple (висок)
E_55_@_V_124 — pain location: occiput (затылок)
E_55_@_V_62 — pain location: top of head (макушка)
E_55_@_V_101 — pain location: upper chest/sternum (грудь, кеуде)
E_55_@_V_55 — pain location: side of chest (бок груди)
E_55_@_V_148 — pain location: pharynx/throat (горло, тамақ)
E_55_@_V_129 — pain location: ear (ухо)
E_55_@_V_187 — pain location: abdomen/belly (живот, іш)
E_55_@_V_87 — pain location: right iliac fossa (правый низ живота)
E_55_@_V_197 — pain location: epigastrium (эпигастрий, под ложечкой)
E_55_@_V_26 — pain location: neck (шея)
E_55_@_V_40 — pain location: lower back (поясница, спина)
E_55_@_V_92 — pain location: knee (колено)
E_55_@_V_194 — pain location: shoulder (плечо)
E_55_@_V_127 — pain location: scapula (лопатка)
E_55_@_V_16 — pain location: groin (пах)
E_57_@_V_195 — pain radiates to left arm/shoulder (отдаёт в левую руку)
E_57_@_V_121 — pain radiates to jaw (отдаёт в челюсть)
E_54_@_V_184 — pain character: pulsating/throbbing (пульсирует)
E_54_@_V_192 — pain character: sharp/stabbing (острая, резкая)
E_54_@_V_179 — pain character: stabbing/knife (колющая)
E_54_@_V_154 — pain character: dull/aching (тупая, ноющая)
E_54_@_V_182 — pain character: cramping/spasm (схваткообразная)
E_54_@_V_183 — pain character: pressing/crushing (давящая, сжимающая)
E_54_@_V_181 — pain character: burning (жжение)

### Respiratory
E_201 — cough (кашель, жөтел)
E_77 — colored/abundant sputum (мокрота, қақырық)
E_181 — nasal congestion / runny nose (насморк, мұрын бітелу)
E_182 — purulent nasal discharge green/yellow (гнойные выделения)
E_66 — shortness of breath / dyspnea (одышка, ентігу)
E_64 — breathless on minimal exertion (задыхается при малейшей нагрузке)
E_67 — nocturnal dyspnea, wakes up breathless (ночная одышка)
E_214 — wheezing on exhale (свист на выдохе, хрипы)
E_112 — stridor / wheezing on inhale (стридор, шум на вдохе)
E_220 — pain on deep breathing / pleuritic (больно при вдохе)
E_45 — hemoptysis / coughing blood (кровь при кашле)
E_203 — severe paroxysmal coughing fits (приступы сильного кашля)
E_97 — sore throat (боль в горле, тамақ ауру)
E_212 — hoarse voice / lost voice (осип, охрип, голос пропал)
E_65 — difficulty swallowing / dysphagia (трудно глотать)

### Cardiovascular
E_14 — chest pain at rest (боль в груди в покое)
E_155 — palpitations / racing heart (сердцебиение, тахикардия)
E_164 — irregular heartbeat / arrhythmia (аритмия, нерегулярный пульс)
E_82 — dizziness / near-syncope (головокружение, бас айналу)
E_159 — loss of consciousness / syncope (потеря сознания)

### Gastrointestinal
E_148 — nausea / vomiting (тошнота, рвота, лоқсу)
E_211 — repeated vomiting >=2x (многократная рвота)
E_51 — diarrhea (понос, диарея)
E_173 — heartburn / acid reflux (изжога)
E_30 — bloating / abdominal distension (вздутие)
E_140 — black/tarry stool (чёрный стул, мелена)
E_179 — blood in stool (кровь в стуле)
E_210 — vomiting blood (рвота с кровью)

### Neurological
E_103 — loss of smell (потеря запаха, иіс сезу жоқ)
E_177 — numbness / tingling (онемение, покалывание)
E_93 — numbness in feet specifically (онемение стоп)
E_84 — bilateral limb weakness (слабость в обеих руках/ногах)
E_156 — facial weakness/paralysis (слабость лица)
E_52 — double vision / diplopia (двоение в глазах)
E_172 — drooping eyelid / ptosis (опущение века)
E_39 — confusion / disorientation (спутанность сознания)
E_43 — seizures / convulsions (судороги, припадок)
E_192 — neck muscle spasm (спазм мышц шеи)
E_193 — facial/body muscle spasms (спазмы мышц лица/тела)
E_168 — can't control tongue (язык вываливается)
E_205 — can't open mouth / trismus (не может открыть рот)
E_111 — fear of dying (страх смерти)
E_171 — depersonalization / derealization (ощущение нереальности)

### Skin / Eyes / ENT
E_129 — rash / skin lesion (сыпь, бөртпе)
E_74 — red eyes / conjunctivitis (красные глаза)
E_9 — swollen lymph nodes (лимфоузлы увеличены)
E_127 — increased tearing / watery eyes (слезятся глаза)
E_154 — pale skin / pallor (бледность)
E_92 — facial flushing / red cheeks (покраснение лица)
E_151 — localized swelling / edema (отёк)
E_178 — unusual bleeding / bruising (необычные синяки, кровотечения)
E_169 — itchy nose / postnasal drip (зуд в носу, аллергия)
E_170 — intense eye itching (сильный зуд в глазах)

### Modifiers / Context
E_220 — pain worse on deep breath (боль при дыхании)
E_217 — symptoms worse lying, better sitting (хуже лёжа, лучше сидя)
E_218 — symptoms worse with exertion (хуже при нагрузке)
E_219 — symptoms worse at night (хуже ночью)
E_216 — pain worse with movement (хуже при движении)
E_221 — worse with coughing/straining (хуже при кашле/натуживании)
E_215 — symptoms worse after eating (хуже после еды)
E_33 — pain better leaning forward (легче наклонившись вперёд)
E_13 — symptoms progressively worsening over 2 weeks (ухудшается 2 недели)
E_41 — contact with sick person (контакт с больным)
E_116 — had cold in past 2 weeks (простыл недавно)
E_99 — history of migraines (мигрень в анамнезе)

### Comorbidities / History
E_105 — history of heart attack / angina (инфаркт/стенокардия в анамнезе)
E_106 — heart failure (сердечная недостаточность)
E_123 — asthma history (астма)
E_124 — uses bronchodilator (использует ингалятор)
E_69 — diabetes (диабет)
E_102 — hypertension (высокое давление)
E_79 — smoker (курит)
E_23 — sleep apnea (апноэ сна)
E_86 — family history of allergy/eczema (аллергия у родственников)

## EXTRACTION RULES

### RULE 1 — NEGATION (critical)
Negated symptoms go to "negated" array only. NEVER put them in "evidences".
Negation patterns: "не [symptom]", "нет [symptom]", "без [symptom]", "[symptom] нет",
"[symptom] жоқ", "[symptom] емес", "no [symptom]", "denies [symptom]", "without [symptom]"

### RULE 2 — PAIN DECOMPOSITION
Any mention of pain REQUIRES at minimum E_53 (pain present).
Add the most specific location code if location is mentioned.
Add character code (E_54_@_V_*) only if explicitly described.
  "болит голова" → [E_53, E_55_@_V_89]
  "острая боль в груди" → [E_53, E_55_@_V_101, E_14, E_54_@_V_192]

### RULE 3 — IMPLICIT SYMPTOMS (add to "implicit" array AND "evidences")
Infer clinically obvious signs from context:
  "не встаю с кровати 3 дня" → add E_88, E_175
  "ничего не ем второй день" → add E_161

### RULE 4 — TEMPORAL EXTRACTION
  "3 дня" / "3 күн" / "3 days" → onset_days: 3
  "неделю" / "апта" / "a week" → onset_days: 7
  "давно" / "уже долго" → onset_days: null

### RULE 5 — SEVERITY ESTIMATION
mild: can function normally. medium: interferes with daily activity.
severe: can't function, emergency markers (E_14 at rest, E_112, E_159, E_210, E_140 → always "severe").

### RULE 6 — VOMITING DISAMBIGUATION
E_148 = nausea OR single vomiting. E_211 = explicitly REPEATED vomiting only.

### RULE 7 — DO NOT INFER CARDIAC WITHOUT EXPLICIT EVIDENCE
NEVER add E_14, E_57_@_V_195, E_57_@_V_121 based on headache or fever alone.

### RULE 8 — DEMOGRAPHIC EXTRACTION
Age: extract from numeric mentions. Sex: "мужчина/ер адам/man" → "M", "женщина/әйел/woman" → "F".

### RULE 9 — KAZAKH LANGUAGE
Detect by: ә, ғ, қ, ң, ө, ұ, ү, һ, і. Apply same extraction logic.

### RULE 10 — UNCERTAINTY / HEDGING
"кажется", "возможно", "наверное", "может быть", "мүмкін" → put in "uncertain", NOT in "evidences".

## ABSOLUTE PROHIBITIONS
- Never add E_14 without explicit mention of chest pain
- Never add E_57_@_V_195 based on chest pain alone
- Never add E_112 based on wheezing alone
- Never fabricate symptoms not present in text
- Never add cardiac markers from anxiety descriptions unless explicitly stated
- Never add E_211 from single vomiting mention
"""

app = FastAPI(
    title="TMS API",
    description="Therapist Machine Support — AI-powered pediatric symptom analysis and triage system",
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

@app.post("/analys")
async def analys_endpoint(body: AnalysRequest, user: dict = Depends(require_patient)):
    patient_id = user["patient_id"]

    top3 = sklearn_predict(body.evidences, age=body.age, sex=body.sex, top_n=3)
    top1_name, top1_score = top3[0]

    preliminary_diagnose = " ".join(name for name, _ in top3)

    avg_score = sum(sc for _, sc in top3) / len(top3)
    day = insert_disease(
        patient_id,
        body.evidences,
        preliminary_diagnose,
        avg_score,
        body.diagnose_setup,
    )

    slices = [
        {"name": name, "label": disease_labels.get(name, name), "score": sc}
        for name, sc in top3
    ]

    zone = classify_zone(top1_name, top1_score, evidences=body.evidences)

    # Override recommendation for red zone with urgent message
    recommendation = disease_recommendations.get(top1_name, "Консультация врача обязательна.")
    if zone == "red":
        recommendation = "⚠️ СРОЧНО обратитесь к врачу или вызовите скорую помощь! " + recommendation

    return {
        "day": day,
        "diseaseName": top1_name,
        "diseaseLabel": disease_labels.get(top1_name, top1_name),
        "doctor": disease_doctors.get(top1_name, "Терапевт"),
        "recommendation": recommendation,
        "slices": slices,
        "score": top1_score,
        "zone": zone,
    }


@app.post("/extract_symptoms")
@limiter.limit("20/minute")
async def extract_symptoms_endpoint(
    request: Request,
    body: ExtractSymptomsRequest,
    user: dict = Depends(require_patient),
):
    """
    3-layer symptom extraction pipeline (all layers always run):
      Layer 1: Groq LLaMA 3.3 70B — context, negation, implicit symptoms, temporal markers.
      Layer 2: Vocab regex + BM25 — catches medical terms LLM may have normalized.
      Layer 3: QAEngine picks up missing binary features via /next_questions.
    Smart merge: union of all layers, filtered by LLM negation list.
    """
    import httpx as _httpx
    from retrieval import get_retriever, vocab_extract, extract_age_sex

    # ── Layer 1: Groq LLaMA 3.3 70B (context + negation + implicit) ─
    llm_confirmed: List[str] = []
    llm_negated: set = set()
    llm_uncertain: List[str] = []
    llm_age = None
    llm_sex = None
    onset_days = None
    severity = None

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        try:
            async with _httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": _LLM_EXTRACTION_PROMPT},
                            {"role": "user", "content": f'Patient complaint (language: auto-detect Russian/Kazakh/English):\n\n"{body.text}"\n\nExtract all diagnostic evidence. Return JSON only.'},
                        ],
                        "temperature": 0.0,
                        "max_tokens": 800,
                    },
                )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()
                layer1 = json.loads(raw)
                llm_confirmed = layer1.get("evidences", [])
                llm_negated = set(layer1.get("negated", []))
                llm_uncertain = layer1.get("uncertain", [])
                llm_age = layer1.get("age")
                llm_sex = layer1.get("sex")
                onset_days = layer1.get("onset_days")
                severity = layer1.get("severity")
                logger.info(
                    "Layer1 LLM: %d confirmed, %d negated, %d uncertain, age=%s sex=%s",
                    len(llm_confirmed), len(llm_negated), len(llm_uncertain),
                    llm_age, llm_sex,
                )
            else:
                logger.error("Layer1 LLM HTTP %d: %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.error("Layer1 LLM failed: %s", e)
    else:
        logger.warning("GROQ_API_KEY not set — Layer 1 skipped")

    # ── Layer 2a: Vocab regex (always runs, high precision) ──────────
    vocab_evs = vocab_extract(body.text)
    regex_age, regex_sex = extract_age_sex(body.text)
    logger.info("Layer2 Vocab: %d evidences: %s", len(vocab_evs), vocab_evs)

    # ── Layer 2b: BM25 (always runs, catches terms LLM normalized) ───
    bm25_evs: List[str] = []
    try:
        retriever = get_retriever()
        bm25_candidates = retriever.retrieve(body.text, top_k=20)
        for ev_id, score in bm25_candidates:
            if score < 6.0:
                break
            meta = retriever._meta.get(ev_id, {})
            if meta.get("data_type") == "B" and not meta.get("is_antecedent", False):
                bm25_evs.append(ev_id)
            if len(bm25_evs) >= 8:
                break
        logger.info(
            "Layer2 BM25: %d evidences (top-5 scores: %s)",
            len(bm25_evs),
            [(eid, round(s, 1)) for eid, s in bm25_candidates[:5]],
        )
    except Exception as e:
        logger.warning("Layer2 BM25 failed: %s", e)

    # ── Smart Merge: union of all layers, filtered by negation ───────
    merged: set = set()
    layers: Dict[str, List[str]] = {"llm": [], "vocab": [], "bm25": []}

    # LLM confirmed (highest trust — understands context & implicit)
    for ev in llm_confirmed:
        merged.add(ev)
        layers["llm"].append(ev)

    # Vocab regex (high precision, negation already handled internally)
    for ev in vocab_evs:
        if ev not in llm_negated:
            if ev not in merged:
                layers["vocab"].append(ev)
            merged.add(ev)

    # BM25 (moderate precision, catches normalized terms, filter by negation)
    for ev in bm25_evs:
        if ev not in llm_negated and ev not in merged:
            merged.add(ev)
            layers["bm25"].append(ev)

    # Promote uncertain symptoms if confirmed by another layer
    vocab_bm25_set = set(vocab_evs) | set(bm25_evs)
    for ev in llm_uncertain:
        if ev in vocab_bm25_set and ev not in merged:
            merged.add(ev)
            layers["llm"].append(ev)

    # Demographics: prefer LLM, fallback to regex
    age = llm_age or regex_age or 25
    sex = llm_sex or regex_sex or "M"
    if not isinstance(age, int) or not (0 <= age <= 120):
        age = 25
    if sex not in ("M", "F"):
        sex = "M"

    valid_evidences, invalid = validate_evidences(list(merged))
    if invalid:
        logger.warning("extract_symptoms: filtered %d invalid tokens: %s", len(invalid), invalid)

    logger.info(
        "3-layer merge: %d total (llm=%d, vocab=%d, bm25=%d) -> %d valid (age=%s sex=%s)",
        len(merged), len(layers["llm"]), len(layers["vocab"]),
        len(layers["bm25"]), len(valid_evidences), age, sex,
    )
    return {
        "evidences": valid_evidences,
        "age": age,
        "sex": sex,
        "onset_days": onset_days,
        "severity": severity,
        "noSymptoms": len(valid_evidences) == 0,
        "layers": {
            "llm": len(layers["llm"]),
            "vocab": len(layers["vocab"]),
            "bm25": len(layers["bm25"]),
        },
    }


@app.post("/next_questions")
@limiter.limit("30/minute")
async def next_questions_endpoint(
    request: Request,  # noqa: used by slowapi limiter
    body: NextQuestionsRequest,
    _user: dict = Depends(require_patient),
):
    """
    Two-round follow-up question selection via QAEngine.
    Round 1 (exploration): JSD across full disease space.
    Round 2 (discrimination): gap between top-1 and top-2.
    Falls back to old find_discriminative_evidences if QAEngine unavailable.
    """
    result = qa_engine.get_questions(
        evidences=body.evidences,
        age=body.age,
        sex=body.sex,
        round_num=body.round_num,
        n_questions=body.n,
    )
    questions = [
        {
            "evidence_id": q.evidence_id,
            "question_ru": q.question_ru,
            "question_kk": q.question_kk,
            "score": q.score,
            "reason": q.reason,
        }
        for q in result.questions
    ]
    top3 = result.top3

    # Safety layer: check current evidences for critical symptoms
    ev_set = set(body.evidences) if body.evidences else set()
    critical_alert = None

    if "E_112" in ev_set:
        critical_alert = {
            "level": "red",
            "message": "⚠️ СТРИДОР (шумное дыхание на вдохе) — признак обструкции дыхательных путей. "
                       "Немедленно вызовите скорую помощь (103)!",
        }
    elif ev_set & RED_FLAG_EVIDENCES:
        critical_alert = {
            "level": "red",
            "message": "⚠️ Обнаружены опасные симптомы. Рекомендуется срочная консультация врача или скорая помощь.",
        }
    else:
        for combo in RED_FLAG_COMBOS:
            if combo <= ev_set:
                critical_alert = {
                    "level": "red",
                    "message": "⚠️ Сочетание симптомов указывает на возможное неотложное состояние. "
                               "Обратитесь к врачу как можно скорее.",
                }
                break

    top1_name = top3[0][0] if top3 else ""
    top1_score = top3[0][1] if top3 else 0.0
    zone = classify_zone(top1_name, top1_score, evidences=body.evidences)

    return {
        "questions": questions,
        "current_top3": [
            {"disease": name, "label": disease_labels.get(name, name), "score": round(sc, 3)}
            for name, sc in top3
        ],
        "zone": zone,
        "critical_alert": critical_alert,
        "should_stop": result.should_stop,
        "stop_reason": result.stop_reason,
        "confidence": round(result.confidence, 4),
        "round_type": result.round_type,
    }


@app.post("/save_explanation")
async def save_explanation_endpoint(body: SaveExplanationRequest, user: dict = Depends(require_patient)):
    patient_id = user.get("patient_id")
    if not day_belongs_to_patient(body.day_id, patient_id):
        raise HTTPException(status_code=403, detail="Access denied")
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE diary_days
            SET patient_explanation = %s, doctor_explanation = %s
            WHERE day_id = %s
        """, (encrypt_field(body.patient_explanation), encrypt_field(body.doctor_explanation), body.day_id))
        conn.commit()
        return {"ok": cur.rowcount > 0}
    finally:
        conn.close()

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
    return {"symptoms": symptoms}

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

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured on server")

    import httpx

    vision_model = os.environ.get("GROQ_VISION_MODEL", "llama-4-scout-17b-16e-instruct")

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

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={
                    "model": vision_model,
                    "messages":[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content":[
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

    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="AI returned invalid JSON")

    if "error" in parsed:
        raise HTTPException(status_code=422, detail=parsed["error"])

    test_type = parsed.get("test_type", "Неизвестный анализ")
    test_date = parsed.get("test_date", "")
    interpretation = parsed.get("interpretation", "")
    results = parsed.get("results",[])

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