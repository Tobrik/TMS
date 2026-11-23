import os
import json
import asyncio
import base64
import logging
import time
import uuid
import io
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, constr, field_validator, model_validator
from typing import Optional, List, Dict, Any, Tuple, Literal
import re
import psycopg2, psycopg2.extras, psycopg2.errors, hashlib, secrets
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
    require_real_doctor,
    require_patient_or_doctor,
)
from crypto_utils import encrypt_field, decrypt_field
from ml_model import (
    EVIDENCES_META, LABEL_CLASSES, _GEO_EXCLUDED, _GROUP_MAP, _NEVER_ASK, _feat_idx,
    evidence_fit, explain_prediction, predict_distribution, qa_engine, validate_evidences,
)
import extraction
import llm
import pipeline_log as plog
import triage
import qa_engine as qa_engine_module
from metrics import metric

# Written by eval/build_report.py; read on every /accuracy_report request.
ACCURACY_REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "accuracy_report.json")

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

def _strong_password(v: str) -> str:
    if not re.search(r'[A-Za-z]', v):
        raise ValueError("Пароль должен содержать хотя бы одну букву")
    if not re.search(r'\d', v):
        raise ValueError("Пароль должен содержать хотя бы одну цифру")
    return v

_USERNAME_RE = re.compile(r"^[a-z0-9._-]{3,32}$")

def _clean_username(v: str) -> str:
    v = v.strip().lower()
    if not _USERNAME_RE.fullmatch(v):
        raise ValueError("username: 3-32 chars, a-z 0-9 . _ -")
    return v

class LoginDoctorRequest(BaseModel):
    username: Optional[str] = Field(default=None, min_length=1, max_length=64)
    doctor_id: Optional[int] = Field(default=None, gt=0)
    password: constr(min_length=1, max_length=64)

    @model_validator(mode="after")
    def exactly_one_identifier(self):
        if (self.username is None) == (self.doctor_id is None):
            raise ValueError("send exactly one of username or doctor_id")
        return self

class DemoPatientRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=200)
    lang: Literal["ru", "kk", "en"] = "ru"

class RegisterDoctorRequest(BaseModel):
    invite_code: str = Field(..., min_length=1, max_length=64)
    username: str = Field(..., max_length=64)
    full_name: str = Field(..., min_length=1, max_length=200)
    specialty: str = Field(..., min_length=1, max_length=200)
    password: str = Field(..., min_length=8, max_length=64)

    @field_validator("username")
    @classmethod
    def username_rules(cls, v: str) -> str:
        return _clean_username(v)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        return _strong_password(v)

class ChangePasswordRequest(BaseModel):
    old_password: constr(min_length=1, max_length=64)
    new_password: str = Field(..., min_length=8, max_length=64)

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        return _strong_password(v)

class LoginResponse(BaseModel):
    login: bool
    access_token: str
    token_type: str = "bearer"

class DoctorInfo(BaseModel):
    doctor_id: int
    username: Optional[str] = None
    is_demo: bool = False
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
    unknown: List[str] = Field(default_factory=list, max_length=300)  # "don't know": stored, not used by the model
    age: Optional[int] = Field(default=None, ge=0, le=120)          # None = not stated
    sex: Optional[str] = Field(default=None, pattern="^[MF]$")
    intake_id: Optional[int] = Field(default=None, gt=0)            # from /extract_symptoms
    diagnose_setup: str = "Nothing"

class ExtractSymptomsRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000)

class NextQuestionsRequest(BaseModel):
    intake_id: Optional[int] = Field(default=None, gt=0)            # only to tie log lines together
    evidences: List[str] = Field(..., min_length=1, max_length=300)
    absent: List[str] = Field(default_factory=list, max_length=300)
    unknown: List[str] = Field(default_factory=list, max_length=300)  # "don't know": never asked again
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
    # Patients send {} and get their own results; doctors must send patient_id.
    patient_id: Optional[int] = Field(default=None, gt=0)

class DoctorFeedbackRequest(BaseModel):
    day_id: int = Field(..., gt=0)
    diagnosis: str = Field(..., min_length=1, max_length=200)  # a /conditions name or "other"
    other_text: Optional[str] = Field(default=None, max_length=200)
    zone: str = Field(..., pattern="^(red|yellow|green)$")
    note: Optional[str] = Field(default=None, max_length=1000)

    @field_validator("diagnosis")
    @classmethod
    def known_diagnosis(cls, v: str) -> str:
        if v != "other" and v not in _CONDITION_NAMES:
            raise ValueError("diagnosis must be one of /conditions or 'other'")
        return v

    @model_validator(mode="after")
    def other_needs_text(self):
        if self.diagnosis == "other" and not (self.other_text or "").strip():
            raise ValueError("other_text is required when diagnosis is 'other'")
        return self

# ─── Data & Config ───

# Bump when the model or its output changes; stored with every analysis so
# doctor feedback can be split by model version.
MODEL_VERSION = "v2.1-nb"

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

# Canonical names predict_distribution can return: LABEL_CLASSES pooled into
# clinical groups, minus the geo-excluded classes (see ml_model.pool_classes).
_CONDITION_NAMES = frozenset(
    _GROUP_MAP.get(d, d) for d in LABEL_CLASSES
) - _GEO_EXCLUDED

CONDITIONS = sorted(
    [{"name": n, "label_ru": disease_labels.get(n, n), "tier": triage.tier_of(n)} for n in _CONDITION_NAMES],
    key=lambda c: c["label_ru"],
)

# The askable symptoms: binary, in the model's features, never asked as a follow-up.
_ASKABLE_CODES = [
    code for code, meta in EVIDENCES_META.items()
    if meta.get("data_type") == "B" and not meta.get("is_antecedent")
    and code in _feat_idx and code not in _NEVER_ASK
]
EVIDENCE_CATALOG = sorted(
    [{"code": code, "label": extraction.evidence_label(code)} for code in _ASKABLE_CODES],
    key=lambda e: e["label"]["ru"],
)

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

        -- Doctor-confirmed diagnosis per diary day. Append-only: the latest row counts.
        CREATE TABLE IF NOT EXISTS diagnosis_feedback (
            feedback_id SERIAL PRIMARY KEY,
            day_id INTEGER NOT NULL REFERENCES diary_days(day_id) ON DELETE CASCADE,
            doctor_id INTEGER REFERENCES doctors(doctor_id) ON DELETE SET NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            diagnosis TEXT NOT NULL,
            other_text TEXT,          -- encrypt_field()
            zone TEXT NOT NULL,
            note TEXT,                -- encrypt_field()
            model_top1 TEXT,          -- diary_days.disease_predict at feedback time
            model_top3 TEXT,          -- JSON list of names from analysis_json["top3"]
            model_zone TEXT,          -- diary_days.zone at feedback time
            model_version TEXT        -- analysis_json["model_version"] or "v2.0"
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_day ON diagnosis_feedback(day_id, feedback_id);

        -- Doctor accounts, demo flags, invites (all idempotent)
        ALTER TABLE doctors  ADD COLUMN IF NOT EXISTS username TEXT;
        ALTER TABLE doctors  ADD COLUMN IF NOT EXISTS is_demo BOOLEAN NOT NULL DEFAULT FALSE;
        ALTER TABLE patients ADD COLUMN IF NOT EXISTS is_demo BOOLEAN NOT NULL DEFAULT FALSE;
        ALTER TABLE patients ADD COLUMN IF NOT EXISTS demo_seed TEXT;
        UPDATE doctors SET username = 'doctor' || doctor_id WHERE username IS NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS idx_doctors_username ON doctors(username);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_patients_demo_seed ON patients(demo_seed) WHERE demo_seed IS NOT NULL;
        CREATE TABLE IF NOT EXISTS doctor_invites (
            code_hash  TEXT PRIMARY KEY,
            created_by INTEGER REFERENCES doctors(doctor_id) ON DELETE SET NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            expires_at TIMESTAMP NOT NULL,
            used_by    INTEGER REFERENCES doctors(doctor_id) ON DELETE SET NULL,
            used_at    TIMESTAMP
        );
        """)
        conn.commit()
    finally:
        conn.close()

RESERVED_USERNAME = "demo"  # the demo doctor's username; no real account may take it

def parse_doctor_seeds(raw: str) -> List[Tuple[Optional[str], str, str, str]]:
    """DOCTOR_SEEDS -> [(username or None, full_name, specialty, password)].
    `username|Full Name|Specialty|password` or the legacy `Full Name|Specialty|password`."""
    out = []
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = [p.strip() for p in entry.split("|")]
        try:
            if len(parts) == 4:
                username = _clean_username(parts[0])
                if username == RESERVED_USERNAME:
                    raise ValueError("reserved username")
                parts = parts[1:]
            elif len(parts) == 3:
                username = None
            else:
                raise ValueError("wrong number of parts")
            if not all(parts):
                raise ValueError("empty field")
        except ValueError:
            logger.warning("Skipping malformed DOCTOR_SEEDS entry")
            continue
        out.append((username, parts[0], parts[1], parts[2]))
    return out

def seed_doctors():
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM doctors WHERE NOT is_demo")
        if cur.fetchone()[0] > 0:
            return

        seeds_raw = os.environ.get("DOCTOR_SEEDS", "")
        if not seeds_raw:
            logger.warning("DOCTOR_SEEDS env variable not set — no doctors seeded")
            return

        for username, full_name, spec, pwd in parse_doctor_seeds(seeds_raw):
            cur.execute("""
                INSERT INTO doctors(full_name, specialty, password_hash, username)
                VALUES (%s,%s,%s,%s) RETURNING doctor_id
            """, (encrypt_field(full_name), encrypt_field(spec), hash_password(pwd), username))
            doctor_id = cur.fetchone()[0]
            if username is None:
                cur.execute("UPDATE doctors SET username = 'doctor' || doctor_id WHERE doctor_id = %s", (doctor_id,))
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

def find_doctor_for_login(username: Optional[str], doctor_id: Optional[int]) -> Optional[Tuple[int, str, bool]]:
    conn = connect()
    try:
        cur = conn.cursor()
        if username is not None:
            cur.execute("SELECT doctor_id, password_hash, is_demo FROM doctors WHERE username = %s",
                        (username.strip().lower(),))
        else:
            cur.execute("SELECT doctor_id, password_hash, is_demo FROM doctors WHERE doctor_id = %s", (doctor_id,))
        row = cur.fetchone()
        return (row[0], row[1], row[2]) if row else None
    finally:
        conn.close()

def get_doctor_info(doctor_id: int) -> Optional[Dict[str, Any]]:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""SELECT doctor_id, username, full_name, specialty, is_demo, created_at
                       FROM doctors WHERE doctor_id = %s""", (doctor_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {"doctor_id": row[0], "username": row[1], "full_name": decrypt_field(row[2]),
                "specialty": decrypt_field(row[3]), "is_demo": row[4], "created_at": str(row[5])}
    finally:
        conn.close()

_INVITE_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"

def _invite_hash(code: str) -> str:
    normalised = re.sub(r"[\s-]", "", code).upper()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

def create_invite(doctor_id: int) -> Dict[str, str]:
    raw = "".join(secrets.choice(_INVITE_ALPHABET) for _ in range(8))
    code = f"TMS-{raw[:4]}-{raw[4:]}"
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO doctor_invites(code_hash, created_by, expires_at)
            VALUES (%s, %s, NOW() + INTERVAL '7 days') RETURNING expires_at
        """, (_invite_hash(code), doctor_id))
        expires_at = cur.fetchone()[0]
        conn.commit()
        return {"code": code, "expires_at": expires_at.replace(microsecond=0).isoformat()}
    finally:
        conn.close()

def register_doctor_with_invite(body: "RegisterDoctorRequest") -> Optional[int]:
    """New doctor id, or None when the invite is unknown, used or expired.
    Raises psycopg2.errors.UniqueViolation when the username is taken (invite stays unused)."""
    conn = connect()
    try:
        cur = conn.cursor()
        code_hash = _invite_hash(body.invite_code)
        # Atomic claim: only one concurrent request can flip used_at.
        cur.execute("""
            UPDATE doctor_invites SET used_at = NOW()
            WHERE code_hash = %s AND used_by IS NULL AND used_at IS NULL AND expires_at > NOW()
        """, (code_hash,))
        if cur.rowcount != 1:
            conn.rollback()
            return None
        cur.execute("""
            INSERT INTO doctors(full_name, specialty, password_hash, username)
            VALUES (%s,%s,%s,%s) RETURNING doctor_id
        """, (encrypt_field(body.full_name.strip()), encrypt_field(body.specialty.strip()),
              hash_password(body.password), body.username))
        doctor_id = cur.fetchone()[0]
        cur.execute("UPDATE doctor_invites SET used_by = %s WHERE code_hash = %s", (doctor_id, code_hash))
        conn.commit()
        return doctor_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def change_doctor_password(doctor_id: int, old_password: str, new_password: str) -> bool:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM doctors WHERE doctor_id = %s", (doctor_id,))
        row = cur.fetchone()
        if not row or not verify_password(old_password, row[0]):
            return False
        cur.execute("UPDATE doctors SET password_hash = %s WHERE doctor_id = %s",
                    (hash_password(new_password), doctor_id))
        conn.commit()
        return True
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
    created_at=None,
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
                intake_id,
                created_at
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, COALESCE(%s::timestamptz, NOW())) RETURNING day_id
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
            created_at,
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

def list_doctors_db(is_demo: bool = False) -> List[Dict[str, Any]]:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT doctor_id, username, full_name, specialty, created_at
            FROM doctors
            WHERE is_demo = %s
            ORDER BY doctor_id
        """, (is_demo,))
        rows = cur.fetchall()
        return [
            {"doctor_id": r[0], "username": r[1], "full_name": decrypt_field(r[2]), "specialty": decrypt_field(r[3]), "created_at": str(r[4])}
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
                d.zone,
                fb.diagnosis,
                fb.other_text,
                fb.zone,
                fb.created_at
            FROM diary_days d
            LEFT JOIN doctors doc ON doc.doctor_id = d.doctor_id
            LEFT JOIN LATERAL (
                SELECT f.diagnosis, f.other_text, f.zone, f.created_at
                FROM diagnosis_feedback f
                WHERE f.day_id = d.day_id
                ORDER BY f.feedback_id DESC
                LIMIT 1
            ) fb ON TRUE
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
            "feedback": _feedback_view(r[11], r[12], r[13], r[14]) if r[11] is not None else None,
        } for r in rows]
    finally:
        conn.close()


def _feedback_view(diagnosis: str, other_text: Optional[str], zone: str, created_at) -> Dict[str, Any]:
    return {
        "diagnosis": diagnosis,
        "label_ru": None if diagnosis == "other" else disease_labels.get(diagnosis, diagnosis),
        "other_text": decrypt_field(other_text),
        "zone": zone,
        "created_at": created_at.isoformat() if created_at else None,
    }


def insert_feedback(day_id: int, doctor_id: int, body: "DoctorFeedbackRequest") -> Optional[Dict[str, Any]]:
    """Append a doctor's diagnosis for a diary day. The model's answer is copied
    from the day row so later changes to the model don't rewrite history.
    None if the day does not exist."""
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT disease_predict, zone, analysis_json FROM diary_days WHERE day_id = %s", (day_id,))
        row = cur.fetchone()
        if row is None:
            return None
        model_top1 = top1_name(decrypt_field(row[0])) or None
        analysis = json.loads(decrypt_field(row[2]) or "{}") if row[2] else {}
        model_top3 = [name for name, _ in (analysis.get("top3") or [])]
        model_version = analysis.get("model_version", "v2.0")
        cur.execute("""
            INSERT INTO diagnosis_feedback(
                day_id, doctor_id, diagnosis, other_text, zone, note,
                model_top1, model_top3, model_zone, model_version
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING feedback_id
        """, (
            day_id, doctor_id, body.diagnosis,
            encrypt_field(body.other_text.strip()) if body.diagnosis == "other" else None,
            body.zone, encrypt_field(body.note),
            model_top1, json.dumps(model_top3, ensure_ascii=False), row[1], model_version,
        ))
        feedback_id = cur.fetchone()[0]
        conn.commit()
        return {
            "feedback_id": feedback_id,
            "agrees_top1": body.diagnosis != "other" and body.diagnosis == model_top1,
            "in_top3": body.diagnosis != "other" and body.diagnosis in model_top3,
        }
    finally:
        conn.close()

_ZONES = ("red", "yellow", "green")
_ZONE_RANK = {"green": 0, "yellow": 1, "red": 2}


# The latest doctor feedback row per diary day: the only rows the accuracy numbers count.
# Feedback on demo patients is left out: demo sessions must not move the real numbers.
_REAL_FEEDBACK_FROM = """
    FROM diagnosis_feedback f
    JOIN diary_days d ON d.day_id = f.day_id
    JOIN patients p ON p.patient_id = d.patient_id
    WHERE NOT p.is_demo
"""
_LATEST_FEEDBACK_SQL = """
    SELECT DISTINCT ON (f.day_id) f.day_id, f.diagnosis, f.zone, f.model_top1, f.model_top3, f.model_zone, f.model_version
""" + _REAL_FEEDBACK_FROM + """
    ORDER BY f.day_id, f.feedback_id DESC
"""


def get_feedback_stats() -> Dict[str, Any]:
    """Accuracy of the model against doctor-confirmed diagnoses (latest row per day)."""
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(_LATEST_FEEDBACK_SQL)
        rows = cur.fetchall()
    finally:
        conn.close()
    return feedback_stats_from_rows(rows)


def feedback_stats_from_rows(rows) -> Dict[str, Any]:
    """rows: (day_id, diagnosis, zone, model_top1, model_top3 JSON, model_zone, model_version),
    one per diary day. Pure function so the arithmetic can be tested without a database."""
    zone_matrix = {d: {m: 0 for m in _ZONES} for d in _ZONES}
    versions: Dict[str, Dict[str, int]] = {}
    confusions: Dict[tuple, int] = {}
    n = 0

    for _day, diagnosis, doctor_zone, model_top1, model_top3, model_zone, model_version in rows:
        n += 1
        other = diagnosis == "other"
        top3 = json.loads(model_top3) if model_top3 else []
        v = versions.setdefault(model_version or "v2.0", {
            "n": 0, "other": 0, "top1": 0, "top3": 0, "zone_n": 0, "zone_exact": 0, "under": 0, "over": 0})
        v["n"] += 1
        v["other"] += other

        if not other:
            v["top1"] += diagnosis == model_top1
            v["top3"] += diagnosis in top3
            if model_top1 and model_top1 != diagnosis:
                key = (model_top1, diagnosis)
                confusions[key] = confusions.get(key, 0) + 1

        if doctor_zone in _ZONE_RANK and model_zone in _ZONE_RANK:
            v["zone_n"] += 1
            v["zone_exact"] += doctor_zone == model_zone
            v["under"] += _ZONE_RANK[model_zone] < _ZONE_RANK[doctor_zone]
            v["over"] += _ZONE_RANK[model_zone] > _ZONE_RANK[doctor_zone]
            zone_matrix[doctor_zone][model_zone] += 1

    def ratio(part: int, whole: int) -> float:
        return round(part / whole, 3) if whole else 0.0

    by_version = {
        name: {
            "n": v["n"], "other": v["other"],
            "top1": ratio(v["top1"], v["n"]), "top3": ratio(v["top3"], v["n"]),
            "zone_exact": ratio(v["zone_exact"], v["zone_n"]),
            "under_triage": v["under"], "over_triage": v["over"],
        }
        for name, v in sorted(versions.items())
    }
    top_confusions = sorted(confusions.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    return {
        "n": n,
        "by_version": by_version,
        "zone_matrix": zone_matrix,
        "confusions": [{"model": m, "doctor": d, "count": c} for (m, d), c in top_confusions],
    }


def real_patients_from_rows(rows, since: Optional[str]) -> Dict[str, Any]:
    """Real-patient accuracy with metric objects. rows: the _LATEST_FEEDBACK_SQL shape,
    one per diary day. Pure function, so the arithmetic can be tested without a database."""
    n = other = top1 = top3 = zone_n = zone_exact = under = over = red_n = red_hit = 0
    versions: Dict[str, Dict[str, int]] = {}
    for _day, diagnosis, doctor_zone, model_top1, model_top3, model_zone, model_version in rows:
        is_other = diagnosis == "other"
        top3_names = json.loads(model_top3) if model_top3 else []
        hit1 = (not is_other) and diagnosis == model_top1
        hit3 = (not is_other) and diagnosis in top3_names
        n += 1
        other += is_other
        top1 += hit1
        top3 += hit3
        v = versions.setdefault(model_version or "v2.0", {"n": 0, "top1": 0, "top3": 0})
        v["n"] += 1
        v["top1"] += hit1
        v["top3"] += hit3
        if doctor_zone in _ZONE_RANK and model_zone in _ZONE_RANK:
            zone_n += 1
            zone_exact += doctor_zone == model_zone
            under += _ZONE_RANK[model_zone] < _ZONE_RANK[doctor_zone]
            over += _ZONE_RANK[model_zone] > _ZONE_RANK[doctor_zone]
        if doctor_zone == "red" and model_zone in _ZONE_RANK:
            red_n += 1
            red_hit += model_zone == "red"
    return {
        "n": n,
        "other": other,
        "top1": metric(top1, n),
        "top3": metric(top3, n),
        "zone_exact": metric(zone_exact, zone_n),
        "red_sensitivity": metric(red_hit, red_n),
        "under_triage": metric(under, zone_n),
        "over_triage": metric(over, zone_n),
        "by_version": {
            name: {"n": v["n"], "top1": metric(v["top1"], v["n"]), "top3": metric(v["top3"], v["n"])}
            for name, v in sorted(versions.items())
        },
        "since": since,
    }


def get_real_patients() -> Dict[str, Any]:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(_LATEST_FEEDBACK_SQL)
        rows = cur.fetchall()
        cur.execute("SELECT MIN(f.created_at)::date " + _REAL_FEEDBACK_FROM)
        first = cur.fetchone()[0]
    finally:
        conn.close()
    return real_patients_from_rows(rows, first.isoformat() if first else None)


def get_accuracy_report() -> Dict[str, Any]:
    """The offline snapshot written by eval/build_report.py, plus live real-patient numbers."""
    real = get_real_patients()
    try:
        with open(ACCURACY_REPORT_PATH, encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, ValueError):
        return {"report_missing": True, "real_patients": real}
    if not isinstance(report, dict):
        return {"report_missing": True, "real_patients": real}
    return {**report, "real_patients": real}


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

def hidden_from_doctor(user: dict, patient_id: Optional[int] = None, day_id: Optional[int] = None) -> bool:
    """The isolation rule, in one place. True when the caller is a doctor and the patient (or the
    patient owning the day) is a demo patient while the token is not a demo token, or the other way
    round. Callers then answer exactly as if the patient or day did not exist. A patient or day that
    really does not exist is not hidden: the endpoint's ordinary not-found path handles it."""
    if user.get("role") != "doctor":
        return False
    conn = connect()
    try:
        cur = conn.cursor()
        if day_id is not None:
            cur.execute("""SELECT p.is_demo FROM diary_days d JOIN patients p ON p.patient_id = d.patient_id
                           WHERE d.day_id = %s""", (day_id,))
        else:
            cur.execute("SELECT is_demo FROM patients WHERE patient_id = %s", (patient_id,))
        row = cur.fetchone()
    finally:
        conn.close()
    return row is not None and bool(row[0]) != bool(user.get("demo"))

# Triage lives in triage.py. Zones are computed once in /analys and stored with
# the diary day, so the doctor's dashboard shows exactly what the patient saw.

_ZONE_ORDER = {"red": 0, "yellow": 1, "green": 2}
COMPLAINT_PREVIEW_CHARS = 160

def _complaint_preview(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    return text if len(text) <= COMPLAINT_PREVIEW_CHARS else text[:COMPLAINT_PREVIEW_CHARS - 1].rstrip() + "…"

def get_all_patients_triage(is_demo: bool = False) -> list:
    """The queue for a doctor: patients whose is_demo equals the caller's flag, red first, then
    yellow, green, not assessed; newest assessment first inside a zone."""
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT p.patient_id, p.full_name, p.city, p.created_at,
                   d.disease_predict, d.score, d.created_at as diag_date,
                   d.day_id, d.zone,
                   EXISTS (SELECT 1 FROM diagnosis_feedback f WHERE f.day_id = d.day_id) AS has_feedback,
                   d.analysis_json, i.text, p.is_demo
            FROM patients p
            LEFT JOIN diary_days d ON d.day_id = (
                SELECT MAX(d2.day_id) FROM diary_days d2 WHERE d2.patient_id = p.patient_id
            )
            LEFT JOIN intakes i ON i.intake_id = d.intake_id
            WHERE p.is_demo = %s
            ORDER BY p.patient_id
        """, (is_demo,))
        rows = cur.fetchall()

        # v1 rows (no stored zone) are re-assessed below; their reported symptoms
        # come from one query for all of them.
        v1_day_ids = [r[7] for r in rows if r[7] is not None and r[8] is None]
        symptoms_by_day: Dict[int, List[str]] = {}
        if v1_day_ids:
            cur.execute(
                "SELECT day_id, symptom_code FROM diary_symptoms WHERE day_id = ANY(%s) AND value > 0",
                (v1_day_ids,),
            )
            for v1_day_id, symptom_code in cur.fetchall():
                symptoms_by_day.setdefault(v1_day_id, []).append(symptom_code)

        result = []
        for r in rows:
            disease = top1_name(decrypt_field(r[4]))
            score = r[5] or 0.0
            day_id = r[7]
            zone = r[8]

            if day_id is None:
                # No diary day at all: nothing to triage yet.
                zone = None
            elif zone is None:
                # v1 row: no stored zone. Re-assess from the top-1 disease and the
                # reported symptoms (v1 stored the top-3 average as score, so it
                # can't serve as a probability).
                evidences = symptoms_by_day.get(day_id, [])
                zone = triage.assess({disease: 1.0} if disease else {}, evidences).zone

            analysis = {}
            if r[10]:
                try:
                    analysis = json.loads(decrypt_field(r[10]) or "{}")
                except (ValueError, TypeError):
                    analysis = {}
            sort_at = r[6] or r[3]

            result.append({
                "patient_id": r[0],
                "full_name": decrypt_field(r[1]),
                "city": decrypt_field(r[2]),
                "created_at": str(r[3]),
                "last_disease": disease,
                "last_score": score,
                "diag_date": str(r[6]) if r[6] else None,
                "zone": zone,
                "last_day_id": day_id,
                "feedback_pending": day_id is not None and not r[9],
                "complaint": _complaint_preview(decrypt_field(r[11])),
                "reasons": analysis.get("reasons") or [],
                "low_confidence": bool(analysis.get("low_confidence", False)),
                "is_demo": bool(r[12]),
                "_sort": (_ZONE_ORDER.get(zone, 3), -sort_at.timestamp()),
            })
        result.sort(key=lambda row: row["_sort"])
        for row in result:
            del row["_sort"]
        return result
    finally:
        conn.close()

# ─── Demo mode ───

DEMO_LIVE_TTL_HOURS = 48

def demo_enabled() -> bool:
    return os.environ.get("DEMO_MODE", "").strip().lower() in ("1", "true", "yes")

def _demo_cases() -> List[dict]:
    import demo_seed
    return demo_seed.DEMO_CASES

def _unusable_password_hash() -> str:
    return hash_password(secrets.token_urlsafe(32))

def ensure_demo_doctor() -> int:
    """The demo doctor's id (found by is_demo, not by username); created if missing."""
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT doctor_id FROM doctors WHERE is_demo ORDER BY doctor_id LIMIT 1")
        row = cur.fetchone()
        if row:
            return row[0]
        try:
            cur.execute("""
                INSERT INTO doctors(full_name, specialty, password_hash, username, is_demo)
                VALUES (%s,%s,%s,%s,TRUE) RETURNING doctor_id
            """, (encrypt_field("Демо-врач"), encrypt_field("Терапевт"), _unusable_password_hash(), RESERVED_USERNAME))
            doctor_id = cur.fetchone()[0]
            conn.commit()
            return doctor_id
        except psycopg2.errors.UniqueViolation:
            conn.rollback()  # a concurrent request created it first
            cur.execute("SELECT doctor_id FROM doctors WHERE is_demo ORDER BY doctor_id LIMIT 1")
            return cur.fetchone()[0]
    finally:
        conn.close()

def _seed_demo_case(case: dict) -> bool:
    """Insert one demo case unless it exists. False when it already exists (or a concurrent
    request is inserting it: the unique partial index on demo_seed lets only one win)."""
    conn = connect()
    try:
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO patients(full_name, city, password_hash, is_demo, demo_seed)
                VALUES (%s,%s,%s,TRUE,%s) ON CONFLICT DO NOTHING RETURNING patient_id
            """, (encrypt_field(case["name"]), encrypt_field(case.get("city", "")),
                  _unusable_password_hash(), case["key"]))
        except psycopg2.errors.UniqueViolation:
            conn.rollback()
            return False
        row = cur.fetchone()
        conn.commit()
    finally:
        conn.close()
    if row is None:
        return False
    patient_id = row[0]
    try:
        for day in case["days"]:
            intake_id = insert_intake(patient_id, day["complaint"], {"mode": "demo", "lang": day.get("lang")})
            conn = connect()
            try:
                cur = conn.cursor()
                cur.execute("SELECT NOW() - %s * INTERVAL '1 minute'", (int(day.get("minutes_ago", 0)),))
                created_at = cur.fetchone()[0]
            finally:
                conn.close()
            analyse_and_store(patient_id, list(day["evidences"]), list(day.get("absent", [])),
                              list(day.get("unknown", [])), case["age"], case["sex"], intake_id, None,
                              created_at=created_at)
    except Exception:
        # Leave no half-built case behind: the next request seeds it again.
        conn = connect()
        try:
            conn.cursor().execute("DELETE FROM patients WHERE patient_id = %s", (patient_id,))
            conn.commit()
        finally:
            conn.close()
        raise
    return True

def seed_demo_cases() -> int:
    return sum(_seed_demo_case(case) for case in _demo_cases())

def delete_old_demo_patients() -> int:
    """Live demo patients (not seeded cases) older than the TTL; days, intakes and feedback cascade."""
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM patients WHERE is_demo AND demo_seed IS NULL AND created_at < NOW() - %s * INTERVAL '1 hour'",
            (DEMO_LIVE_TTL_HOURS,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()

def create_demo_patient(name: Optional[str], lang: str) -> int:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO patients(full_name, city, password_hash, is_demo) VALUES (%s,%s,%s,TRUE) RETURNING patient_id",
            (encrypt_field(name or "Guest"), encrypt_field(""), _unusable_password_hash()))
        patient_id = cur.fetchone()[0]
        if not name:
            guest = {"ru": "Гость", "kk": "Қонақ"}.get(lang, "Guest")
            cur.execute("UPDATE patients SET full_name = %s WHERE patient_id = %s",
                        (encrypt_field(f"{guest} {patient_id}"), patient_id))
        conn.commit()
        return patient_id
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

# A database outage must not crash the whole function at import: then every
# endpoint (even /health) answers Vercel's opaque FUNCTION_INVOCATION_FAILED.
# Start anyway, log why, and retry from /health.
_DB_STARTUP_ERROR: Optional[str] = None
try:
    create_tables()
    seed_doctors()
except Exception as e:
    _DB_STARTUP_ERROR = f"{type(e).__name__}: {str(e).strip().splitlines()[0][:300]}"
    logger.error("Database unavailable at startup, tables not checked: %s", _DB_STARTUP_ERROR)

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

# CORS: see cors_origins(). Any origin unless CORS_ALLOWED_ORIGINS is set.
# The frontend sends a Bearer header, not cookies, so credentials stay off.
def cors_origins() -> List[str]:
    """CORS_ALLOWED_ORIGINS (comma-separated) restricts which sites may call the
    API; unset or empty allows any origin. Opt-in on purpose: the older
    ALLOWED_ORIGINS was never applied, and deployments still carry its
    localhost default, so honouring it would block the live frontend."""
    raw = os.environ.get("CORS_ALLOWED_ORIGINS", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ─── Health Check ───

@app.get("/health")
def health():
    global _DB_STARTUP_ERROR
    try:
        conn = connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
        finally:
            conn.close()
        if _DB_STARTUP_ERROR:  # the database came back: create any missing tables now
            create_tables()
            seed_doctors()
            _DB_STARTUP_ERROR = None
        return {"status": "healthy", "version": "1.0.0", "database": "ok"}
    except Exception as e:
        logger.error("Health check: database unreachable: %s", str(e).strip().splitlines()[0][:300])
        # No connection details in the public response; they are in the logs.
        return JSONResponse(status_code=503, content={"status": "unhealthy", "database": "unreachable"})

@app.get("/")
def root():
    return {"message": "TMS API is running"}

@app.get("/healthz")
def healthz():
    """Readiness of the naive Bayes model production uses. Does not load the
    optional RandomForest (scikit-learn is not installed in production)."""
    from ml_model import EVIDENCE_STATS, LABEL_CLASSES, N_FEATURES, POSTERIOR
    ready = N_FEATURES > 0 and bool(EVIDENCE_STATS.get("p_evidence_given_disease"))
    return {
        "model_loaded": ready,
        "posterior": POSTERIOR,
        "model_version": MODEL_VERSION,
        "n_features": N_FEATURES,
        "n_classes": len(LABEL_CLASSES),
    }

# ─── Auth endpoints (public, rate-limited) ───

@app.post("/regist_as_patient", response_model=RegisterResponse)
@limiter.limit("10/minute")
def regist_as_patient(request: Request, body: RegisterRequest):
    patient_id = register_as_patient(body.name, body.city, body.password)
    logger.info("Patient %d registered", patient_id)
    token = create_access_token({"role": "patient", "patient_id": patient_id})
    return RegisterResponse(patient_id=patient_id, access_token=token)

@app.post("/login_patient", response_model=LoginResponse)
@limiter.limit("5/minute")
def login_patient_endpoint(request: Request, body: LoginPatientRequest):
    flag = log_in_patient(body.patient_id, body.password)
    if not flag:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    logger.info("Patient %d logged in", body.patient_id)
    token = create_access_token({"role": "patient", "patient_id": body.patient_id})
    return LoginResponse(login=True, access_token=token)

def _doctor_login_response(doctor_id: int) -> DoctorLoginResponse:
    info = get_doctor_info(doctor_id)
    token_claims = {"role": "doctor", "doctor_id": doctor_id}
    if info and info["is_demo"]:
        token_claims["demo"] = True
    return DoctorLoginResponse(login=True, access_token=create_access_token(token_claims), doctor=info)

@app.post("/login_doctor", response_model=DoctorLoginResponse)
@limiter.limit("5/minute")
def login_doctor_endpoint(request: Request, body: LoginDoctorRequest):
    found = find_doctor_for_login(body.username, body.doctor_id)
    if found is None:
        hash_password(body.password)  # same cost as a real check, so timing does not reveal unknown users
        raise HTTPException(status_code=401, detail="Invalid credentials")
    doctor_id, stored_hash, is_demo = found
    if not verify_password(body.password, stored_hash) or is_demo:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    logger.info("Doctor %d logged in", doctor_id)
    return _doctor_login_response(doctor_id)

@app.post("/register_doctor", response_model=DoctorLoginResponse)
@limiter.limit("5/minute")
def register_doctor_endpoint(request: Request, body: RegisterDoctorRequest):
    if body.username == RESERVED_USERNAME:
        raise HTTPException(status_code=409, detail="username_taken")
    try:
        doctor_id = register_doctor_with_invite(body)
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(status_code=409, detail="username_taken")
    if doctor_id is None:
        raise HTTPException(status_code=400, detail="invalid_invite")
    logger.info("Doctor %d registered with an invite", doctor_id)
    return _doctor_login_response(doctor_id)

def _require_demo_enabled():
    if not demo_enabled():
        raise HTTPException(status_code=404, detail="demo_disabled")

@app.get("/demo/status")
def demo_status_endpoint():
    return {"enabled": demo_enabled()}

@app.post("/demo/doctor", response_model=DoctorLoginResponse)
@limiter.limit("20/minute")
def demo_doctor_endpoint(request: Request):
    _require_demo_enabled()
    doctor_id = ensure_demo_doctor()
    seed_demo_cases()
    delete_old_demo_patients()
    return _doctor_login_response(doctor_id)

@app.post("/demo/patient", response_model=RegisterResponse)
@limiter.limit("10/minute")
def demo_patient_endpoint(request: Request, body: Optional[DemoPatientRequest] = None):
    _require_demo_enabled()
    body = body or DemoPatientRequest()
    name = body.name.strip() if body.name and body.name.strip() else None
    patient_id = create_demo_patient(name, body.lang)
    token = create_access_token({"role": "patient", "patient_id": patient_id, "demo": True})
    return RegisterResponse(patient_id=patient_id, access_token=token)

@app.post("/doctor/invites")
@limiter.limit("10/minute")
def create_invite_endpoint(request: Request, user: dict = Depends(require_real_doctor)):
    return create_invite(user["doctor_id"])

@app.post("/doctor/change_password")
@limiter.limit("5/minute")
def change_password_endpoint(request: Request, body: ChangePasswordRequest, user: dict = Depends(require_real_doctor)):
    if not change_doctor_password(user["doctor_id"], body.old_password, body.new_password):
        raise HTTPException(status_code=400, detail="wrong_password")
    return {"ok": True}

@app.get("/doctor/me", response_model=DoctorInfo)
@limiter.limit("60/minute")
def doctor_me_endpoint(request: Request, user: dict = Depends(require_doctor)):
    info = get_doctor_info(user["doctor_id"])
    if info is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return info

@app.post("/list_doctor")
@limiter.limit("20/minute")
def list_doctor_endpoint(request: Request, _user: dict = Depends(require_doctor)):
    docs = list_doctors_db(bool(_user.get("demo")))
    return {"doctors": docs}

# ─── Patient endpoints (require patient or doctor token) ───

@app.get("/conditions")
def conditions_endpoint(_user: dict = Depends(require_patient_or_doctor)):
    """The diagnoses the model can output (the names /analys returns), with triage tier."""
    return {"conditions": CONDITIONS}

@app.get("/evidence_catalog")
def evidence_catalog_endpoint(_user: dict = Depends(require_patient_or_doctor)):
    """The askable binary symptoms, with labels in en / ru / kk."""
    return {"evidences": EVIDENCE_CATALOG}

@app.post("/get_history")
@limiter.limit("120/minute")
def get_history_endpoint(request: Request, body: HistoryRequest, user: dict = Depends(require_patient_or_doctor)):
    if user.get("role") == "patient" and user.get("patient_id") != body.patient_id:
        raise HTTPException(status_code=403, detail="Access denied")
    if hidden_from_doctor(user, patient_id=body.patient_id):
        return {"history": []}
    history = get_patient_history(body.patient_id)
    return {"history": history}

@app.post("/get_symptoms")
@limiter.limit("120/minute")
def get_symptoms_endpoint(request: Request, body: SymptomsRequest, user: dict = Depends(require_patient_or_doctor)):
    if user.get("role") == "patient" and user.get("patient_id") != body.patient_id:
        raise HTTPException(status_code=403, detail="Access denied")
    if hidden_from_doctor(user, patient_id=body.patient_id):
        get_symptom_graph(body.patient_id, body.symptom_str)  # same validation errors as for a real patient
        return {"symptoms_arr": []}
    graph = get_symptom_graph(body.patient_id, body.symptom_str)
    return {"symptoms_arr": graph}

def _clean(evidences: List[str], where: str) -> List[str]:
    valid, invalid = validate_evidences(list(dict.fromkeys(evidences)))
    if invalid:
        logger.warning("%s: dropped %d unknown evidence tokens", where, len(invalid))
    return valid


def _review_summary(review: Optional[dict]) -> dict:
    """Safety review for the logs. available=False means the review call failed
    (e.g. rate-limited), so no out-of-scope / emergency signal reached triage."""
    if not review:
        return {"available": False}
    summary = {
        "available": True,
        "model": review.get("model"),
        "emergency": review.get("emergency"),
        "verified": review.get("emergency_verified"),
        "out_of_scope": review.get("out_of_scope"),
        "likely_condition": review.get("likely_condition"),
    }
    if review.get("emergency_probability") is not None:
        summary["emergency_probability"] = review["emergency_probability"]
    if plog.patient_text_allowed():
        summary["reason"] = review.get("reason")  # LLM's summary of the complaint
    return summary


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


def analyse_and_store(patient_id: int, evidences: List[str], absent: List[str], unknown: List[str],
                      age: Optional[int], sex: Optional[str], intake_id: Optional[int],
                      review: Optional[dict], created_at=None,
                      diagnose_setup: Optional[str] = None) -> dict:
    """Model + triage + storage for one already-validated set of answers; returns the /analys
    response. Seeding the demo cases calls this too, so seeded days take the production path.
    created_at backdates the diary day (None: the database default, now)."""
    notes: List[str] = []
    distribution = predict_distribution(evidences, age, sex, absent, notes=notes)
    top3 = list(distribution.items())[:3]
    top1, top1_score = top3[0]
    # Below the question engine's stop threshold the top-1 is much less reliable
    # (realistic simulation: 10% of patients, 54.9% top-1 vs 94.1% for the rest).
    low_confidence = top1_score < qa_engine_module.STOP_CONFIDENCE
    assessed = triage.assess(distribution, evidences, review)
    out_of_scope = bool(review and review.get("out_of_scope"))
    likely_condition = review.get("likely_condition") if out_of_scope else None

    day = insert_disease(
        patient_id,
        evidences,
        top1,
        top1_score,
        diagnose_setup,
        absent=absent,
        zone=assessed.zone,
        analysis={
            "top3": [[name, round(p, 4)] for name, p in top3],
            "reasons": assessed.reasons,
            "red_mass": round(assessed.red_mass, 4),
            "yellow_mass": round(assessed.yellow_mass, 4),
            "age": age,
            "sex": sex,
            "out_of_scope": out_of_scope,
            "likely_condition": likely_condition,
            "unknown": unknown,  # "don't know" answers: not evidence, kept for analysis
            "model_version": MODEL_VERSION,
            "low_confidence": low_confidence,
        },
        intake_id=intake_id,
        created_at=created_at,
    )
    plog.log_stage(
        "analysis", intake=intake_id, day=day, patient=patient_id,
        present=plog.labelled(evidences), absent=plog.labelled(absent), unknown=plog.labelled(unknown),
        age=age, sex=sex,
        top=plog.top(distribution), zone=assessed.zone, reasons=assessed.reasons, low_confidence=low_confidence,
        red_mass=round(assessed.red_mass, 3), yellow_mass=round(assessed.yellow_mass, 3),
        review=_review_summary(review), adjustments=notes,
        # P(answer | disease) for the top 3: at the 0.03 floor = never seen with that disease
        fit=evidence_fit(evidences, [name for name, _ in top3]),
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
        "low_confidence": low_confidence,
        "zone": assessed.zone,
        "reasons": assessed.reasons,
        "out_of_scope": {"likely_condition": likely_condition} if out_of_scope else None,
    }


@app.post("/analys")
@limiter.limit("30/minute")
def analys_endpoint(request: Request, body: AnalysRequest, user: dict = Depends(require_patient)):
    patient_id = user["patient_id"]
    evidences = _clean(body.evidences, "analys")
    absent = [e for e in _clean(body.absent, "analys") if e not in set(evidences)]
    unknown = [e for e in _clean(body.unknown, "analys") if e not in set(evidences)]
    if not evidences:
        raise HTTPException(status_code=422, detail="No valid evidences")

    review = None
    if body.intake_id is not None:
        intake = get_intake(body.intake_id, patient_id)
        if intake is None:
            raise HTTPException(status_code=403, detail="Access denied")
        review = intake["result"].get("review")

    return analyse_and_store(patient_id, evidences, absent, unknown, body.age, body.sex,
                             body.intake_id, review, diagnose_setup=body.diagnose_setup)


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
    started = time.perf_counter()
    result = (await extraction.extract_complaint(body.text)).to_dict()
    intake_id = await run_in_threadpool(insert_intake, user["patient_id"], body.text, result)

    review = result.get("review") or {}
    alert = triage.critical_alert(result["evidences"])
    if alert is None and review.get("emergency"):
        alert = {
            "level": "red" if review.get("emergency_verified") else "yellow",
            "code": "llm_emergency",
            "message": "⚠️ Ваше описание похоже на неотложное состояние. "
                       "Если вам становится хуже — немедленно вызовите скорую помощь (103).",
        }

    with_text = plog.patient_text_allowed()
    plog.log_stage(
        "extract", intake=intake_id, patient=user["patient_id"], lang=plog.detect_language(body.text),
        chars=len(body.text), backend=result["mode"], model=result["model"],
        ms=round((time.perf_counter() - started) * 1000), tokens=result["usage"],
        findings=[
            {"code": f["code"], "label": plog.label(f["code"]), "status": f["status"], "source": f["source"],
             **({"confidence": f["confidence"]} if f.get("confidence") is not None else {}),
             **({"quote": f["quote"]} if with_text and f.get("quote") else {})}
            for f in result["findings"]
        ],
        age=result["age"], sex=result["sex"], review=_review_summary(result.get("review")),
        alert=alert["level"] if alert else None,
        **({"text": body.text} if with_text else {}),
    )

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
def next_questions_endpoint(
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
    unknown = [e for e in _clean(body.unknown, "next_questions") if e not in set(evidences)]
    if not evidences:
        raise HTTPException(status_code=422, detail="No valid evidences")

    result = qa_engine.get_questions(
        evidences=evidences,
        age=body.age,
        sex=body.sex,
        round_num=body.round_num,
        n_questions=body.n,
        absent=absent,
        unknown=unknown,
    )
    distribution = predict_distribution(evidences, body.age, body.sex, absent)
    zone = triage.assess(distribution, evidences).zone
    plog.log_stage(
        "questions", intake=body.intake_id, round=body.round_num,
        present=plog.labelled(evidences), absent=plog.labelled(absent), unknown=plog.labelled(unknown),
        age=body.age, sex=body.sex,
        top=plog.top(distribution), confidence=round(result.confidence, 3),
        stop=result.stop_reason if result.should_stop else None,
        asked=[[q.evidence_id, plog.label(q.evidence_id), round(q.score, 3)] for q in result.questions],
        zone=zone,
    )

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
        "zone": zone,
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


def _explanation_facts(day: Dict[str, Any]) -> Tuple[str, Dict[str, list]]:
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
    return "\n".join(lines), attributions


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
    day = await run_in_threadpool(get_day_for_explanation, body.day_id, patient_id)
    if day is None:
        raise HTTPException(status_code=403, detail="Access denied")

    lang = _LANG_NAMES[body.lang]
    facts, attributions = await run_in_threadpool(_explanation_facts, day)
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
    await run_in_threadpool(save_explanations, body.day_id, patient_id, patient_text, doctor_text)
    plog.log_stage(
        "explain", day=body.day_id, patient=patient_id, lang=body.lang,
        patient_model=patient_result.model if patient_result else "fallback",
        doctor_model=doctor_result.model if doctor_result else "fallback",
        tokens={"input": sum(r.input_tokens for r in (patient_result, doctor_result) if r),
                "output": sum(r.output_tokens for r in (patient_result, doctor_result) if r)},
        # leave-one-out effect of each answer on each top disease, as given to the LLM
        attributions={d: [[ev, round(delta, 3)] for ev, delta in pairs] for d, pairs in attributions.items()},
    )
    return {"patientExplanation": patient_text, "doctorExplanation": doctor_text}

# ─── Doctor-only endpoints ───

@app.post("/update_by_doctor")
@limiter.limit("60/minute")
def update_by_doctor_endpoint(request: Request, body: UpdateByDoctorRequest, user: dict = Depends(require_doctor)):
    doctor_id = user.get("doctor_id")
    if hidden_from_doctor(user, patient_id=body.patient_id, day_id=body.day_id):
        return {"ok": False}
    ok = doctor_update_day(
        body.patient_id,
        body.day_id,
        doctor_id,
        body.recept,
        body.disease_setup,
        doctor_id,
    )
    return {"ok": ok}

@app.post("/doctor_feedback")
@limiter.limit("60/minute")
def doctor_feedback_endpoint(request: Request, body: DoctorFeedbackRequest, user: dict = Depends(require_doctor)):
    """The doctor's confirmed diagnosis for a diary day, kept next to the model's answer."""
    result = None if hidden_from_doctor(user, day_id=body.day_id) else insert_feedback(body.day_id, user.get("doctor_id"), body)
    if result is None:
        raise HTTPException(status_code=404, detail="Day not found")
    return {"ok": True, **result}

@app.post("/list_patients_triage")
@limiter.limit("120/minute")
def list_patients_triage_endpoint(request: Request, user: dict = Depends(require_doctor)):
    patients = get_all_patients_triage(bool(user.get("demo")))
    return {"patients": patients}

@app.get("/feedback_stats")
@limiter.limit("60/minute")
def feedback_stats_endpoint(request: Request, user: dict = Depends(require_doctor)):
    """Real-world accuracy: doctor-confirmed diagnoses vs the model, per model version."""
    return get_feedback_stats()

@app.get("/accuracy_report")
@limiter.limit("60/minute")
def accuracy_report_endpoint(request: Request, user: dict = Depends(require_doctor)):
    """Offline benchmark snapshot (eval/accuracy_report.json) plus live real_patients numbers."""
    return get_accuracy_report()

@app.post("/get_patient_info")
@limiter.limit("120/minute")
def get_patient_info_endpoint(request: Request, body: PatientInfoRequest, user: dict = Depends(require_patient_or_doctor)):
    if user.get("role") == "patient" and user.get("patient_id") != body.patient_id:
        raise HTTPException(status_code=403, detail="Access denied")
    info = None if hidden_from_doctor(user, patient_id=body.patient_id) else select_patient(body.patient_id)
    if info is None:
        return {"found": False, "patient": None}
    return {"found": True, "patient": info}

@app.post("/get_day_symptoms")
@limiter.limit("120/minute")
def get_day_symptoms_endpoint(request: Request, body: DaySymptomsRequest, user: dict = Depends(require_patient_or_doctor)):
    if user.get("role") == "patient":
        if not day_belongs_to_patient(body.day_id, user["patient_id"]):
            raise HTTPException(status_code=403, detail="Access denied")
    if hidden_from_doctor(user, day_id=body.day_id):
        return {"symptoms": {}, "labels": {}}
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

def save_lab_result(patient_id: int, test_type: str, test_date: str, results: list,
                    interpretation: str, image_filename: str) -> int:
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO lab_results(patient_id, test_type, test_date, results_json, interpretation, image_filename)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING result_id
        """, (patient_id, test_type, test_date, encrypt_field(json.dumps(results, ensure_ascii=False)), encrypt_field(interpretation), image_filename))
        result_id = cur.fetchone()[0]
        conn.commit()
        return result_id
    finally:
        conn.close()


@app.post("/upload_lab_result")
@limiter.limit("5/minute")
async def upload_lab_result_endpoint(
    request: Request,  # noqa: used by slowapi limiter
    image: UploadFile = File(...),
    user: dict = Depends(require_patient),
):
    patient_id = user["patient_id"]

    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPG, PNG, or WebP.")

    # Read one byte past the limit: enough to reject, without buffering a huge upload.
    raw_bytes = await image.read(MAX_IMAGE_SIZE + 1)
    if len(raw_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB)")

    image_bytes = await run_in_threadpool(blur_pii_region, raw_bytes)

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

    result_id = await run_in_threadpool(
        save_lab_result, patient_id, test_type, test_date, results, interpretation, safe_filename,
    )

    return {
        "result_id": result_id,
        "test_type": test_type,
        "test_date": test_date,
        "results": results,
        "interpretation": interpretation,
    }


@app.post("/get_lab_results")
@limiter.limit("120/minute")
def get_lab_results_endpoint(request: Request, body: GetLabResultsRequest = None, user: dict = Depends(require_patient_or_doctor)):
    if user.get("role") == "patient":
        patient_id = user["patient_id"]
    else:
        if body is None or body.patient_id is None:
            raise HTTPException(status_code=400, detail="patient_id required for doctors")
        patient_id = body.patient_id
        if hidden_from_doctor(user, patient_id=patient_id):
            return {"results": []}

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