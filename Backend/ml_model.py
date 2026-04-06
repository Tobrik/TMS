"""
ML inference module for TMS backend.
Loads the trained RandomForest model from 'training sklearn/' and exposes
a single predict function used by the /analys endpoint.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger("tms")

# ─── Paths ───
_BASE = Path(__file__).parent
_TRAINING_DIR = _BASE / "training sklearn"

_MODEL_PATH    = _TRAINING_DIR / "model.pkl"
_FEATURES_PATH = _TRAINING_DIR / "feature_names.json"
_LABELS_PATH   = _TRAINING_DIR / "label_classes.json"
_EVIDENCES_PATH = _TRAINING_DIR / "release_evidences.json"

# ─── Load model and metadata at import time ───
try:
    import joblib
    _clf = joblib.load(_MODEL_PATH)
    logger.info("sklearn model loaded from %s", _MODEL_PATH)
except Exception as e:
    _clf = None
    logger.error("Failed to load sklearn model: %s", e)

try:
    with open(_FEATURES_PATH, encoding="utf-8") as f:
        FEATURE_NAMES: List[str] = json.load(f)
    _feat_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    N_FEATURES = len(FEATURE_NAMES)
    logger.info("Feature names loaded: %d features", N_FEATURES)
except Exception as e:
    FEATURE_NAMES = []
    _feat_idx = {}
    N_FEATURES = 0
    logger.error("Failed to load feature_names.json: %s", e)

try:
    with open(_LABELS_PATH, encoding="utf-8") as f:
        LABEL_CLASSES: List[str] = json.load(f)
    logger.info("Label classes loaded: %d classes", len(LABEL_CLASSES))
except Exception as e:
    LABEL_CLASSES = []
    logger.error("Failed to load label_classes.json: %s", e)

# ─── Load evidences metadata for NLP prompt building ───
try:
    with open(_EVIDENCES_PATH, encoding="utf-8") as f:
        EVIDENCES_META: dict = json.load(f)
    logger.info("Evidences metadata loaded: %d entries", len(EVIDENCES_META))
except Exception as e:
    EVIDENCES_META = {}
    logger.error("Failed to load release_evidences.json: %s", e)

def build_feature_vector(evidences: List[str], age: int = 25, sex: str = "M") -> np.ndarray:
    """
    Build a 1D numpy feature vector from a list of evidence token strings.

    Args:
        evidences: list of strings like ["E_91", "E_54_@_V_161", ...]
        age: patient age (0-120)
        sex: "M" or "F"

    Returns:
        numpy array of shape (N_FEATURES,)
    """
    vec = np.zeros(N_FEATURES, dtype=np.float32)

    if "__AGE__" in _feat_idx:
        vec[_feat_idx["__AGE__"]] = min(max(age, 0), 120) / 100.0

    if sex.upper() == "M" and "__SEX_M__" in _feat_idx:
        vec[_feat_idx["__SEX_M__"]] = 1.0

    for ev in evidences:
        if ev in _feat_idx:
            vec[_feat_idx[ev]] = 1.0

    return vec


# ─── Clinical syndrome grouping ───────────────────────────────────────────────
# DDXPlus splits clinically identical/overlapping syndromes into separate classes.
# We pool their probabilities so the model returns clinically meaningful results.
# The first disease in each list becomes the "canonical" name for the group.
CLINICAL_GROUPS: List[List[str]] = [
    # Acute Coronary Syndrome: НС и NSTEMI/STEMI неразличимы без тропонина
    # Клинически это один синдром (ОКС) — объединяем
    ["Possible NSTEMI / STEMI", "Unstable angina"],
]
# NOTE: We intentionally do NOT group URTI/Influenza/Pharyngitis —
# they have different treatments and the model should show them separately.
# We do NOT group PSVT/AF — different management (cardioversion vs rate control).
# Only merge diseases that are clinically INDISTINGUISHABLE without lab tests.

# Build lookup: disease → canonical group representative
_GROUP_MAP: dict = {}
for _group in CLINICAL_GROUPS:
    _canonical = _group[0]
    for _disease in _group:
        _GROUP_MAP[_disease] = _canonical

# ─── Geographic filter ───────────────────────────────────────────────────────
# Diseases essentially absent in CIS/Central Asia — suppress from final output
# (their probability mass is redistributed to the next most likely disease)
_GEO_EXCLUDED: set = {
    "Ebola",          # Sub-Saharan Africa only
    "Inguinal hernia", # Surgical — not a diagnostic differential
}


# ─── Rule-based exclusion dictionary ─────────────────────────────────────────
# Each rule: if ALL `require` evidences present AND NONE of `absent` evidences
# present → multiply listed diseases' probability by `penalty`.
_EXCLUSION_RULES: list = [
    {
        # Headache without chest symptoms → Boerhaave is nonsensical.
        # Boerhaave = esophageal rupture, requires chest/epigastric pain + vomiting.
        "require": {"E_55_@_V_89"},          # headache present
        "absent":  {"E_55_@_V_101", "E_14"}, # no chest pain
        "diseases": ["Boerhaave"],
        "penalty": 0.05,
    },
    {
        # Vomiting (E_211) without chest pain → Boerhaave extremely unlikely.
        "require": {"E_211"},
        "absent":  {"E_55_@_V_101", "E_14"},
        "diseases": ["Boerhaave"],
        "penalty": 0.10,
    },
    {
        # Sore throat without chest symptoms → not ACS.
        "require": {"E_97"},
        "absent":  {"E_14", "E_57"},          # no chest pain, no radiation
        "diseases": ["Possible NSTEMI / STEMI", "Unstable angina"],
        "penalty": 0.15,
    },
]

# ─── Bayesian age/sex priors ─────────────────────────────────────────────────
# Soft penalties for diseases epidemiologically near-impossible in a demographic.
# age_range is inclusive [lo, hi]. sex: "M"/"F"/None (None = any).
_AGE_SEX_PRIORS: list = [
    {
        # Boerhaave: median age ~60, exceedingly rare under 30
        "age_range": (0, 30),
        "sex": None,
        "diseases": ["Boerhaave"],
        "penalty": 0.10,
    },
    {
        # ACS (MI/unstable angina): very rare under 30 without risk factors
        "age_range": (0, 30),
        "sex": None,
        "diseases": ["Possible NSTEMI / STEMI", "Unstable angina"],
        "penalty": 0.20,
    },
    {
        # ACS in women under 40: even rarer
        "age_range": (0, 40),
        "sex": "F",
        "diseases": ["Possible NSTEMI / STEMI", "Unstable angina"],
        "penalty": 0.30,
    },
    {
        # Pulmonary embolism: rare under 20
        "age_range": (0, 20),
        "sex": None,
        "diseases": ["Pulmonary embolism"],
        "penalty": 0.25,
    },
    {
        # COPD: essentially absent under 35 without alpha-1 antitrypsin deficiency
        "age_range": (0, 35),
        "sex": None,
        "diseases": ["COPD"],
        "penalty": 0.10,
    },
]


def sklearn_predict(
    evidences: List[str],
    age: int = 25,
    sex: str = "M",
    top_n: int = 3,
) -> List[Tuple[str, float]]:
    """
    Predict top-N diagnoses from evidence list.
    Applies clinical grouping: probabilities of clinically equivalent
    diseases are pooled under a single canonical name.

    Returns:
        List of (disease_name, probability) tuples, sorted descending.
        Falls back to [("Unknown", 0.0)] if model not loaded.
    """
    if _clf is None or N_FEATURES == 0:
        logger.error("sklearn model not available for prediction")
        return [("Unknown", 0.0)]

    vec = build_feature_vector(evidences, age, sex)
    proba = _clf.predict_proba(vec.reshape(1, -1))[0]

    # Pool probabilities within clinical groups
    pooled: dict = {}
    for i, disease in enumerate(LABEL_CLASSES):
        canonical = _GROUP_MAP.get(disease, disease)
        pooled[canonical] = pooled.get(canonical, 0.0) + float(proba[i])

    # Apply geographic filter — skip diseases absent in CIS/Central Asia
    filtered = {d: p for d, p in pooled.items() if d not in _GEO_EXCLUDED}

    # ─── Clinical correction rules ────────────────────────────────────────
    ev_set = set(evidences)

    # Rule 1: Pleuritic pain (E_220) is AGAINST MI / ACS.
    if "E_220" in ev_set:
        for acs_name in ("Possible NSTEMI / STEMI", "Unstable angina"):
            if acs_name in filtered:
                filtered[acs_name] *= 0.4  # 60% penalty

    # Rule 2: GERD context (E_173 present) — penalize PE and pulmonary edema.
    if "E_173" in ev_set and "E_14" not in ev_set:
        for noise_name in ("Pulmonary embolism", "Acute pulmonary edema",
                           "Spontaneous pneumothorax"):
            if noise_name in filtered:
                filtered[noise_name] *= 0.3  # 70% penalty

    # ─── Rule-based exclusion dictionary ──────────────────────────────────
    # Each rule: if ALL `require` evidences are present AND NONE of `absent`
    # evidences are present → penalize listed diseases by `penalty` factor.
    for rule in _EXCLUSION_RULES:
        if rule["require"] <= ev_set and not (rule["absent"] & ev_set):
            for disease in rule["diseases"]:
                if disease in filtered:
                    filtered[disease] *= rule["penalty"]

    # ─── Bayesian age/sex priors ──────────────────────────────────────────
    # Penalize diseases that are epidemiologically near-impossible for the
    # patient's demographic. These are soft priors (penalties, not hard blocks).
    for prior in _AGE_SEX_PRIORS:
        age_lo, age_hi = prior["age_range"]
        sex_match = prior.get("sex")  # None = any sex
        if age_lo <= age <= age_hi and (sex_match is None or sex == sex_match):
            for disease in prior["diseases"]:
                if disease in filtered:
                    filtered[disease] *= prior["penalty"]

    # Renormalize after adjustments
    total = sum(filtered.values())
    if total > 0:
        filtered = {d: p / total for d, p in filtered.items()}

    sorted_results = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]


def validate_evidences(evidences: List[str]) -> Tuple[List[str], List[str]]:
    """
    Filter evidence list to only valid feature tokens.
    Returns (valid, invalid).
    """
    valid = [e for e in evidences if e in _feat_idx]
    invalid = [e for e in evidences if e not in _feat_idx]
    return valid, invalid


# ─── Disease → relevant evidences mapping ────────────────────────────────────
# Precomputed from RF: for each disease, the top-K binary evidences whose
# presence causes the largest increase in P(disease).
# Built once at import time; used to filter candidates in follow-up questions.
_DISEASE_RELEVANT_EVIDENCES: dict = {}  # disease_name → set of evidence IDs
_RELEVANCE_TOP_K = 30  # keep top-30 evidences per disease


def _build_disease_evidence_map() -> dict:
    """
    For each disease class, probe the RF model to find which binary evidences
    are most relevant (largest probability increase when feature is turned on).
    Returns: {disease_name: set(ev_id, ...)}.
    """
    if _clf is None or N_FEATURES == 0:
        return {}

    # Identify binary evidence features (exclude __AGE__, __SEX_M__, categorical)
    binary_evs = [
        ev_id for ev_id, meta in EVIDENCES_META.items()
        if meta.get("data_type") == "B"
        and ev_id in _feat_idx
    ]
    if not binary_evs:
        return {}

    # Base prediction on empty patient (only default age/sex)
    base_vec = build_feature_vector([], age=25, sex="M")
    base_proba = _clf.predict_proba(base_vec.reshape(1, -1))[0]

    # Batch: one row per binary evidence with that feature set to 1
    batch = np.tile(base_vec, (len(binary_evs), 1))
    for i, ev_id in enumerate(binary_evs):
        batch[i, _feat_idx[ev_id]] = 1.0
    probas = _clf.predict_proba(batch)  # (n_binary, n_classes)

    # For each disease, collect (ev_id, delta_prob) and keep top-K
    result = {}
    for cls_idx, disease in enumerate(LABEL_CLASSES):
        base_p = float(base_proba[cls_idx])
        deltas = []
        for i, ev_id in enumerate(binary_evs):
            delta = float(probas[i][cls_idx]) - base_p
            if delta > 0.001:  # only positive shifts
                deltas.append((ev_id, delta))
        deltas.sort(key=lambda x: x[1], reverse=True)
        result[disease] = {ev_id for ev_id, _ in deltas[:_RELEVANCE_TOP_K]}

    logger.info(
        "disease→evidence map built: %d diseases, avg %.0f evidences each",
        len(result),
        sum(len(v) for v in result.values()) / max(len(result), 1),
    )
    return result


try:
    _DISEASE_RELEVANT_EVIDENCES = _build_disease_evidence_map()
except Exception as e:
    logger.error("Failed to build disease→evidence map: %s", e)


# ─── Prerequisite logic for dependent evidences ─────────────────────────────
# E_55 (pain location) and E_57 (pain radiation) are child evidences of E_53
# (pain present). E_54 (pain character) also depends on E_53.
# Don't ask these until the parent is confirmed.
_EVIDENCE_PREREQUISITES: dict = {
    # child_evidence_prefix → parent evidence that must be present
    "E_55": "E_53",   # pain location requires pain present
    "E_57": "E_53",   # pain radiation requires pain present
    "E_54": "E_53",   # pain character requires pain present
}


# ─── Clinical filter: evidences that should never be asked as follow-ups ────
_NEVER_ASK = {
    "E_211",  # repeated vomiting — only from patient text, not as question
}


def find_discriminative_evidences(
    evidences: List[str],
    age: int = 25,
    sex: str = "M",
    top_n: int = 3,
    min_shift: float = 0.01,
    confidence_threshold: float = 0.60,
) -> List[dict]:
    """
    Find the top_n most discriminative binary evidences to ask about next.

    Filtering pipeline:
      1. Only binary evidences not yet known
      2. Prerequisite check: skip child evidences (E_55, E_57, E_54) if parent
         (E_53) is not yet confirmed
      3. Relevance filter: only consider evidences relevant to current top-3
         diagnoses (from precomputed disease→evidence map)
      4. Score by probability shift on top-1 disease

    If the top-1 diagnosis already has confidence >= confidence_threshold,
    no follow-up questions needed.

    Returns list of {evidence_id, question_en, question_ru, question_kk, score}.
    """
    if _clf is None or N_FEATURES == 0:
        return []

    base_vec = build_feature_vector(evidences, age, sex)
    base_proba = _clf.predict_proba(base_vec.reshape(1, -1))[0]
    top3_idx = np.argsort(base_proba)[::-1][:3]

    # If top-1 confidence is already high, no clarification needed
    top1_conf = float(base_proba[top3_idx[0]])
    if top1_conf >= confidence_threshold:
        logger.info(
            "skip follow-up questions: top-1 confidence %.2f >= %.2f",
            top1_conf, confidence_threshold,
        )
        return []

    ev_set = set(evidences)

    # Step 1: Only consider binary evidences not yet known
    candidates = [
        ev_id for ev_id, meta in EVIDENCES_META.items()
        if meta.get("data_type") == "B"
        and ev_id not in ev_set
        and ev_id in _feat_idx
        and not meta.get("is_antecedent", False)
        and ev_id not in _NEVER_ASK
    ]

    # Step 2: Prerequisite check — skip children whose parent is absent
    def _prereq_ok(ev_id: str) -> bool:
        for prefix, parent in _EVIDENCE_PREREQUISITES.items():
            if ev_id == parent:
                return True  # parent itself is always ok to ask
            if ev_id.startswith(prefix) and parent not in ev_set:
                return False
        return True

    candidates = [c for c in candidates if _prereq_ok(c)]

    # Step 3: Relevance filter — only evidences relevant to top-3 diagnoses
    if _DISEASE_RELEVANT_EVIDENCES:
        top3_names = [LABEL_CLASSES[i] for i in top3_idx]
        relevant = set()
        for d_name in top3_names:
            relevant |= _DISEASE_RELEVANT_EVIDENCES.get(d_name, set())
            # Also check canonical group name
            canonical = _GROUP_MAP.get(d_name, d_name)
            if canonical != d_name:
                relevant |= _DISEASE_RELEVANT_EVIDENCES.get(canonical, set())
        candidates = [c for c in candidates if c in relevant]
        logger.info(
            "relevance filter: %d candidates for top-3 %s",
            len(candidates),
            [f"{LABEL_CLASSES[i]}({base_proba[i]:.2f})" for i in top3_idx],
        )

    if not candidates:
        return []

    # Batch predict: one row per candidate with that feature set to 1
    batch = np.tile(base_vec, (len(candidates), 1))
    for i, ev_id in enumerate(candidates):
        batch[i, _feat_idx[ev_id]] = 1.0

    probas = _clf.predict_proba(batch)  # (n_candidates, n_classes)

    # Score by how much each candidate shifts the top-1 diagnosis probability
    top1_idx = top3_idx[0]
    base_top1_p = base_proba[top1_idx]
    scores = []
    for i, ev_id in enumerate(candidates):
        shift = abs(float(probas[i][top1_idx]) - base_top1_p)
        if shift >= min_shift:
            scores.append((ev_id, shift))

    scores.sort(key=lambda x: x[1], reverse=True)

    result = []
    for ev_id, score in scores[:top_n]:
        meta = EVIDENCES_META.get(ev_id, {})
        q_en = meta.get("question_en", ev_id)
        result.append({
            "evidence_id": ev_id,
            "question_en": q_en,
            "question_ru": _QUESTION_TRANSLATIONS.get(ev_id, {}).get("ru", q_en),
            "question_kk": _QUESTION_TRANSLATIONS.get(ev_id, {}).get("kk", q_en),
            "score": score,
        })

    return result


# ─── Manual translations for the most common diagnostic questions ───
# Covers the ~60 most frequently selected discriminative evidences
_QUESTION_TRANSLATIONS: dict = {
    "E_91": {"ru": "У вас есть температура (ощущение жара или измеренная термометром)?", "kk": "Сізде қызба бар ма (қыздырып тұру немесе термометрмен өлшенген)?"},
    "E_94": {"ru": "У вас был озноб или дрожь?", "kk": "Сізде қалтырау немесе дірілдеу болды ма?"},
    "E_97": {"ru": "У вас болит горло?", "kk": "Сіздің тамағыңыз ауырады ма?"},
    "E_201": {"ru": "У вас есть кашель?", "kk": "Сізде жөтел бар ма?"},
    "E_181": {"ru": "У вас заложен нос или прозрачные выделения из носа?", "kk": "Сіздің мұрныңыз бітелген немесе мөлдір бөліндіге бар ма?"},
    "E_66": {"ru": "У вас значительная одышка или затруднение дыхания?", "kk": "Сізде айтарлықтай ентігу немесе дем алу қиындығы бар ма?"},
    "E_14": {"ru": "У вас есть боль в груди даже в состоянии покоя?", "kk": "Сізде тыныштық жағдайда да кеуде ауруы бар ма?"},
    "E_53": {"ru": "У вас есть боль, связанная с причиной вашего обращения?", "kk": "Сіздің шағымыңызбен байланысты ауырсыну бар ма?"},
    "E_57": {"ru": "Боль распространяется в другое место (например, в руку, шею, челюсть)?", "kk": "Ауырсыну басқа жерге (мысалы, қолға, мойынға, жаққа) тарайды ма?"},
    "E_50": {"ru": "У вас значительно усилилось потоотделение?", "kk": "Сізде тер шығару айтарлықтай күшейді ме?"},
    "E_155": {"ru": "Вы чувствуете сильное сердцебиение или перебои в работе сердца?", "kk": "Сіз жүрек соғысының жылдамдауын немесе үзілісін сезесіз бе?"},
    "E_148": {"ru": "Вас тошнит или есть ощущение, что вас вырвет?", "kk": "Сізде жүрек айну немесе құсу сезімі бар ма?"},
    "E_51": {"ru": "У вас диарея или учащённый стул?", "kk": "Сізде диарея немесе жиі нәжіс бар ма?"},
    "E_89": {"ru": "Вы постоянно чувствуете усталость или у вас нарушен сон?", "kk": "Сіз үнемі шаршауды сезесіз бе немесе ұйқыңыз бұзылды ма?"},
    "E_88": {"ru": "Вы настолько устали, что не можете вести привычный образ жизни или лежите в постели весь день?", "kk": "Сіз соншалықты шаршадыңыз ба, күнделікті іспен айналыса алмайсыз немесе күні бойы жатасыз?"},
    "E_175": {"ru": "Вы заметили общую слабость, недомогание или изменение общего самочувствия?", "kk": "Жалпы әлсіздік, нашарлау немесе жалпы жағдайдың өзгеруін байқадыңыз ба?"},
    "E_144": {"ru": "У вас распространённая боль в мышцах?", "kk": "Сізде кең таралған бұлшықет ауруы бар ма?"},
    "E_77": {"ru": "У вас кашель с окрашенной или более обильной, чем обычно, мокротой?", "kk": "Сізде түрлі-түсті немесе әдеттегіден көп қақырықпен жөтел бар ма?"},
    "E_220": {"ru": "Боль усиливается при глубоком вдохе?", "kk": "Терең дем алғанда ауырсыну күшейе ме?"},
    "E_9": {"ru": "У вас опухшие или болезненные лимфоузлы?", "kk": "Сізде ісінген немесе ауырсынатын лимфа түйіндері бар ма?"},
    "E_65": {"ru": "Вам трудно глотать или есть дискомфорт при глотании?", "kk": "Сіздің жұтуыңыз қиынға соғады ма немесе жұтқанда ыңғайсыздық бар ма?"},
    "E_173": {"ru": "У вас есть жжение, которое поднимается из желудка в горло с кисловатым привкусом?", "kk": "Сізде асқазаннан тамаққа қышқыл дәммен бірге жоғары көтерілетін жану сезімі бар ма?"},
    "E_30": {"ru": "Ваш живот вздут или раздут?", "kk": "Сіздің қарныңыз кеуіп немесе ісіп тұр ма?"},
    "E_64": {"ru": "Вы задыхаетесь при минимальной физической нагрузке?", "kk": "Ең аз дене жаттығуынан кейін ентігесіз бе?"},
    "E_129": {"ru": "У вас есть высыпания, покраснение или проблемы с кожей?", "kk": "Сізде бөртпе, қызару немесе тері мәселелері бар ма?"},
    "E_82": {"ru": "У вас кружится голова или вы чувствуете, что можете упасть в обморок?", "kk": "Сіздің басыңыз айналады ма немесе естен тануға жақын сезінесіз бе?"},
    "E_164": {"ru": "Вы чувствуете, что ваше сердце бьётся очень нерегулярно или хаотично?", "kk": "Сіздің жүрегіңіз өте тұрақсыз немесе хаотты соғып тұрғанын сезесіз бе?"},
    "E_124": {"ru": "У вас есть астма или вы когда-либо использовали бронходилататор?", "kk": "Сізде демікпе бар ма немесе бұрын бронходилататор қолдандыңыз ба?"},
    "E_214": {"ru": "Вы замечаете свистящий звук при выдохе?", "kk": "Ыдырау кезінде ысқырған дыбыс байқайсыз ба?"},
    "E_211": {"ru": "Вас несколько раз вырвало или были многократные позывы к рвоте?", "kk": "Сіздің бірнеше рет қусы ма немесе бірнеше рет құсуға ынтызарлық болды ма?"},
    "E_154": {"ru": "Ваша кожа значительно бледнее, чем обычно?", "kk": "Сіздің терінгіз әдеттегіден айтарлықтай бозарып кетті ме?"},
    "E_76": {"ru": "Вы чувствуете лёгкое головокружение или неустойчивость?", "kk": "Сіз жеңіл бас айналуды немесе тұрақсыздықты сезесіз бе?"},
    "E_13": {"ru": "Симптомы ухудшились за последние 2 недели и для их появления требуется всё меньше усилий?", "kk": "Соңғы 2 аптада белгілер нашарлады ма және олардың пайда болуы үшін күш азая ма?"},
    "E_41": {"ru": "Вы контактировали с человеком со схожими симптомами за последние 2 недели?", "kk": "Соңғы 2 аптада ұқсас белгілері бар адаммен байланыста болдыңыз ба?"},
    "E_116": {"ru": "У вас была простуда за последние 2 недели?", "kk": "Соңғы 2 аптада суықтадыңыз ба?"},
    "E_105": {"ru": "У вас когда-либо был сердечный приступ или стенокардия (боль в груди)?", "kk": "Сізде бұрын инфаркт немесе стенокардия (кеуде ауруы) болды ма?"},
    "E_22": {"ru": "У вас диагностирована проблема с сердечным клапаном?", "kk": "Сізде жүрек клапанының мәселесі диагноз қойылды ма?"},
    "E_139": {"ru": "У вас врождённый порок сердца?", "kk": "Сізде туа біткен жүрек ақауы бар ма?"},
    "E_106": {"ru": "У вас сердечная недостаточность?", "kk": "Сізде жүрек жеткіліксіздігі бар ма?"},
    "E_64": {"ru": "Вы задыхаетесь при минимальной физической нагрузке?", "kk": "Ең аз күш жұмсаған кезде ентігесіз бе?"},
    "E_67": {"ru": "У вас бывают приступы удушья или одышки, которые будят вас ночью?", "kk": "Түнде ұйқыдан оятатын тұншығу немесе ентігу ұстамалары болады ма?"},
    "E_33": {"ru": "Боль уменьшается, когда вы наклоняетесь вперёд?", "kk": "Алға еңкейгенде ауырсыну азаяды ма?"},
    "E_128": {"ru": "Вы когда-либо чувствовали, что задыхаетесь — кратковременно не могли дышать или говорить?", "kk": "Сіз қысқа уақытқа тыныс ала алмай немесе сөйлей алмай қалдыңыз ба?"},
    "E_216": {"ru": "Боль усиливается при движении?", "kk": "Қозғалыс кезінде ауырсыну күшейе ме?"},
    "E_30": {"ru": "Ваш живот вздут или распирает изнутри?", "kk": "Сіздің қарныңыз ішінен қысым сезіп кеуіп тұр ма?"},
    "E_129": {"ru": "У вас есть высыпания, покраснение или проблемы с кожей?", "kk": "Сізде бөртпе, қызару немесе тері мәселелері бар ма?"},
    "E_169": {"ru": "У вас зудит нос или задняя стенка горла?", "kk": "Сіздің мұрныңыз немесе тамағыңыздың артқы қабырғасы қышиды ма?"},
    "E_170": {"ru": "У вас сильный зуд в одном или обоих глазах?", "kk": "Бір немесе екі көзіңіз қатты қышиды ма?"},
    "E_182": {"ru": "У вас зелёные или жёлтые выделения из носа?", "kk": "Сізде жасыл немесе сары мұрын бөліндісі бар ма?"},
    "E_121": {"ru": "У вас искривлённая носовая перегородка?", "kk": "Сіздің мұрын қалқасы қисайған ба?"},
    "E_86": {"ru": "Есть ли у ваших близких родственников аллергия, сенная лихорадка или экзема?", "kk": "Жақын туыстарыңызда аллергия, шөп безгегі немесе экзема бар ма?"},
    "E_45": {"ru": "Вы кашляете кровью?", "kk": "Сіз қан жөтелесіз бе?"},
    "E_203": {"ru": "У вас сильные приступы кашля?", "kk": "Сізде күшті жөтел ұстамалары бар ма?"},
    "E_202": {"ru": "У пациента коклюш (спастический кашель)?", "kk": "Пациентте күкірт жөтел бар ма?"},
    "E_40": {"ru": "Вы контактировали с больным коклюшем?", "kk": "Сіз күкірт жөтелімен ауырған адаммен байланыста болдыңыз ба?"},
    "E_112": {"ru": "При вдохе у вас свистящее или шумное дыхание после приступов кашля?", "kk": "Жөтел ұстамасынан кейін дем алғанда ысқырған немесе шулы тыныс алу бар ма?"},
    "E_194": {"ru": "Вы замечаете высокочастотный звук при вдохе?", "kk": "Дем алғанда жоғары жиілікті дыбыс байқайсыз ба?"},
    # ── Additional translations (previously shown in English) ─────────────
    "E_23": {"ru": "Вы когда-нибудь временно переставали дышать во сне?", "kk": "Ұйқыда уақытша тыныс алуыңыз тоқтады ма?"},
    "E_32": {"ru": "У вас снижен аппетит?", "kk": "Тәбетіңіз төмендеді ме?"},
    "E_38": {"ru": "У вас есть боль или слабость в челюсти?", "kk": "Жағыңызда ауырсыну немесе әлсіздік бар ма?"},
    "E_39": {"ru": "Вы чувствовали спутанность сознания или дезориентацию в последнее время?", "kk": "Соңғы кезде сананың шатасуын немесе бағдарсыздықты сездіңіз бе?"},
    "E_42": {"ru": "Вы контактировали с чем-то или съели что-то, на что у вас аллергия?", "kk": "Аллергия тудыратын нәрсемен байланыста болдыңыз ба немесе жедіңіз бе?"},
    "E_43": {"ru": "Вы теряли сознание с судорогами или у вас были эпизоды «отключения»?", "kk": "Құрысу ұстамаларымен есіңізден таңдыңыз ба немесе «өшу» эпизодтары болды ма?"},
    "E_52": {"ru": "Вы видите двоение — два изображения одного предмета?", "kk": "Сіз бір заттың екі бейнесін көресіз бе (қос көру)?"},
    "E_63": {"ru": "Вам трудно выговаривать слова или говорить?", "kk": "Сіздің сөздерді айту немесе сөйлеу қиынға соғады ма?"},
    "E_74": {"ru": "У вас покраснение одного или обоих глаз?", "kk": "Бір немесе екі көзіңіз қызарып тұр ма?"},
    "E_75": {"ru": "Вы чувствуете (или чувствовали), что задыхаетесь?", "kk": "Сіз тұншығып жатқаныңызды сезесіз бе (немесе сездіңіз бе)?"},
    "E_83": {"ru": "Вы заметили слабость мышц лица и/или глаз?", "kk": "Бет бұлшықеттері және/немесе көз әлсіздігін байқадыңыз ба?"},
    "E_84": {"ru": "Вы чувствуете слабость в обеих руках и/или обеих ногах?", "kk": "Екі қолда және/немесе екі аяқта әлсіздік сезесіз бе?"},
    "E_90": {"ru": "Мышечная слабость усиливается при усталости и/или стрессе?", "kk": "Бұлшықет әлсіздігі шаршау және/немесе стресс кезінде күшейе ме?"},
    "E_92": {"ru": "У вас внезапно покраснели щёки?", "kk": "Бетіңіз кенет қызарып кетті ме?"},
    "E_93": {"ru": "У вас онемение, потеря чувствительности или покалывание в ступнях?", "kk": "Аяқ ұштарыңызда жансыздану, сезімталдықтың жоғалуы немесе шаншу бар ма?"},
    "E_96": {"ru": "Вы недавно набрали вес?", "kk": "Соңғы кезде салмақ қостыңыз ба?"},
    "E_103": {"ru": "Вы потеряли обоняние (чувство запаха)?", "kk": "Иіс сезу қабілетіңіз жоғалды ма?"},
    "E_111": {"ru": "Вы чувствуете, что умираете, или испытываете сильный страх смерти?", "kk": "Сіз өліп жатқандай сезінесіз бе немесе өлім қорқынышын сезесіз бе?"},
    "E_114": {"ru": "Вы стали более раздражительны или ваше настроение очень нестабильно?", "kk": "Сіз тітіркенгіш болдыңыз ба немесе көңіл-күйіңіз тұрақсыз ба?"},
    "E_127": {"ru": "Ваши глаза слезятся больше обычного?", "kk": "Көздеріңіз әдеттегіден көп жасаурайды ма?"},
    "E_140": {"ru": "У вас недавно был чёрный стул (как уголь)?", "kk": "Жақында көмірдей қара нәжіс болды ма?"},
    "E_145": {"ru": "У вас очень обильные или длительные менструации?", "kk": "Сізде өте мол немесе ұзақ етеккір бар ма?"},
    "E_150": {"ru": "Вы могли ходить в туалет (стул или газы) после начала симптомов?", "kk": "Белгілер басталғаннан кейін дәретке немесе газ шығара алдыңыз ба?"},
    "E_151": {"ru": "У вас есть отёк в одной или нескольких частях тела?", "kk": "Дененің бір немесе бірнеше бөлігінде ісіну бар ма?"},
    "E_156": {"ru": "У вас была слабость или паралич одной стороны лица?", "kk": "Беттің бір жағында әлсіздік немесе сал болды ма?"},
    "E_157": {"ru": "У вас недавно было онемение или покалывание в руках, ногах и вокруг рта?", "kk": "Қолдарда, аяқтарда және ауыз айналасында жансыздану немесе шаншу болды ма?"},
    "E_159": {"ru": "Вы теряли сознание?", "kk": "Сіз есіңізден таңдыңыз ба?"},
    "E_161": {"ru": "У вас снижен аппетит или вы наедаетесь быстрее обычного?", "kk": "Тәбетіңіз төмендеді ме немесе әдеттегіден тез тоясыз ба?"},
    "E_162": {"ru": "У вас была непреднамеренная потеря веса за последние 3 месяца?", "kk": "Соңғы 3 айда байқамай салмақ жоғалтуыңыз болды ма?"},
    "E_163": {"ru": "У вас есть вагинальные выделения?", "kk": "Сізде вагинальді бөлініс бар ма?"},
    "E_166": {"ru": "Вас рвало после кашля?", "kk": "Жөтелден кейін құстыңыз ба?"},
    "E_168": {"ru": "Вам трудно удерживать язык во рту?", "kk": "Тіліңізді ауызда ұстау қиынға соғады ма?"},
    "E_171": {"ru": "Вы чувствуете отстранённость от своего тела или окружающего мира?", "kk": "Өз денеңізден немесе айналаңыздан алшақтағанды сезесіз бе?"},
    "E_172": {"ru": "Вам трудно открыть или поднять одно или оба века?", "kk": "Бір немесе екі қабақты ашу/көтеру қиынға соғады ма?"},
    "E_174": {"ru": "Вы непреднамеренно теряете вес или у вас пропал аппетит?", "kk": "Байқамай салмақ жоғалтасыз ба немесе тәбетіңіз жоғалды ма?"},
    "E_176": {"ru": "У вас была или есть слабость/паралич в руках, ногах или лице?", "kk": "Қолдарда, аяқтарда немесе бетте әлсіздік/сал болды ма немесе бар ма?"},
    "E_177": {"ru": "У вас есть или было онемение, потеря чувствительности или покалывание?", "kk": "Денеңізде жансыздану, сезімталдықтың жоғалуы немесе шаншу бар ма?"},
    "E_178": {"ru": "Вы заметили необычное кровотечение или синяки?", "kk": "Әдеттен тыс қан кету немесе көгерулер байқадыңыз ба?"},
    "E_179": {"ru": "Вы замечали алую кровь или сгустки крови в стуле?", "kk": "Нәжісте ашық қан немесе қан ұйынділарын байқадыңыз ба?"},
    "E_180": {"ru": "Вы не можете контролировать направление взгляда?", "kk": "Көз қарасыңыздың бағытын бақылай алмайсыз ба?"},
    "E_188": {"ru": "У вас светлый стул и тёмная моча?", "kk": "Сізде ақшыл нәжіс және қоңыр зәр бар ма?"},
    "E_190": {"ru": "Вы заметили, что у вас выделяется больше слюны, чем обычно?", "kk": "Әдеттегіден көп сілекей бөлінетінін байқадыңыз ба?"},
    "E_192": {"ru": "У вас спазм или боль в мышцах шеи, мешающие повернуть голову?", "kk": "Мойын бұлшықеттерінде спазм немесе ауырсыну бас бұруға кедергі келтіреді ме?"},
    "E_193": {"ru": "У вас есть мышечные спазмы в лице, шее или другой части тела?", "kk": "Бет, мойын немесе дененің басқа бөлігінде бұлшықет спазмдары бар ма?"},
    "E_205": {"ru": "У вас внезапно возникла трудность или невозможность открыть рот?", "kk": "Кенет ауыз ашу қиындады ма немесе ауыз аша алмайсыз ба?"},
    "E_206": {"ru": "У вас есть болезненные язвочки во рту?", "kk": "Ауызда ауырсынатын жаралар бар ма?"},
    "E_210": {"ru": "Вас недавно рвало кровью или массой, похожей на кофейную гущу?", "kk": "Жақында қан немесе кофе қалдығына ұқсас массамен құстыңыз ба?"},
    "E_212": {"ru": "Вы заметили, что ваш голос стал более низким, тихим или хриплым?", "kk": "Дауысыңыз бұрынғыдан төмен, тыныш немесе қарлығып кеткенін байқадыңыз ба?"},
    "E_215": {"ru": "Ваши симптомы ухудшаются после еды?", "kk": "Белгілер тамақтанғаннан кейін нашарлайды ма?"},
    "E_217": {"ru": "Симптомы ухудшаются лёжа и облегчаются сидя?", "kk": "Белгілер жатқанда нашарлайды ма және отырғанда жеңілдейді ме?"},
    "E_218": {"ru": "Симптомы усиливаются при физической нагрузке и облегчаются в покое?", "kk": "Белгілер дене жүктемесі кезінде күшейеді ме және тыныштықта жеңілдейді ме?"},
    "E_219": {"ru": "Ваши симптомы более выражены ночью?", "kk": "Белгілер түнде айқынырақ па?"},
    "E_221": {"ru": "Симптомы усиливаются при кашле, натуживании или поднятии тяжести?", "kk": "Белгілер жөтелде, күш түскенде немесе ауырлық көтергенде күшейеді ме?"},
}


