"""
ML inference module for TMS backend.
Loads the trained RandomForest model from 'training sklearn/' and exposes the
posterior over diseases used by the question engine and the /analys endpoint.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("tms")

# ─── Paths ───
_BASE = Path(__file__).parent
_TRAINING_DIR = _BASE / "training sklearn"

_MODEL_PATH    = _TRAINING_DIR / "model.pkl"
_FEATURES_PATH = _TRAINING_DIR / "feature_names.json"
_LABELS_PATH   = _TRAINING_DIR / "label_classes.json"
_EVIDENCES_PATH = _TRAINING_DIR / "release_evidences.json"
_STATS_PATH    = _TRAINING_DIR / "evidence_stats.json"

# ─── Load model and metadata at import time ───
try:
    import joblib
    _clf = joblib.load(_MODEL_PATH)
    # Saved with n_jobs=-1 and verbose=1: every predict_proba spun up a thread
    # pool (~13 ms instead of ~1 ms) and logged two lines. Run inline, quietly.
    _clf.n_jobs = 1
    _clf.verbose = 0
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

# ─── Evidence statistics (built by 'training sklearn/build_evidence_stats.py') ───
# P(evidence | disease), P(disease) and demographics per disease, counted from
# the DDXPlus validation split.
try:
    with open(_STATS_PATH, encoding="utf-8") as f:
        EVIDENCE_STATS: dict = json.load(f)
    logger.info("Evidence stats loaded: %d patients", EVIDENCE_STATS.get("n_patients", 0))
except Exception as e:
    EVIDENCE_STATS = {}
    logger.error("Failed to load evidence_stats.json: %s", e)

# Smoothing: likelihoods are clipped to [floor, 1 - floor]. 0.03 measured best in
# eval/simulate.py, including with a wrongly extracted symptom (--noise 0.3).
_LIKELIHOOD_FLOOR = 0.03


def _build_likelihood_matrix(floor: float = _LIKELIHOOD_FLOOR) -> np.ndarray:
    """(n_classes, N_FEATURES) matrix of P(feature token | disease), clipped to
    [floor, 1 - floor]. The floor is the smoothing that keeps one wrong symptom
    from zeroing out the right disease."""
    lik = np.full((len(LABEL_CLASSES), N_FEATURES), floor, dtype=np.float64)
    table = EVIDENCE_STATS.get("p_evidence_given_disease", {})
    for ci, disease in enumerate(LABEL_CLASSES):
        for token, p in table.get(disease, {}).items():
            if token in _feat_idx:
                lik[ci, _feat_idx[token]] = min(max(p, floor), 1 - floor)
    return lik


LIKELIHOODS = _build_likelihood_matrix()
_CLASS_PRIORS = np.array(
    [EVIDENCE_STATS.get("priors", {}).get(d, 1.0 / max(len(LABEL_CLASSES), 1)) for d in LABEL_CLASSES]
)
_AGE_BIN_EDGES: List[int] = EVIDENCE_STATS.get("age_bin_edges", [])
_AGE_LIKELIHOODS = np.array(
    [EVIDENCE_STATS.get("p_age_bin_given_disease", {}).get(d, [1.0] * (len(_AGE_BIN_EDGES) + 1))
     for d in LABEL_CLASSES]
)
_SEX_LIKELIHOODS = {
    s: np.array([EVIDENCE_STATS.get("p_sex_given_disease", {}).get(d, {}).get(s, 0.5) for d in LABEL_CLASSES])
    for s in ("M", "F")
}


def build_feature_vector(evidences: Iterable[str], age: int = 25, sex: str = "M") -> np.ndarray:
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


def _mentions(ev_set: set, code: str) -> bool:
    """True if `code` is present, or (for categorical codes like E_57) any value
    of it other than V_123 ("nowhere") is."""
    if code in ev_set:
        return True
    prefix = code + "_@_"
    return any(e.startswith(prefix) and not e.endswith("_@_V_123") for e in ev_set)

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
        "diseases": ["Acute COPD exacerbation / infection"],
        "penalty": 0.10,
    },
]


# ─── Posterior over diseases ─────────────────────────────────────────────────
# Base posterior:
#   "nb": naive Bayes over evidence_stats.json. Handles the three states a chat
#         produces — reported, denied, never mentioned (skipped) — and uses age/sex
#         only when the patient stated them.
#   "rf": the RandomForest. Trained on complete DDXPlus records (~20 evidences,
#         0 = "absent"), so a chat's handful of evidences reads to it as "denies
#         everything else"; it cannot use denied answers at all.
# eval/simulate.py on 3,000 held-out test patients (chief complaint + 2 symptoms,
# then follow-up questions): nb 95.8% top-1 vs rf 65.0%. With the full patient
# record both reach ~99%. triage.RED_MASS_THRESHOLD is tuned for nb; re-tune it
# before switching (with rf it sends ~2/3 of green-tier patients to red).
POSTERIOR = "nb"

_CLASS_IDX = {d: i for i, d in enumerate(LABEL_CLASSES)}


def _age_bin(age: int) -> int:
    return sum(age >= edge for edge in _AGE_BIN_EDGES)


def _rf_class_proba(evidences: Iterable[str], age: Optional[int], sex: Optional[str]) -> np.ndarray:
    vec = build_feature_vector(evidences, 25 if age is None else age, sex or "M")
    return _clf.predict_proba(vec.reshape(1, -1))[0]


def _nb_class_proba(
    present: Iterable[str],
    absent: Iterable[str],
    age: Optional[int] = None,
    sex: Optional[str] = None,
) -> np.ndarray:
    log_p = np.log(_CLASS_PRIORS)
    idx = [_feat_idx[e] for e in present if e in _feat_idx]
    if idx:
        log_p = log_p + np.log(LIKELIHOODS[:, idx]).sum(axis=1)
    idx = [_feat_idx[e] for e in absent if e in _feat_idx]
    if idx:
        log_p = log_p + np.log(1.0 - LIKELIHOODS[:, idx]).sum(axis=1)
    if age is not None and len(_AGE_BIN_EDGES):
        log_p = log_p + np.log(_AGE_LIKELIHOODS[:, _age_bin(age)])
    if sex in _SEX_LIKELIHOODS:
        log_p = log_p + np.log(_SEX_LIKELIHOODS[sex])
    p = np.exp(log_p - log_p.max())
    return p / p.sum()


def _apply_clinical_adjustments(
    proba: np.ndarray, ev_set: set, age: Optional[int], sex: Optional[str], hand_priors: bool = True,
) -> np.ndarray:
    """Geographic filter, correction rules and age/sex priors, applied per class
    (before pooling, so both members of a clinical group are adjusted alike)."""
    p = proba.astype(np.float64).copy()

    def scale(disease: str, factor: float) -> None:
        if disease in _CLASS_IDX:
            p[_CLASS_IDX[disease]] *= factor

    # Geographic filter — diseases absent in CIS/Central Asia
    for disease in _GEO_EXCLUDED:
        scale(disease, 0.0)

    # Rule 1: Pleuritic pain (E_220) is AGAINST MI / ACS.
    if "E_220" in ev_set:
        for acs_name in ("Possible NSTEMI / STEMI", "Unstable angina"):
            scale(acs_name, 0.4)  # 60% penalty

    # Rule 2: GERD context (E_173 present) — penalize PE and pulmonary edema.
    if "E_173" in ev_set and "E_14" not in ev_set:
        for noise_name in ("Pulmonary embolism", "Acute pulmonary edema",
                           "Spontaneous pneumothorax"):
            scale(noise_name, 0.3)  # 70% penalty

    # Rule-based exclusion dictionary: ALL `require` present, NONE of `absent`.
    for rule in _EXCLUSION_RULES:
        if rule["require"] <= ev_set and not any(_mentions(ev_set, a) for a in rule["absent"]):
            for disease in rule["diseases"]:
                scale(disease, rule["penalty"])

    # Age/sex priors: soft penalties, not hard blocks. Only for a stated age —
    # v1 defaulted unknown ages to 25, penalising ACS for anyone who didn't say.
    # Skipped for naive Bayes, which already has P(age|d), P(sex|d) from data.
    if hand_priors and age is not None:
        for prior in _AGE_SEX_PRIORS:
            age_lo, age_hi = prior["age_range"]
            sex_match = prior.get("sex")  # None = any sex
            if age_lo <= age <= age_hi and (sex_match is None or sex == sex_match):
                for disease in prior["diseases"]:
                    scale(disease, prior["penalty"])

    total = p.sum()
    return p / total if total > 0 else p


def class_posterior(
    evidences: Iterable[str],
    absent: Iterable[str] = (),
    age: Optional[int] = None,
    sex: Optional[str] = None,
    posterior: Optional[str] = None,
) -> np.ndarray:
    """P(disease | answers) per LABEL_CLASSES entry, after clinical adjustments.
    age/sex = None means the patient didn't say."""
    posterior = posterior or POSTERIOR
    present = list(dict.fromkeys(evidences))
    absent = [e for e in dict.fromkeys(absent) if e not in set(present)]

    if posterior == "nb":
        proba = _nb_class_proba(present, absent, age, sex)
    else:
        proba = _rf_class_proba(present, age, sex)
    return _apply_clinical_adjustments(proba, set(present), age, sex, hand_priors=posterior != "nb")


def pool_classes(proba: np.ndarray) -> Dict[str, float]:
    """Sum class probabilities into clinical groups; sorted, highest first."""
    pooled: Dict[str, float] = {}
    for i, disease in enumerate(LABEL_CLASSES):
        canonical = _GROUP_MAP.get(disease, disease)
        pooled[canonical] = pooled.get(canonical, 0.0) + float(proba[i])
    pooled = {d: p for d, p in pooled.items() if d not in _GEO_EXCLUDED}
    return dict(sorted(pooled.items(), key=lambda x: x[1], reverse=True))


def predict_distribution(
    evidences: Iterable[str],
    age: Optional[int] = None,
    sex: Optional[str] = None,
    absent: Iterable[str] = (),
    **kwargs,
) -> Dict[str, float]:
    """Full differential: canonical disease → probability, highest first."""
    return pool_classes(class_posterior(evidences, absent, age, sex, **kwargs))


def predict_top(
    evidences: List[str],
    age: Optional[int] = None,
    sex: Optional[str] = None,
    top_n: int = 3,
    absent: Iterable[str] = (),
) -> List[Tuple[str, float]]:
    """
    Predict top-N diagnoses from evidence list.
    Applies clinical grouping: probabilities of clinically equivalent
    diseases are pooled under a single canonical name.

    Returns:
        List of (disease_name, probability) tuples, sorted descending.
        Falls back to [("Unknown", 0.0)] if the model data is not loaded.
    """
    if N_FEATURES == 0 or (POSTERIOR == "rf" and _clf is None):
        logger.error("model not available for prediction")
        return [("Unknown", 0.0)]

    absent = list(absent)
    top = list(predict_distribution(evidences, age, sex, absent).items())[:top_n]
    logger.info("predict_top: %d present, %d absent -> top-%d %s",
                len(evidences), len(absent), top_n, [(d, f"{p:.4f}") for d, p in top])
    return top


def explain_prediction(
    evidences: List[str],
    age: Optional[int] = None,
    sex: Optional[str] = None,
    absent: Iterable[str] = (),
    diseases: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Leave-one-out attributions: for each disease, how much its probability drops
    when one reported evidence is removed (positive = that evidence supports it).
    These are measured facts about the model, given to the LLM to verbalise
    instead of letting it invent symptom weights.
    """
    absent = list(absent)
    full = predict_distribution(evidences, age, sex, absent)
    diseases = diseases or list(full)[:3]
    result: Dict[str, List[Tuple[str, float]]] = {d: [] for d in diseases}

    for ev in dict.fromkeys(evidences):
        reduced = predict_distribution([e for e in evidences if e != ev], age, sex, absent)
        for d in diseases:
            result[d].append((ev, full.get(d, 0.0) - reduced.get(d, 0.0)))
    for ev in dict.fromkeys(absent):
        reduced = predict_distribution(evidences, age, sex, [e for e in absent if e != ev])
        for d in diseases:
            result[d].append((f"NOT {ev}", full.get(d, 0.0) - reduced.get(d, 0.0)))

    return {
        d: sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_k]
        for d, pairs in result.items()
    }


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


# ─── QAEngine initialisation ────────────────────────────────────────────────
# Instantiate the two-round QA engine using all objects already loaded above.
# Importing qa_engine here (at module bottom) avoids circular deps.
# NO try/except — if this fails, the server must crash loudly.
from qa_engine import QAEngine
from triage import tier_of


def make_qa_engine(strategy: str = "eig", **posterior_kwargs) -> QAEngine:
    """Build a QAEngine; posterior_kwargs (posterior="rf"|"nb") are for evals."""
    return QAEngine(
        clf=_clf,
        feature_names=FEATURE_NAMES,
        label_classes=LABEL_CLASSES,
        evidences_meta=EVIDENCES_META,
        disease_relevant_evidences=_DISEASE_RELEVANT_EVIDENCES,
        question_translations=_QUESTION_TRANSLATIONS,
        evidence_prerequisites=_EVIDENCE_PREREQUISITES,
        never_ask=_NEVER_ASK,
        likelihoods=LIKELIHOODS,
        posterior_fn=lambda ev, ab, age, sex: class_posterior(ev, ab, age, sex, **posterior_kwargs),
        pool_fn=pool_classes,
        tier_of=tier_of,
        strategy=strategy,
    )


qa_engine = make_qa_engine()
logger.info("QAEngine initialised successfully")
