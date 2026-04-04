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


def find_discriminative_evidences(
    evidences: List[str],
    age: int = 25,
    sex: str = "M",
    top_n: int = 3,
) -> List[dict]:
    """
    Find the top_n most discriminative binary evidences to ask about next.
    Uses probability-shift method: for each candidate evidence, compute how much
    setting it to 1 would shift the top-3 disease probabilities.
    Returns list of {evidence_id, question_en, question_ru, question_kk, score}.
    """
    if _clf is None or N_FEATURES == 0:
        return []

    base_vec = build_feature_vector(evidences, age, sex)
    base_proba = _clf.predict_proba(base_vec.reshape(1, -1))[0]
    top3_idx = np.argsort(base_proba)[::-1][:3]

    # Only consider binary evidences not yet known
    candidates = [
        ev_id for ev_id, meta in EVIDENCES_META.items()
        if meta.get("data_type") == "B"
        and ev_id not in evidences
        and ev_id in _feat_idx
        and not meta.get("is_antecedent", False)  # skip history/family questions
    ]

    if not candidates:
        return []

    # Batch predict: one row per candidate with that feature set to 1
    batch = np.tile(base_vec, (len(candidates), 1))
    for i, ev_id in enumerate(candidates):
        batch[i, _feat_idx[ev_id]] = 1.0

    probas = _clf.predict_proba(batch)  # (n_candidates, n_classes)

    base_top3 = base_proba[top3_idx]
    scores = []
    for i, ev_id in enumerate(candidates):
        shift = float(np.max(np.abs(probas[i][top3_idx] - base_top3)))
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
}


_FEW_SHOT_EXAMPLES = '''
EXAMPLES (study these carefully before extracting):

Input: "I have a fever, runny nose, body aches, headache and have been sick for 2 days. Male, 28."
Output: {"age": 28, "sex": "M", "evidences": ["E_91", "E_94", "E_181", "E_175", "E_144", "E_53", "E_55_@_V_89"]}
Reasoning: E_91=fever, E_94=chills/shivers, E_181=runny nose, E_175=general malaise+muscle aches, E_144=diffuse muscle pain, E_53=pain present, E_55_@_V_89=forehead pain(headache)

Input: "У меня болит голова и температура 38, насморк, ломота в теле, болею 2 дня. Мужчина, 28 лет."
Output: {"age": 28, "sex": "M", "evidences": ["E_91", "E_94", "E_181", "E_175", "E_144", "E_53", "E_55_@_V_89"]}
Reasoning: E_91=температура/fever, E_94=ломота/chills, E_181=насморк/runny nose, E_175=общее недомогание, E_144=боль в мышцах, E_53=боль есть, E_55_@_V_89=головная боль/forehead

Input: "Сильная боль в правом нижнем животе, тошнота, температура 37.8. Женщина, 22 года."
Output: {"age": 22, "sex": "F", "evidences": ["E_91", "E_53", "E_55_@_V_87", "E_148"]}
Reasoning: E_91=fever, E_53=pain present, E_55_@_V_87=right iliac fossa(lower right abdomen), E_148=nausea

Input: "Chest pain radiating to left arm, sweating, shortness of breath. Male, 55."
Output: {"age": 55, "sex": "M", "evidences": ["E_14", "E_57", "E_50", "E_66", "E_53", "E_155"]}
Reasoning: E_14=chest pain at rest, E_57=pain radiates to another location, E_50=increased sweating, E_66=shortness of breath, E_53=pain present, E_155=palpitations/racing heart

Input: "Сильная боль в груди отдаёт в левую руку, потею, трудно дышать. Мужчина 55 лет."
Output: {"age": 55, "sex": "M", "evidences": ["E_14", "E_57", "E_50", "E_66", "E_53", "E_155"]}
Reasoning: E_14=боль в груди в покое, E_57=боль иррадиирует, E_50=потливость, E_66=одышка, E_53=боль есть, E_155=сердцебиение

Input: "Severe difficulty breathing, wheezing sound when exhaling, history of asthma. Female, 30."
Output: {"age": 30, "sex": "F", "evidences": ["E_66", "E_214", "E_112", "E_124", "E_46"]}
Reasoning: E_66=shortness of breath, E_214=wheezing on exhale, E_112=noisy breathing, E_124=asthma history, E_46=asthma attacks in past year

Input: "Изжога, жжение от желудка поднимается к горлу, кислый привкус. Мужчина 40 лет."
Output: {"age": 40, "sex": "M", "evidences": ["E_173", "E_53", "E_55_@_V_197"]}
Reasoning: E_173=burning from stomach to throat with bitter taste, E_53=pain present, E_55_@_V_197=epigastric pain location

Input: "Сильно болит горло, больно глотать, температура 38.5, увеличены лимфоузлы. 19 лет, женщина."
Output: {"age": 19, "sex": "F", "evidences": ["E_97", "E_65", "E_91", "E_9"]}
Reasoning: E_97=sore throat, E_65=difficulty swallowing, E_91=fever, E_9=swollen/painful lymph nodes

Input: "Внезапное сильное сердцебиение, страх, онемение в руках, трудно дышать, всё прошло за 20 минут."
Output: {"age": null, "sex": null, "evidences": ["E_155", "E_66", "E_82", "E_128"]}
Reasoning: E_155=racing/palpitations, E_66=shortness of breath, E_82=lightheaded/dizzy/faint, E_128=suffocating episode that resolved
'''


def build_extraction_prompt() -> str:
    """
    Build a condensed system prompt listing all evidences for LLaMA NLP extractor.
    Returns prompt string to be used in /extract_symptoms endpoint.
    """
    lines = [
        "You are a medical symptom extractor for the DDXPlus medical dataset. "
        "Your task is to map a patient's description to the correct evidence IDs from the list below.\n"
        "CRITICAL RULES:\n"
        "- Only use evidence IDs from the list below — do NOT invent IDs\n"
        "- For pain location (E_55), you MUST use E_55_@_V_XX format with the correct value\n"
        "- Common symptoms mapping: fever→E_91, chills/body aches→E_94+E_175+E_144, "
        "runny nose→E_181, cough→E_201, sore throat→E_97, nausea→E_116, fatigue→E_89\n"
        "- Headache = E_53 + E_55_@_V_89 (forehead) or E_55_@_V_25 (back of head)\n"
        "- Do NOT include risk factors or history unless explicitly mentioned\n",
        _FEW_SHOT_EXAMPLES,
        "\nCOMPLETE EVIDENCE LIST:\n"
    ]

    for ev_id, meta in EVIDENCES_META.items():
        q = meta.get("question_en", "")
        dtype = meta.get("data_type", "B")
        if dtype == "B":
            lines.append(f"- {ev_id}: {q}")
        else:
            vals = meta.get("possible-values", [])
            val_meanings = meta.get("value_meaning", {})
            val_list = []
            for v in vals[:10]:  # limit to avoid overly long prompt
                meaning = val_meanings.get(v, {})
                en = meaning.get("en", v) if isinstance(meaning, dict) else v
                val_list.append(f"{v}={en}")
            lines.append(f"- {ev_id}: {q} [values: {', '.join(val_list)}{'...' if len(vals) > 10 else ''}]")

    lines.append(
        "\nReturn ONLY a valid JSON object (no markdown, no explanation):\n"
        '{"age": <integer or null>, "sex": "<M or F or null>", '
        '"evidences": ["E_XX", "E_YY_@_V_ZZ", ...]}\n'
        "- Only include evidences clearly present in the patient text\n"
        "- If age/sex not mentioned, use null"
    )

    return "\n".join(lines)
