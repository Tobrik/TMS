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


def sklearn_predict(
    evidences: List[str],
    age: int = 25,
    sex: str = "M",
    top_n: int = 3,
) -> List[Tuple[str, float]]:
    """
    Predict top-N diagnoses from evidence list.

    Returns:
        List of (disease_name, probability) tuples, sorted descending.
        Falls back to [("Unknown", 0.0)] if model not loaded.
    """
    if _clf is None or N_FEATURES == 0:
        logger.error("sklearn model not available for prediction")
        return [("Unknown", 0.0)]

    vec = build_feature_vector(evidences, age, sex)
    proba = _clf.predict_proba(vec.reshape(1, -1))[0]

    top_indices = np.argsort(proba)[::-1][:top_n]
    return [(LABEL_CLASSES[i], float(proba[i])) for i in top_indices]


def build_extraction_prompt() -> str:
    """
    Build a condensed system prompt listing all evidences for LLaMA NLP extractor.
    Returns prompt string to be used in /extract_symptoms endpoint.
    """
    lines = [
        "You are a medical symptom extractor. Your task is to analyze a patient's description "
        "and identify which symptoms and risk factors are present.\n"
        "Below is the complete list of evidence IDs with their descriptions:\n"
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
        "Rules:\n"
        "- Include evidence ID as plain E_XX for binary symptoms that are present\n"
        "- Include E_XX_@_V_YY for categorical/multi-choice values\n"
        "- Only include evidences that are clearly mentioned or strongly implied\n"
        "- Do NOT invent evidences not supported by the text\n"
        "- If age/sex not mentioned, use null"
    )

    return "\n".join(lines)
