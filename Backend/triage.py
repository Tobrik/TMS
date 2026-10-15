"""
Triage: maps a differential plus reported symptoms to a red / yellow / green zone.

Principles:
  - Triage by the worst plausible condition, not the most likely one: the zone
    follows the probability mass on dangerous diseases, not the top-1 name.
  - Symptom red flags override the model (they don't depend on what it guessed).
  - Every signal can only raise the zone, never lower it.
  - Model confidence is not urgency: a confident URTI is still green.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger("tms")

_CONDITIONS_PATH = Path(__file__).parent / "training sklearn" / "release_conditions.json"

ZONE_ORDER = {"green": 0, "yellow": 1, "red": 2}

# Hand-curated disease zones (clinical judgement), unchanged from v1.
CURATED_RED = {
    "Anaphylaxis", "Boerhaave", "Ebola", "Epiglottitis",
    "Acute pulmonary edema", "Possible NSTEMI / STEMI", "Pulmonary embolism",
    "Spontaneous pneumothorax", "Unstable angina",
}
CURATED_YELLOW = {
    "Atrial fibrillation", "Bronchospasm / acute asthma exacerbation",
    "Guillain-Barré syndrome", "HIV (initial infection)", "Myocarditis",
    "Pancreatic neoplasm", "PSVT", "Pneumonia", "Pulmonary neoplasm",
    "Stable angina", "Tuberculosis",
}

# DDXPlus severity per condition (1 = most severe ... 5 = least).
try:
    with open(_CONDITIONS_PATH, encoding="utf-8") as f:
        DISEASE_SEVERITY: Dict[str, int] = {
            c["cond-name-eng"]: int(c["severity"]) for c in json.load(f).values()
        }
except Exception as e:
    DISEASE_SEVERITY = {}
    logger.error("Failed to load release_conditions.json: %s", e)


def tier_of(disease: str) -> str:
    """
    Disease tier = max(curated zone, DDXPlus severity zone), where severity 1 →
    red and 2 → yellow. The severity data catches conditions the curated lists
    missed (e.g. laryngospasm is severity 1 but was green in v1).
    """
    severity = DISEASE_SEVERITY.get(disease, 5)
    if disease in CURATED_RED or severity == 1:
        return "red"
    if disease in CURATED_YELLOW or severity == 2:
        return "yellow"
    return "green"


# ─── Symptom-based red flags (safety layer) ──────────────────────────────────
# Medical triage depends on SYMPTOMS, not on what the model guessed.
# If ANY of these evidences are present → force that zone regardless of diagnosis.

RED_FLAG_EVIDENCES = {
    # Stridor = airway obstruction → minutes to death
    "E_112",
    # Chest pain at rest → possible ACS
    "E_14",
}

YELLOW_FLAG_EVIDENCES = {
    "E_91",   # Fever — needs medical evaluation
    "E_220",  # Pleuritic pain (pain on deep breathing) — pneumonia/PE marker
    "E_66",   # Shortness of breath
    "E_82",   # Dizziness / near-syncope
    "E_155",  # Palpitations / tachycardia
    "E_211",  # Repeated vomiting — dehydration risk
}

# Combinations that escalate to RED even if individual symptoms are only yellow
RED_FLAG_COMBOS = [
    {"E_91", "E_220"},          # Fever + pleuritic pain = pneumonia until proven otherwise
    {"E_91", "E_66"},           # Fever + shortness of breath = possible pneumonia/sepsis
    {"E_14", "E_50"},           # Chest pain + sweating = ACS
    {"E_14", "E_57_@_V_195"},   # Chest pain + radiation to left shoulder
    {"E_14", "E_57_@_V_121"},   # Chest pain + radiation to jaw
    {"E_14", "E_148"},          # Chest pain + nausea = ACS
]

# Probability-mass thresholds, from the sweep in eval/simulate.py (3,000 test
# patients, naive Bayes posterior). RED 0.10: red under-triage 0.0% (0.4% with a
# wrongly extracted symptom in 30% of patients), over-triage 4.9% (5.6%).
# 0.15 halves over-triage but misses ~1% of red cases. Under-triage first.
RED_MASS_THRESHOLD = 0.10     # P(any red-tier disease) at or above → red
YELLOW_MASS_THRESHOLD = 0.30  # P(any red- or yellow-tier disease) at or above → yellow


@dataclass
class TriageResult:
    zone: str
    reasons: List[dict] = field(default_factory=list)
    red_mass: float = 0.0
    yellow_mass: float = 0.0


def risk_mass(distribution: Dict[str, float]) -> Dict[str, float]:
    mass = {"red": 0.0, "yellow": 0.0, "green": 0.0}
    for disease, p in distribution.items():
        mass[tier_of(disease)] += p
    return mass


def assess(
    distribution: Dict[str, float],
    evidences: Iterable[str],
    review: Optional[dict] = None,
) -> TriageResult:
    """
    Args:
        distribution: full differential, canonical disease → probability
        evidences: evidences the patient reported as present
        review: optional LLM safety review of the original complaint, with keys
            emergency (bool), emergency_verified (bool), out_of_scope (bool)
    """
    ev_set = set(evidences)
    mass = risk_mass(distribution)
    result = TriageResult(zone="green", red_mass=mass["red"], yellow_mass=mass["red"] + mass["yellow"])

    def raise_to(zone: str, reason: dict) -> None:
        result.reasons.append({"zone": zone, **reason})
        if ZONE_ORDER[zone] > ZONE_ORDER[result.zone]:
            result.zone = zone

    for ev in sorted(ev_set & RED_FLAG_EVIDENCES):
        raise_to("red", {"code": "red_flag", "evidence": ev})
    for combo in RED_FLAG_COMBOS:
        if combo <= ev_set:
            raise_to("red", {"code": "red_flag_combo", "evidences": sorted(combo)})

    review = review or {}
    if review.get("emergency"):
        # Escalate-only. An emergency the reviewer could not back with a quote
        # from the patient's text still warrants a same-day look.
        zone = "red" if review.get("emergency_verified") else "yellow"
        raise_to(zone, {"code": "llm_emergency", "verified": bool(review.get("emergency_verified"))})
    if review.get("out_of_scope"):
        raise_to("yellow", {"code": "out_of_scope"})

    if result.red_mass >= RED_MASS_THRESHOLD:
        raise_to("red", {"code": "red_mass", "value": round(result.red_mass, 3)})
    for ev in sorted(ev_set & YELLOW_FLAG_EVIDENCES):
        raise_to("yellow", {"code": "yellow_flag", "evidence": ev})
    if result.yellow_mass >= YELLOW_MASS_THRESHOLD:
        raise_to("yellow", {"code": "yellow_mass", "value": round(result.yellow_mass, 3)})

    return result


def critical_alert(evidences: Iterable[str]) -> Optional[dict]:
    """Immediate warning shown while the patient is still answering questions."""
    ev_set = set(evidences)
    if "E_112" in ev_set:
        return {
            "level": "red",
            "message": "⚠️ СТРИДОР (шумное дыхание на вдохе) — признак обструкции дыхательных путей. "
                       "Немедленно вызовите скорую помощь (103)!",
        }
    if ev_set & RED_FLAG_EVIDENCES:
        return {
            "level": "red",
            "message": "⚠️ Обнаружены опасные симптомы. Рекомендуется срочная консультация врача или скорая помощь.",
        }
    if any(combo <= ev_set for combo in RED_FLAG_COMBOS):
        return {
            "level": "red",
            "message": "⚠️ Сочетание симптомов указывает на возможное неотложное состояние. "
                       "Обратитесь к врачу как можно скорее.",
        }
    return None
