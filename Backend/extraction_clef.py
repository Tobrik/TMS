"""
Symptom extraction with a Cloudflare Clef decision model (EXPERIMENTAL).

Same output as extraction.extract() — an Extraction with present / absent /
uncertain findings and a safety review — so eval/compare_extraction.py can run
both through the identical model + triage.

Clef answers with probabilities, not text:
  - every symptom/history code is a `choice`: not_mentioned / reported / denied
  - pain location, radiation, character, intensity, onset, travel and sex are
    `choice` questions over the code lists (choice allows up to 255 options)
  - the safety review is an emergency yes/no plus an out-of-scope choice
There are no quotes to ground, so the probability is the confidence, shown to
the patient on the confirmation step: ≥ PRECHECK_MIN is a pre-ticked chip,
UNCERTAIN_MIN–PRECHECK_MIN a "?" chip the patient decides on. Patients tend to
accept defaults, so only confident findings are pre-ticked.

5 requests per complaint, sent concurrently. Measured: ~22K input tokens
($0.84 per 1,000 complaints on clef-flash; ~130 a day inside the free allowance).
Enable with EXTRACTION_BACKEND=clef (see extraction.extract_complaint).
"""

import asyncio
import logging
from typing import Dict, List, Optional

import clef
from extraction import (
    BINARY_CODES, COMMON_LOCATIONS, TRAVEL, _PAIN_CHARACTER,
    Extraction, Finding, _fallback, apply_safety_net, short_label,
)
from ml_model import LABEL_CLASSES, _feat_idx

logger = logging.getLogger("tms.extraction_clef")

PRECHECK_MIN = 0.8      # P(reported) at or above → present, pre-ticked
UNCERTAIN_MIN = 0.25    # P(reported) at or above → uncertain, "?" for the patient
DENIAL_MIN = 0.8        # P(denied) at or above → absent, pre-marked "no"
EMERGENCY_WARNING = 0.9  # P(emergency) at or above → soft warning (zone ≥ yellow, never red)
OUT_OF_SCOPE_MIN = 0.5

BATCH = 52  # 208 codes → 4 requests of 52 (limit 64 questions per request)

# Ties go to the first option, so "not_mentioned" comes first: when unsure, don't invent.
_STATUS = {"not_mentioned": None, "reported": None, "denied": None}

_OUT_OF_SCOPE = {
    "in_scope": "Plausibly explained by one of the conditions listed",
    "appendicitis": None, "migraine": None, "urinary_tract_infection": None, "gastroenteritis": None,
    "stroke": None, "injury_or_fracture": None, "pregnancy_related": None,
    "other_outside": "Clearly something else that is not listed",
}


def _symptom_batches() -> List[dict]:
    # The option ids say it all; repeating "does the patient report it..." in
    # every question cost ~3K tokens per complaint.
    questions = {
        code: {"type": "choice", "instructions": short_label(code), "criteria": _STATUS}
        for code in BINARY_CODES
    }
    codes = list(questions)
    return [{c: questions[c] for c in codes[i:i + BATCH]} for i in range(0, len(codes), BATCH)]


def _context_questions() -> dict:
    scale = {"not_stated": None, **{str(i): None for i in range(11)}}
    return {
        "pain_location": {
            "type": "choice", "instructions": "Where does the patient say it hurts? (R/L = right/left)",
            "criteria": {"none": "No pain or no location given", **COMMON_LOCATIONS},
        },
        "pain_radiation": {
            "type": "choice", "instructions": "Where does the pain spread / radiate to, if the patient says?",
            "criteria": {"not_stated": None, **COMMON_LOCATIONS},
        },
        "pain_character": {
            "type": "choice", "instructions": "How does the patient describe the pain?",
            "criteria": {"not_stated": None, **_PAIN_CHARACTER},
        },
        "pain_intensity": {
            "type": "choice", "instructions": "Pain intensity the patient states, 0 (none) to 10 (worst)?",
            "criteria": scale,
        },
        "pain_onset": {
            "type": "choice", "instructions": "How suddenly did the pain start, 0 (very gradually) to 10 (all at once)?",
            "criteria": scale,
        },
        "travel": {
            "type": "choice", "instructions": "Travel outside the country in the last 4 weeks, as the patient states?",
            "criteria": {"not_mentioned": None, **TRAVEL},
        },
        "sex": {
            "type": "choice", "instructions": "The patient's sex, if stated or clear from the text?",
            "criteria": {"not_stated": None, "M": "male", "F": "female"},
        },
        "emergency": {
            "type": "noul",
            "instructions": ("Does the text describe a condition needing emergency care now (chest pain at rest or "
                             "with sweating/radiation, breathing difficulty at rest, stridor, fainting, confusion, "
                             "seizure, vomiting blood, black stool, coughing blood, face/throat swelling, sudden "
                             "severe headache, stroke signs)?"),
            "criteria": {"true": "emergency now", "false": "not an emergency"},
        },
        "scope": {
            "type": "choice",
            "instructions": "Could one of these conditions plausibly explain the complaint: " + ", ".join(LABEL_CLASSES),
            "criteria": _OUT_OF_SCOPE,
        },
    }


SYMPTOM_BATCHES = _symptom_batches()
CONTEXT_QUESTIONS = _context_questions()


def _probs(answer: Optional[dict]) -> Dict[str, float]:
    return (answer or {}).get("probabilities") or {}


def _banded(code: str, p: float) -> Optional[Finding]:
    """present if p ≥ PRECHECK_MIN (pre-ticked chip), uncertain if ≥ UNCERTAIN_MIN ("?" chip)."""
    if p >= PRECHECK_MIN:
        return Finding(code, "present", votes=1, source="clef", confidence=round(p, 3))
    if p >= UNCERTAIN_MIN:
        return Finding(code, "uncertain", votes=1, source="clef", confidence=round(p, 3))
    return None


def _findings(answers: dict) -> List[Finding]:
    findings: List[Finding] = []
    for code in BINARY_CODES:
        p = _probs(answers.get(code))
        if not p:
            continue
        reported, denied = p.get("reported", 0.0), p.get("denied", 0.0)
        if denied >= DENIAL_MIN:
            # A wrong "no" pushes the model away from the right disease, so only
            # confident denials are pre-marked; unsure ones are simply left out.
            findings.append(Finding(code, "absent", votes=1, source="clef", confidence=round(denied, 3)))
        elif (finding := _banded(code, reported)) is not None:
            findings.append(finding)

    def options(qid: str, skip: str, limit: int = 3) -> List[tuple]:
        p = _probs(answers.get(qid))
        picked = sorted(((v, pr) for v, pr in p.items() if v != skip and pr >= UNCERTAIN_MIN), key=lambda x: -x[1])
        return picked[:limit]

    pain: List[Finding] = []
    for qid, prefix, skip in (("pain_location", "E_55", "none"), ("pain_radiation", "E_57", "not_stated"),
                              ("pain_character", "E_54", "not_stated")):
        pain += [f for v, pr in options(qid, skip) if (f := _banded(f"{prefix}_@_{v}", pr))]
    for qid, prefix in (("pain_intensity", "E_56"), ("pain_onset", "E_59")):
        pain += [f for v, pr in options(qid, "not_stated", limit=1) if (f := _banded(f"{prefix}_@_{v}", pr))]
    pain = [f for f in pain if f.code in _feat_idx]

    # A confident pain location means pain: make sure E_53 is at least as sure.
    if pain:
        strongest = max(pain, key=lambda f: f.confidence)
        current = next((f for f in findings if f.code == "E_53"), None)
        if current is None or current.status == "absent" or (current.confidence or 0) < strongest.confidence:
            findings = [f for f in findings if f.code != "E_53"]
            findings.append(Finding("E_53", strongest.status, votes=1, source="clef", confidence=strongest.confidence))
        findings += pain

    # Only travel abroad counts: Clef picks "no travel" (V_10) for texts that never
    # mention travel, which is a default, not something the patient said.
    for v, pr in options("travel", "not_mentioned", limit=2):
        if v != "V_10" and f"E_204_@_{v}" in _feat_idx and (f := _banded(f"E_204_@_{v}", pr)):
            findings.append(f)
            break
    return findings


def _review(answers: dict, model: str) -> dict:
    p_emergency = float((answers.get("emergency") or {}).get("noul") or 0.0)
    scope = _probs(answers.get("scope"))
    outside = sorted((o for o in scope if o != "in_scope"), key=lambda o: -scope[o])
    out_of_scope = bool(outside) and scope.get("in_scope", 1.0) < 1 - OUT_OF_SCOPE_MIN
    return {
        # Soft warning only: Clef's emergency score barely separated red-tier from
        # green-tier cases in the benchmark (0.5: 7/9 vs 9/24; 0.8: 3/9 vs 1/24),
        # and it has no quote to check. Never "verified", so it can raise triage to
        # yellow at most; red comes from confirmed red-flag symptoms and risk mass.
        "emergency": p_emergency >= EMERGENCY_WARNING,
        "emergency_verified": False,
        "emergency_quote": "",
        "emergency_probability": round(p_emergency, 3),
        "out_of_scope": out_of_scope,
        "likely_condition": outside[0].replace("_", " ") if out_of_scope and outside[0] != "other_outside" else None,
        "reason": "",
        "model": model,
    }


def interpret(answers: dict, text: str, model: str, usage: Optional[Dict[str, int]] = None) -> Extraction:
    """Clef answers → Extraction. Also accepts the `raw` form saved by extract()
    ({question: probabilities} / {"yes": p}), so cached runs can be re-scored when
    thresholds change, without new API calls."""
    answers = {
        qid: a if "type" in a else ({"type": "noul", "noul": a["yes"]} if set(a) == {"yes"}
                                    else {"type": "choice", "probabilities": a})
        for qid, a in answers.items()
    }
    from retrieval import extract_age_sex
    age, _regex_sex = extract_age_sex(text)
    sex = _probs(answers.get("sex"))
    sex_choice = max(sex, key=sex.get) if sex else "not_stated"

    return Extraction(
        findings=apply_safety_net(_findings(answers), text),
        age=age,
        sex=sex_choice if sex_choice in ("M", "F") and sex[sex_choice] >= 0.5 else None,
        review=_review(answers, model),
        mode="clef",
        samples=1,
        model=model,
        usage=usage or {"input_tokens": 0, "output_tokens": 0},
        raw={qid: (a.get("probabilities") or {"yes": a.get("noul")}) for qid, a in answers.items()},
    )


async def extract(text: str, model: Optional[str] = None) -> Extraction:
    model = model or clef.default_model()
    batches = SYMPTOM_BATCHES + [CONTEXT_QUESTIONS]
    state = {"patient_complaint": text}
    results = await asyncio.gather(*(clef.ask(state, batch, model=model) for batch in batches))
    usage = {"input_tokens": sum(r.input_tokens for r in results if r), "output_tokens": 0}

    if any(r is None for r in results):
        # A partial answer would silently miss a quarter of the symptoms: treat as a failure.
        extraction = _fallback(text)
        extraction.usage = usage
        logger.info("extract (clef): %d/%d requests failed, fallback mode",
                    sum(r is None for r in results), len(results))
        return extraction

    answers: dict = {}
    for r in results:
        answers.update(r.answers)
    return interpret(answers, text, results[0].model, usage)
