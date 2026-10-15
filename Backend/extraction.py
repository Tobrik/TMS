"""
Symptom extraction: patient free text (RU / KK / EN) → DDXPlus evidence tokens.

Pipeline (all calls run concurrently):
  1. EXTRACTION_SAMPLES structured-output calls (default 1: Groq's base tier allows
     8K tokens/minute, and one request is ~4.4K). Each finding carries a status
     (present / absent / uncertain) and a verbatim quote from the patient's text.
  2. Grounding: a finding whose quote is not in the text is dropped.
  3. Voting across samples: kept if a majority agree; minority findings become
     "uncertain" (shown to the patient to confirm).
  4. Safety net: red-flag symptoms caught by the regex vocabulary are added even
     if the LLM missed them, unless the LLM grounded a denial.
  5. Reviewer call (more reasoning): emergency? outside the 49 conditions?
     It can only escalate triage, never lower it.
Without an LLM (no key, or every model failed) falls back to v1 regex + BM25.
"""

import asyncio
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from typing import Dict, List, Optional, Tuple

import llm
from ml_model import EVIDENCES_META, LABEL_CLASSES, validate_evidences, _feat_idx
from triage import RED_FLAG_COMBOS, RED_FLAG_EVIDENCES

logger = logging.getLogger("tms.extraction")

EXTRACTION_SAMPLES = int(os.environ.get("LLM_EXTRACTION_SAMPLES", "1"))

# Red flags the regex vocabulary may add on its own (see module docstring, step 4)
SAFETY_NET = (
    RED_FLAG_EVIDENCES
    | {e for combo in RED_FLAG_COMBOS for e in combo}
    | {"E_159", "E_210", "E_140", "E_45"}  # syncope, haematemesis, melena, haemoptysis
)

# DDXPlus's English glosses for pain character are poor translations of the
# French originals ("haunting" = lancinante); give the model usable ones.
_PAIN_CHARACTER = {
    "V_71": "tearing / ripping", "V_112": "shooting / lancinating", "V_154": "nagging / dull",
    "V_161": "tender to touch", "V_179": "stabbing, like a knife", "V_180": "pulling / tugging",
    "V_181": "burning", "V_182": "cramping", "V_183": "heavy / pressing", "V_184": "throbbing / pulsating",
    "V_191": "violent", "V_192": "sharp", "V_193": "sickening", "V_196": "frightening", "V_198": "exhausting",
}


# ─── Catalog & schema ────────────────────────────────────────────────────────

def _values(ev: str) -> Dict[str, str]:
    meta = EVIDENCES_META[ev]
    return {v: meta["value_meaning"][v]["en"] for v in meta["possible-values"]
            if f"{ev}_@_{v}" in _feat_idx}


_QUESTION_PREFIX = re.compile(
    r"^(do you (currently )?(have|feel|suffer from)|have you (ever )?(been|had|noticed|recently had|felt)?"
    r"|are you( currently)?|did you( previously, or do you currently,)?( have)?|is your"
    r"|does the person have|were you|do your)\s+",
    re.I,
)


def short_label(code: str) -> str:
    """'Do you have a fever (...)?' -> 'A fever (...)' — ~25% fewer prompt tokens."""
    text = _QUESTION_PREFIX.sub("", EVIDENCES_META[code]["question_en"].strip()).rstrip("?").strip()
    return text[:1].upper() + text[1:]


BINARY_CODES = sorted(
    (k for k, v in EVIDENCES_META.items() if v["data_type"] == "B" and k in _feat_idx),
    key=lambda k: int(k.split("_")[1]),
)
BODY_LOCATIONS = {v: name for v, name in _values("E_55").items() if v != "V_123"}
# The prompt lists the 70 locations patients actually name, not all 164 (saves ~700 tokens)
_COMMON_LOCATION_NAMES = {
    "forehead", "temple", "back of head", "occiput", "top of the head", "eye", "ear", "nose", "cheek", "jaw",
    "under the jaw", "mouth", "pharynx", "tonsil", "trachea", "thyroid cartilage", "back of the neck",
    "side of the neck", "cervical spine", "upper chest", "lower chest", "side of the chest", "breast",
    "posterior chest wall", "scapula", "shoulder", "thoracic spine", "epigastric", "belly", "hypochondrium",
    "flank", "iliac fossa", "pubis", "lumbar spine", "renal fossa", "groin", "hip", "thigh", "knee", "calf",
    "tibia", "ankle", "biceps", "forearm", "elbow",
}
COMMON_LOCATIONS = {v: name for v, name in BODY_LOCATIONS.items()
                    if re.sub(r"\(.\)", "", name).strip() in _COMMON_LOCATION_NAMES}
TRAVEL = {v: ("no travel" if v == "V_10" else name) for v, name in _values("E_204").items()}


def _catalog() -> str:
    lines = ["## Symptoms and history (code: what it means)"]
    for code in BINARY_CODES:
        tag = " [history]" if EVIDENCES_META[code].get("is_antecedent") else ""
        lines.append(f"{code}{tag}: {short_label(code)}")
    lines.append("\n## Body locations (pain.locations / pain.radiates_to; R/L = right/left)")
    lines.append("; ".join(f"{v} {name}" for v, name in COMMON_LOCATIONS.items()))
    lines.append("\n## Pain character (pain.character)")
    lines.append("; ".join(f"{v} {name}" for v, name in _PAIN_CHARACTER.items()))
    lines.append("\n## Travel outside the country in the last 4 weeks (travel.value)")
    lines.append("; ".join(f"{v} {name}" for v, name in TRAVEL.items()))
    return "\n".join(lines)


def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


# Codes are free strings, checked against the model's features after the call:
# enumerating ~450 codes in the schema cost ~2.5K tokens per request.
EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["findings", "pain", "travel", "age", "sex"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["code", "status", "quote"],
                "properties": {
                    "code": {"type": "string"},
                    "status": {"type": "string", "enum": ["present", "absent", "uncertain"]},
                    "quote": {"type": "string"},
                },
            },
        },
        "pain": _nullable({
            "type": "object",
            "additionalProperties": False,
            "required": ["locations", "radiates_to", "character", "intensity", "onset_speed", "quote"],
            "properties": {
                "locations": {"type": "array", "items": {"type": "string"}},
                "radiates_to": {"type": "array", "items": {"type": "string"}},
                "character": {"type": "array", "items": {"type": "string"}},
                "intensity": {"type": ["integer", "null"]},
                "onset_speed": {"type": ["integer", "null"]},
                "quote": {"type": "string"},
            },
        }),
        "travel": _nullable({
            "type": "object",
            "additionalProperties": False,
            "required": ["value", "quote"],
            "properties": {
                "value": {"type": "string"},
                "quote": {"type": "string"},
            },
        }),
        "age": {"type": ["integer", "null"]},
        "sex": {"type": ["string", "null"], "enum": ["M", "F", None]},
    },
}

EXTRACTION_PROMPT = """\
You are the clinical NLP layer of a triage system in Kazakhstan. Patients write in Russian, Kazakh or English, often colloquially.
Map the complaint to the codes listed below. Return JSON matching the schema; nothing else.

Rules:
1. Every finding needs `quote`: the exact words from the patient's text that support it, copied verbatim in the original language (no translation, no paraphrase). If you cannot quote it, leave it out.
2. status "present": the patient reports it. "absent": the patient explicitly denies it ("нет температуры", "жөтел жоқ", "no fever"). "uncertain": hedged ("кажется", "возможно", "мүмкін", "maybe").
3. Code only what is said or unmistakably implied ("не встаю с кровати" → bedridden). Never add symptoms from general knowledge of a disease.
4. Pain: if the patient has pain, add code E_53 as present AND fill `pain` (locations, radiation, character, intensity 0-10, onset_speed 0-10 where 10 = sudden) with what is stated; leave unstated parts empty/null. pain = null if no pain is mentioned.
5. Never code chest pain at rest (E_14) or radiation to the arm/jaw without an explicit statement. Anxiety alone is not cardiac.
6. One vomiting episode is E_148; E_211 only for repeated vomiting.
7. travel: only if travel abroad (or its absence) is mentioned. age/sex: only if stated (мужчина/ер адам/man → M, женщина/әйел/woman → F), else null.

""" + _catalog()


REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["emergency", "emergency_quote", "reason", "in_scope", "likely_condition"],
    "properties": {
        "emergency": {"type": "boolean"},
        "emergency_quote": {"type": "string"},
        "reason": {"type": "string"},
        "in_scope": {"type": "boolean"},
        "likely_condition": {"type": ["string", "null"]},
    },
}

REVIEW_PROMPT = f"""\
You are a triage safety reviewer. Read a patient's complaint (Russian, Kazakh or English) and answer two questions. Return JSON only.

1. emergency: does the text describe a condition needing emergency care NOW? Examples: chest pain at rest or with sweating/radiation, difficulty breathing at rest, stridor, fainting, confusion, seizure, vomiting blood or black stool, coughing blood, signs of anaphylaxis (swelling of face/throat with breathing trouble), sudden severe headache, signs of stroke, severe abdominal pain with rigid belly, suicidal intent. If true, `emergency_quote` must be the exact words from the text (verbatim, original language) that show it; else "".
2. in_scope: could the complaint plausibly be explained by one of the conditions below? If the most likely explanation is clearly something else (e.g. appendicitis, migraine, urinary tract infection, gastroenteritis, stroke, fracture, pregnancy-related), set in_scope=false and name it in `likely_condition` (English); else likely_condition=null.

`reason`: one short sentence, English.

Conditions the system can assess: {", ".join(LABEL_CLASSES)}.
"""


# ─── Display labels ──────────────────────────────────────────────────────────

_FIELD_LABELS = {  # categorical evidence → (en, ru, kk) prefix
    "E_55": ("Pain location", "Где болит", "Ауырсыну орны"),
    "E_57": ("Pain radiates to", "Боль отдаёт в", "Ауырсыну тарайды"),
    "E_54": ("Pain character", "Характер боли", "Ауырсыну сипаты"),
    "E_56": ("Pain intensity (0-10)", "Сила боли (0-10)", "Ауырсыну күші (0-10)"),
    "E_59": ("How suddenly the pain began (0-10)", "Насколько внезапно началась боль (0-10)",
             "Ауырсыну қаншалықты кенет басталды (0-10)"),
    "E_204": ("Travel abroad, last 4 weeks", "Поездка за границу за 4 недели", "Соңғы 4 аптада шетелге сапар"),
}


def evidence_label(code: str) -> Dict[str, str]:
    """{en, ru, kk} label for an evidence token. Binary evidences use the
    question text (translated where _QUESTION_TRANSLATIONS has it); categorical
    values use "<field>: <value>" with the value name in English."""
    from ml_model import _QUESTION_TRANSLATIONS

    base, _, value = code.partition("_@_")
    meta = EVIDENCES_META.get(base, {})
    if value:
        en, ru, kk = _FIELD_LABELS.get(base, (meta.get("question_en", base),) * 3)
        name = _PAIN_CHARACTER.get(value) if base == "E_54" else TRAVEL.get(value) if base == "E_204" else None
        name = name or meta.get("value_meaning", {}).get(value, {}).get("en", value)
        return {"en": f"{en}: {name}", "ru": f"{ru}: {name}", "kk": f"{kk}: {name}"}
    q_en = meta.get("question_en", code)
    translated = _QUESTION_TRANSLATIONS.get(code, {})
    return {"en": q_en, "ru": translated.get("ru", q_en), "kk": translated.get("kk", q_en)}


# ─── Grounding ───────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[«»\"'`“”„.,!?;:()\[\]]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def grounded(quote: str, text: str) -> bool:
    """True if `quote` appears in `text` (case/punctuation-insensitive), or all of
    its words of 3+ letters do (tolerates small reorderings by the model)."""
    q, t = _normalize(quote or ""), _normalize(text)
    if not q:
        return False
    if q in t:
        return True
    words = [w for w in q.split() if len(w) >= 3]
    t_words = set(t.split())
    return bool(words) and all(w in t_words for w in words)


# ─── Result ──────────────────────────────────────────────────────────────────

@dataclass
class Finding:
    code: str              # evidence token, e.g. "E_91" or "E_55_@_V_148"
    status: str            # present | absent | uncertain
    quote: str = ""
    votes: int = 0
    source: str = "llm"    # llm | safety_net | regex | bm25 | clef
    confidence: Optional[float] = None  # Clef: probability of this status


@dataclass
class Extraction:
    findings: List[Finding] = field(default_factory=list)
    age: Optional[int] = None
    sex: Optional[str] = None
    review: Optional[dict] = None
    mode: str = "llm"      # llm | fallback | clef
    samples: int = 0
    model: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    raw: Optional[dict] = None  # backend-specific output kept for offline analysis (Clef probabilities)

    def codes(self, status: str) -> List[str]:
        return [f.code for f in self.findings if f.status == status]

    def to_dict(self) -> dict:
        return {
            "evidences": self.codes("present"),
            "absent": self.codes("absent"),
            "uncertain": self.codes("uncertain"),
            "findings": [vars(f) for f in self.findings],
            "age": self.age,
            "sex": self.sex,
            "review": self.review,
            "mode": self.mode,
            "samples": self.samples,
            "model": self.model,
            "usage": self.usage,
        }


# ─── Parsing one sample ──────────────────────────────────────────────────────

def _sample_findings(data: dict, text: str) -> List[Tuple[str, str, str]]:
    """(token, status, quote) for every grounded finding in one LLM sample."""
    out: List[Tuple[str, str, str]] = []
    for f in data.get("findings") or []:
        code, status, quote = f.get("code"), f.get("status"), f.get("quote", "")
        if code in _feat_idx and status in ("present", "absent", "uncertain") and grounded(quote, text):
            out.append((code, status, quote))

    pain = data.get("pain")
    if isinstance(pain, dict) and grounded(pain.get("quote", ""), text):
        quote = pain["quote"]
        tokens = ["E_53"]
        tokens += [f"E_55_@_{v}" for v in pain.get("locations") or []]
        tokens += [f"E_57_@_{v}" for v in pain.get("radiates_to") or []]
        tokens += [f"E_54_@_{v}" for v in pain.get("character") or []]
        for field_name, ev in (("intensity", "E_56"), ("onset_speed", "E_59")):
            value = pain.get(field_name)
            if isinstance(value, int):
                tokens.append(f"{ev}_@_{min(max(value, 0), 10)}")
        out += [(t, "present", quote) for t in tokens if t in _feat_idx]

    travel = data.get("travel")
    if isinstance(travel, dict) and grounded(travel.get("quote", ""), text):
        token = f"E_204_@_{travel.get('value')}"
        if token in _feat_idx:
            out.append((token, "present", travel["quote"]))
    return out


def _vote(samples: List[dict], text: str) -> List[Finding]:
    n = len(samples)
    majority = n // 2 + 1
    per_token: Dict[str, Counter] = {}
    quotes: Dict[Tuple[str, str], str] = {}
    for data in samples:
        seen = set()
        for token, status, quote in _sample_findings(data, text):
            if (token, status) in seen:
                continue
            seen.add((token, status))
            per_token.setdefault(token, Counter())[status] += 1
            quotes.setdefault((token, status), quote)

    findings = []
    for token, votes in per_token.items():
        status, count = votes.most_common(1)[0]
        if count < majority:
            status = "uncertain"  # samples disagree → ask the patient
        quote = quotes.get((token, status)) or next(iter(q for (t, _), q in quotes.items() if t == token))
        findings.append(Finding(token, status, quote, votes=count))

    # Scalar scales: keep one value per field (median of the samples that set it)
    for prefix in ("E_56_@_", "E_59_@_"):
        scale = [f for f in findings if f.code.startswith(prefix)]
        if len(scale) > 1:
            keep = f"{prefix}{int(median(int(f.code.rsplit('_', 1)[1]) for f in scale))}"
            findings = [f for f in findings if not f.code.startswith(prefix)] + [
                Finding(keep, "present", scale[0].quote, votes=sum(f.votes for f in scale))]
    return findings


def _majority(values: List, default=None):
    values = [v for v in values if v is not None]
    if not values:
        return default
    value, count = Counter(values).most_common(1)[0]
    return value if count * 2 > len(values) or len(values) == 1 else default


# ─── Fallback (v1 behaviour) ─────────────────────────────────────────────────

def _fallback(text: str) -> Extraction:
    from retrieval import extract_age_sex, get_retriever, vocab_extract

    findings = [Finding(ev, "present", source="regex") for ev in vocab_extract(text)]
    try:
        retriever = get_retriever()
        known = {f.code for f in findings}
        added = 0
        for ev_id, score in retriever.retrieve(text, top_k=20):
            if score < 6.0 or added >= 8:
                break
            meta = retriever._meta.get(ev_id, {})
            if meta.get("data_type") == "B" and not meta.get("is_antecedent", False) and ev_id not in known:
                findings.append(Finding(ev_id, "present", source="bm25"))
                added += 1
    except Exception as e:
        logger.warning("BM25 fallback failed: %s", e)
    age, sex = extract_age_sex(text)
    valid, _ = validate_evidences([f.code for f in findings])
    return Extraction([f for f in findings if f.code in set(valid)], age, sex, mode="fallback")


# ─── Public API ──────────────────────────────────────────────────────────────

def apply_safety_net(findings: List[Finding], text: str) -> List[Finding]:
    """Add red flags the regex vocabulary finds in `text` unless the extractor
    already decided on them (present, or a denial it could support)."""
    from retrieval import vocab_extract

    status_of = {f.code: f.status for f in findings}
    for ev in sorted(set(vocab_extract(text)) & SAFETY_NET):
        if status_of.get(ev) not in ("present", "absent"):
            findings = [f for f in findings if f.code != ev]
            findings.append(Finding(ev, "present", source="safety_net"))
    return findings


async def review(text: str, usage: Optional[Dict[str, int]] = None) -> Optional[dict]:
    result = await llm.chat_json(
        [{"role": "system", "content": REVIEW_PROMPT}, {"role": "user", "content": text}],
        REVIEW_SCHEMA, name="review", reasoning="medium", temperature=0.5, max_tokens=3000,
    )
    if result is not None and usage is not None:
        usage["input_tokens"] += result.input_tokens
        usage["output_tokens"] += result.output_tokens
    if result is None or not isinstance(result.data, dict):
        return None
    data = result.data
    emergency = bool(data.get("emergency"))
    return {
        "emergency": emergency,
        "emergency_verified": emergency and grounded(data.get("emergency_quote", ""), text),
        "emergency_quote": data.get("emergency_quote", "") if emergency else "",
        "out_of_scope": data.get("in_scope") is False,
        "likely_condition": data.get("likely_condition") if data.get("in_scope") is False else None,
        "reason": data.get("reason", ""),
        "model": result.model,
    }


async def extract_complaint(text: str) -> Extraction:
    """
    Entry point for /extract_symptoms. EXTRACTION_BACKEND=clef tries Cloudflare
    Clef first (extraction_clef.py); if it fails — no credentials, an outage, the
    free daily allowance used up — the complaint goes through Groq, then regex,
    exactly as without Clef. Default: Groq.
    """
    if os.environ.get("EXTRACTION_BACKEND", "groq").strip().lower() == "clef":
        import extraction_clef  # imports this module, so the import stays lazy

        result = await extraction_clef.extract(text)
        if result.mode == "clef":
            return result
        logger.warning("extract: Clef unavailable, falling back to Groq")
    return await extract(text)


async def extract(text: str) -> Extraction:
    messages = [{"role": "system", "content": EXTRACTION_PROMPT}, {"role": "user", "content": text}]
    calls = [
        llm.chat_json(messages, EXTRACTION_SCHEMA, name="extract", reasoning="low", temperature=0.6)
        for _ in range(max(EXTRACTION_SAMPLES, 1))
    ]
    usage = {"input_tokens": 0, "output_tokens": 0}
    *results, safety_review = await asyncio.gather(*calls, review(text, usage))
    for r in results:
        if r is not None:
            usage["input_tokens"] += r.input_tokens
            usage["output_tokens"] += r.output_tokens
    samples = [r.data for r in results if r is not None and isinstance(r.data, dict)]

    if not samples:
        extraction = _fallback(text)
        extraction.review = safety_review
        extraction.usage = usage
        logger.info("extract: fallback mode, %d findings", len(extraction.findings))
        return extraction

    findings = apply_safety_net(_vote(samples, text), text)

    from retrieval import extract_age_sex
    regex_age, regex_sex = extract_age_sex(text)
    age = _majority([s.get("age") for s in samples], regex_age)
    sex = _majority([s.get("sex") for s in samples], regex_sex)
    extraction = Extraction(
        findings=findings,
        age=age if isinstance(age, int) and 0 <= age <= 120 else None,
        sex=sex if sex in ("M", "F") else None,
        review=safety_review,
        samples=len(samples),
        model=next(r.model for r in results if r is not None),
        usage=usage,
    )
    logger.info(
        "extract: %d/%d samples, %d present / %d absent / %d uncertain, review=%s",
        len(samples), len(results), len(extraction.codes("present")), len(extraction.codes("absent")),
        len(extraction.codes("uncertain")),
        None if safety_review is None else {k: safety_review[k] for k in ("emergency", "out_of_scope")},
    )
    return extraction
