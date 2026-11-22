"""
Structured pipeline logs: one JSON line per stage, so a single complaint can be
followed end to end in the Vercel logs.

    PIPELINE {"stage":"extract","intake":12,...}
    PIPELINE {"stage":"questions","intake":12,"round":1,...}
    PIPELINE {"stage":"analysis","intake":12,"day":40,...}
    PIPELINE {"stage":"explain","intake":12,"day":40,...}

Search the logs for `PIPELINE` (or `"intake":12`) and paste them into
`python eval/trace.py` for a readable timeline.

Privacy: by default no patient text is logged — only evidence codes, labels and
numbers. LOG_PATIENT_TEXT=true adds the complaint and the quotes behind each
finding; use it only while debugging (Vercel stores logs on its servers).
"""

import json
import logging
import os
import re
from typing import Dict, Iterable, List

logger = logging.getLogger("tms.pipeline")

_KAZAKH_LETTERS = re.compile(r"[әғқңөұүһі]")
_CYRILLIC = re.compile(r"[а-яё]")


def patient_text_allowed() -> bool:
    return os.environ.get("LOG_PATIENT_TEXT", "").strip().lower() in ("1", "true", "yes")


def detect_language(text: str) -> str:
    """Script-based guess: kk if any Kazakh-specific letter, ru if Cyrillic, else en."""
    lowered = text.lower()
    if _KAZAKH_LETTERS.search(lowered):
        return "kk"
    return "ru" if _CYRILLIC.search(lowered) else "en"


def label(code: str) -> str:
    """Short English label for an evidence token, e.g. 'E_55_@_V_84' -> 'Pain location: flank(R)'."""
    from extraction import evidence_label, short_label
    from ml_model import EVIDENCES_META

    if "_@_" in code or code not in EVIDENCES_META:
        text = evidence_label(code)["en"]
    else:
        text = short_label(code)
    return text if len(text) <= 60 else text[:57] + "..."


def labelled(codes: Iterable[str]) -> List[str]:
    return [f"{code} {label(code)}" for code in codes]


def top(distribution: Dict[str, float], n: int = 5) -> List[list]:
    return [[name, round(p, 3)] for name, p in list(distribution.items())[:n]]


def log_stage(stage: str, **fields) -> None:
    logger.info("PIPELINE %s", json.dumps({"stage": stage, **fields}, ensure_ascii=False,
                                          default=str, separators=(",", ":")))
