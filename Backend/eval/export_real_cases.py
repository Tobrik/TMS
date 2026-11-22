"""
Export doctor-confirmed diary days as eval cases (same format as eval/cases.jsonl).

One case per diary day that has a doctor's diagnosis (diagnosis_feedback, latest
row wins) and an intake. The patient's text is written only with --with-text.

Usage (from Backend/):
    python eval/export_real_cases.py OUT.jsonl [--with-text] [--since YYYY-MM-DD]

DATABASE_URL comes from the environment or Backend/.env. Counts go to stderr;
patient text is never printed.
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND / ".env")

import psycopg2  # noqa: E402

import pipeline_log  # noqa: E402
import triage  # noqa: E402
from crypto_utils import decrypt_field  # noqa: E402


def export_cases(conn, with_text: bool, since: Optional[str]) -> List[dict]:
    """Cases for every diary day with doctor feedback. `since` filters on the
    latest feedback's date (YYYY-MM-DD). Read-only."""
    since_date = datetime.date.fromisoformat(since) if since else None
    cur = conn.cursor()
    cur.execute("""
        SELECT d.day_id, i.text, d.analysis_json,
               fb.diagnosis, fb.other_text, fb.zone, fb.model_top1, fb.model_zone, fb.model_version
        FROM diary_days d
        JOIN intakes i ON i.intake_id = d.intake_id
        JOIN LATERAL (
            SELECT f.diagnosis, f.other_text, f.zone, f.model_top1, f.model_zone, f.model_version, f.created_at
            FROM diagnosis_feedback f
            WHERE f.day_id = d.day_id
            ORDER BY f.feedback_id DESC
            LIMIT 1
        ) fb ON TRUE
        WHERE %s::date IS NULL OR fb.created_at::date >= %s::date
        ORDER BY d.day_id
    """, (since_date, since_date))
    rows = cur.fetchall()

    # Patient-confirmed symptoms: value > 0 reported present, value 0 denied.
    symptoms = {}
    day_ids = [r[0] for r in rows]
    if day_ids:
        cur.execute(
            "SELECT day_id, symptom_code, value FROM diary_symptoms WHERE day_id = ANY(%s) ORDER BY day_id, symptom_code",
            (day_ids,),
        )
        for day_id, code, value in cur.fetchall():
            present, absent = symptoms.setdefault(day_id, ([], []))
            (present if value > 0 else absent).append(code)

    cases = []
    for day_id, intake_text, analysis_json, diagnosis, other_text, doctor_zone, model_top1, model_zone, model_version in rows:
        text = decrypt_field(intake_text) or ""
        analysis = json.loads(decrypt_field(analysis_json) or "{}") if analysis_json else {}
        present, absent = symptoms.get(day_id, ([], []))
        other = diagnosis == "other"
        age, sex = analysis.get("age"), analysis.get("sex")
        cases.append({
            "id": f"r{day_id}",
            "lang": pipeline_log.detect_language(text),
            "text": text if with_text else "",
            "present": present,
            "absent": absent,
            "equivalent": {},
            "also_ok": [],
            "age_stated": age,
            "sex_stated": sex,
            "pathology": None if other else diagnosis,
            "tier": None if other else triage.tier_of(diagnosis),
            "out_of_scope": other,
            "likely_condition": decrypt_field(other_text) if other else None,
            "age": age,
            "sex": sex,
            "record": None,
            "doctor_zone": doctor_zone,
            "model_top1": model_top1,
            "model_zone": model_zone,
            "model_version": model_version,
        })
    return cases


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export doctor-confirmed diary days as eval cases (JSONL).")
    parser.add_argument("out", help="output .jsonl path")
    parser.add_argument("--with-text", action="store_true",
                        help="include the patient's complaint text (sensitive; default writes \"\")")
    parser.add_argument("--since", metavar="YYYY-MM-DD",
                        help="only cases whose latest doctor feedback is on or after this date")
    args = parser.parse_args(argv)
    if args.since:
        try:
            datetime.date.fromisoformat(args.since)
        except ValueError:
            parser.error("--since must be a date like 2026-11-01")

    url = os.environ.get("DATABASE_URL")
    if not url:
        parser.error("DATABASE_URL is not set (environment or Backend/.env)")

    conn = psycopg2.connect(url)
    try:
        cases = export_cases(conn, args.with_text, args.since)
    finally:
        conn.close()

    with open(args.out, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    out_of_scope = sum(c["out_of_scope"] for c in cases)
    print(f"exported {len(cases)} cases ({out_of_scope} out of scope, "
          f"text {'included' if args.with_text else 'omitted'}) to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
