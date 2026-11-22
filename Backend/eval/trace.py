"""
Readable timeline of PIPELINE log lines (see pipeline_log.py), one block per case.

Copy the backend's logs from Vercel (Logs → search "PIPELINE" → copy, or
`vercel logs <deployment-url> > logs.txt`), then:

    python eval/trace.py logs.txt            # every case
    python eval/trace.py logs.txt --intake 12
    pbpaste | python eval/trace.py           # straight from the clipboard (macOS)

Accepts raw log lines, lines with timestamps/levels in front, or Vercel's JSON
export (one object per line with a "message" field).
"""

import argparse
import json
import sys
from typing import Dict, Iterable, List, Optional


def parse(lines: Iterable[str]) -> List[dict]:
    records = []
    for line in lines:
        line = line.strip()
        if line.startswith("{") and '"message"' in line:  # Vercel JSON export
            try:
                line = json.loads(line).get("message", "")
            except json.JSONDecodeError:
                pass
        marker = line.find("PIPELINE {")
        if marker < 0:
            continue
        try:
            records.append(json.loads(line[marker + len("PIPELINE "):]))
        except json.JSONDecodeError:
            continue  # truncated line
    return records


def _pct(p: float) -> str:
    return f"{p:.0%}" if p >= 0.01 else f"{p:.1%}"


def _top(rows: List[list]) -> str:
    return ", ".join(f"{name} {_pct(p)}" for name, p in rows)


def _review(review: dict) -> str:
    if not review or not review.get("available"):
        return "UNAVAILABLE — the review call failed (see LLM warnings), no out-of-scope / emergency signal"
    parts = [f"out_of_scope={review.get('out_of_scope')}"]
    if review.get("likely_condition"):
        parts.append(f"likely={review['likely_condition']}")
    emergency = f"emergency={review.get('emergency')}"
    if review.get("emergency_probability") is not None:
        emergency += f" (p={review['emergency_probability']})"
    parts.append(emergency + (" verified" if review.get("verified") else ""))
    if review.get("reason"):
        parts.append(f"“{review['reason']}”")
    return "  ".join(parts)


def render(records: List[dict], intake: Optional[int] = None) -> str:
    day_to_intake = {r["day"]: r.get("intake") for r in records if r["stage"] == "analysis"}
    cases: Dict[object, List[dict]] = {}
    for r in records:
        key = r.get("intake") if r["stage"] != "explain" else day_to_intake.get(r.get("day"), f"day {r.get('day')}")
        cases.setdefault(key, []).append(r)

    out = []
    for key, stages in cases.items():
        if intake is not None and key != intake:
            continue
        out.append("═" * 100)
        for r in stages:
            stage = r["stage"]
            if stage == "extract":
                tokens = r.get("tokens") or {}
                out.append(f"intake {r.get('intake')} · patient {r.get('patient')} · {r.get('lang')} · "
                           f"{r.get('chars')} chars · {r.get('backend')} {r.get('model') or ''} · {r.get('ms')} ms · "
                           f"{tokens.get('input_tokens', 0):,} in / {tokens.get('output_tokens', 0):,} out tokens")
                if r.get("text"):
                    out.append(f"  text      “{r['text']}”")
                symbol = {"present": "+", "absent": "−", "uncertain": "?"}
                for f in r.get("findings", []):
                    extra = f" ({f['confidence']})" if "confidence" in f else ""
                    quote = f"  «{f['quote']}»" if f.get("quote") else ""
                    out.append(f"  found  {symbol.get(f['status'], ' ')} {f['code']:<14} {f['label']}{extra}  [{f['source']}]{quote}")
                out.append(f"  age/sex   {r.get('age')} / {r.get('sex')}    alert: {r.get('alert')}")
                out.append(f"  review    {_review(r.get('review'))}")
            elif stage == "questions":
                out.append(f"  round {r.get('round')}   top: {_top(r.get('top', []))}  (zone {r.get('zone')})")
                if r.get("stop"):
                    out.append(f"            stop: {r['stop']}")
                for code, label, score in r.get("asked", []):
                    out.append(f"            asked {code:<14} {label}  (info gain {score})")
            elif stage == "analysis":
                out.append(f"  ANALYSIS  day {r.get('day')} · zone {r.get('zone').upper()} · red mass {_pct(r.get('red_mass', 0))}"
                           f" · yellow mass {_pct(r.get('yellow_mass', 0))} · age/sex {r.get('age')}/{r.get('sex')}")
                out.append(f"            present: {'; '.join(r.get('present', [])) or '—'}")
                out.append(f"            denied:  {'; '.join(r.get('absent', [])) or '—'}")
                out.append(f"            top: {_top(r.get('top', []))}")
                for disease, fit in (r.get("fit") or {}).items():
                    cells = ", ".join(f"{code} {p}" + (" (never seen)" if p <= 0.03 else "") for code, p in fit.items())
                    out.append(f"            fit {disease}: {cells}")
                reasons = "; ".join(f"{x['zone']}: {x['code']}" + (f" {x.get('evidence') or x.get('value') or ''}")
                                    for x in r.get("reasons", []))
                out.append(f"            zone reasons: {reasons or '—'}")
                out.append(f"            review: {_review(r.get('review'))}")
                for note in r.get("adjustments", []):
                    out.append(f"            rule: {note}")
            elif stage == "explain":
                out.append(f"  explain   day {r.get('day')} · {r.get('lang')} · patient text: {r.get('patient_model')}"
                           f" · doctor text: {r.get('doctor_model')}")
                for disease, pairs in (r.get("attributions") or {}).items():
                    out.append(f"            {disease}: " + ", ".join(f"{ev} {d:+.0%}" for ev, d in pairs))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="log file (default: stdin)")
    ap.add_argument("--intake", type=int, help="only this intake id")
    args = ap.parse_args()
    lines = open(args.file, encoding="utf-8") if args.file else sys.stdin
    records = parse(lines)
    if not records:
        sys.exit("No PIPELINE lines found.")
    print(render(records, args.intake))


if __name__ == "__main__":
    main()
