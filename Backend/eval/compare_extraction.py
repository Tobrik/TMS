"""
Compare symptom-extraction backends end to end on eval/cases.jsonl:
free text → findings → naive Bayes differential → triage.

Backends:
  gold        the case's own labels (upper bound: perfect extraction)
  regex       v1 regex + BM25 fallback (no LLM)
  groq        extraction.extract — gpt-oss on Groq (GROQ_API_KEY)
  clef-flash  extraction_clef.extract — Cloudflare Clef-flash (CF_ACCOUNT_ID, CF_API_TOKEN)
  clef        the same with the 27B Clef

Results are cached per backend in eval/results/<backend>.jsonl: a rerun only calls
the API for cases not done yet, so free quotas are never spent twice. A backend
stops at its first failure (usually a used-up daily quota); rerun later to resume.

Usage (from Backend/):
    python eval/compare_extraction.py --backends regex,groq,clef-flash [--limit 10] [--errors]

Case format (one JSON object per line):
  text, lang, present / absent      gold codes the text states / explicitly denies
  equivalent {code: [alternatives]} other codes that count as the same finding
  also_ok [codes, "E_55_*"]          defensible extras, not counted as false positives
  pathology, tier, record, age, sex DDXPlus ground truth (None for out-of-scope cases)
  out_of_scope, likely_condition    for complaints outside the 49 conditions
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import extraction  # noqa: E402
import llm  # noqa: E402
import ml_model as m  # noqa: E402
import triage  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# $ per million tokens (input, output), 2026-10
PRICES = {"groq": (0.15, 0.60), "clef-flash": (0.038, 0.0), "clef": (0.24, 0.0)}


# ─── Running a backend ───────────────────────────────────────────────────────

async def run_case(backend: str, case: dict) -> dict:
    started = time.perf_counter()
    if backend == "gold":
        review = {"out_of_scope": case["out_of_scope"], "likely_condition": case.get("likely_condition")}
        result = {"present": case["present"], "absent": case["absent"], "uncertain": [],
                  "age": case.get("age_stated"), "sex": case.get("sex_stated"), "review": review,
                  "usage": {"input_tokens": 0, "output_tokens": 0}, "mode": "gold"}
    else:
        if backend == "regex":
            ext = extraction._fallback(case["text"])
        elif backend == "groq":
            ext = await extraction.extract(case["text"])
        else:
            import extraction_clef
            ext = await extraction_clef.extract(case["text"], model=backend)
        if backend != "regex" and ext.mode == "fallback":
            raise RuntimeError(f"{backend} unavailable (credentials, rate limit or daily quota)")
        result = {"present": ext.codes("present"), "absent": ext.codes("absent"), "uncertain": ext.codes("uncertain"),
                  "age": ext.age, "sex": ext.sex, "review": ext.review, "usage": ext.usage, "mode": ext.mode,
                  "raw": ext.raw}
    return {"id": case["id"], **result, "latency_s": round(time.perf_counter() - started, 2)}


def rederive(record: dict, case: dict) -> dict:
    """Re-interpret a cached Clef result from its saved probabilities with the
    current thresholds in extraction_clef — re-scoring costs no API calls."""
    if not record.get("raw"):
        return record
    import extraction_clef
    ext = extraction_clef.interpret(record["raw"], case["text"], record.get("model") or "clef", record["usage"])
    return {**record, "present": ext.codes("present"), "absent": ext.codes("absent"),
            "uncertain": ext.codes("uncertain"), "sex": ext.sex, "review": ext.review}


def load_cached(backend: str) -> Dict[str, dict]:
    path = RESULTS / f"{backend}.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


async def run_backend(backend: str, cases: List[dict], fresh: bool, pause: float) -> Dict[str, dict]:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{backend}.jsonl"
    cached = {} if fresh else load_cached(backend)
    if fresh and path.exists():
        path.unlink()
    todo = [c for c in cases if c["id"] not in cached]
    if todo:
        print(f"  {backend}: running {len(todo)} case(s), {len(cached)} cached", flush=True)
    for i, case in enumerate(todo):
        try:
            record = await run_case(backend, case)
        except Exception as e:
            print(f"  {backend}: stopped at {case['id']} — {e}. Rerun later to resume.", flush=True)
            break
        cached[case["id"]] = record
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if pause and backend not in ("gold", "regex") and i < len(todo) - 1:
            await asyncio.sleep(pause)
    return cached


# ─── Scoring ─────────────────────────────────────────────────────────────────

def _matches(code: str, predicted: set, case: dict) -> bool:
    return code in predicted or any(alt in predicted for alt in case["equivalent"].get(code, []))


def _acceptable(code: str, case: dict) -> bool:
    if code in case["present"] or code in case["absent"]:
        return True
    if any(code in alts for alts in case["equivalent"].values()):
        return True
    return any(code == ok or (ok.endswith("*") and code.startswith(ok[:-1])) for ok in case["also_ok"])


def confirmed(result: dict, case: dict) -> dict:
    """
    The result after the confirmation step (chat: "here's what I understood"),
    assuming the patient corrects every chip: wrong 'present' chips are switched
    to 'no' (the patient doesn't have it), right 'uncertain' chips to 'yes', wrong
    denials to 'yes'. Missed findings stay missed — the patient can't add chips.
    `fixes` counts the taps needed: the burden extra findings put on the patient.
    """
    gold_absent = lambda code: any(code == g or code in case["equivalent"].get(g, []) for g in case["absent"])
    present, absent, fixes = [], list(result["absent"]), 0
    for code in result["present"]:
        if _acceptable(code, case) and not gold_absent(code):
            present.append(code)
        else:
            absent.append(code)
            fixes += 1
    for code in result["uncertain"]:
        if _acceptable(code, case) and not gold_absent(code):
            present.append(code)
            fixes += 1
    for code in list(absent):
        if any(_matches(g, {code}, case) for g in case["present"]):
            absent.remove(code)
            present.append(code)
            fixes += 1
    return {**result, "present": present, "absent": absent, "uncertain": [], "fixes": fixes}


def _ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def simulate_questions(present: List[str], absent: List[str], age, sex, record: set):
    """Two rounds of three questions, answered truthfully from the DDXPlus record."""
    present, absent = list(present), list(absent)
    for round_num in (1, 2):
        res = m.qa_engine.get_questions(present, age, sex, round_num=round_num, n_questions=3, absent=absent)
        if res.should_stop:
            break
        for q in res.questions:
            (present if q.evidence_id in record else absent).append(q.evidence_id)
    return present, absent


def score(backend: str, cases: List[dict], results: Dict[str, dict], show_errors: bool) -> Optional[dict]:
    done = [c for c in cases if c["id"] in results]
    if not done:
        return None
    s = {k: 0 for k in ("tp", "fn", "fp", "recovered", "absent_gold", "absent_tp", "negation_errors",
                        "redflag_gold", "redflag_tp", "top1", "top3", "q_top1", "q_top3", "in_scope",
                        "red", "red_missed", "q_red_missed", "green", "over", "q_over",
                        "oos", "oos_flagged", "false_oos", "emergency_red", "false_emergency")}
    by_lang: Dict[str, List[int]] = {}
    tokens_in = tokens_out = latency = 0.0

    for case in done:
        r = results[case["id"]]
        present, absent, uncertain = set(r["present"]), set(r["absent"]), set(r["uncertain"])
        tp = sum(_matches(g, present, case) for g in case["present"])
        fn_codes = [g for g in case["present"] if not _matches(g, present, case)]
        fp_codes = [p for p in present if not _acceptable(p, case)]
        s["tp"] += tp
        s["fn"] += len(fn_codes)
        s["fp"] += len(fp_codes)
        s["recovered"] += sum(_matches(g, uncertain, case) for g in fn_codes)
        s["absent_gold"] += len(case["absent"])
        s["absent_tp"] += sum(_matches(g, absent, case) for g in case["absent"])
        negations = [g for g in case["absent"] if _matches(g, present, case)]
        s["negation_errors"] += len(negations)
        red_gold = [g for g in case["present"] if g in extraction.SAFETY_NET]
        s["redflag_gold"] += len(red_gold)
        s["redflag_tp"] += sum(_matches(g, present, case) for g in red_gold)
        lang = by_lang.setdefault(case["lang"], [0, 0, 0])
        lang[0] += tp
        lang[1] += len(fn_codes)
        lang[2] += len(fp_codes)

        review = r.get("review") or {}
        if case["out_of_scope"]:
            s["oos"] += 1
            s["oos_flagged"] += bool(review.get("out_of_scope"))
        else:
            s["false_oos"] += bool(review.get("out_of_scope"))
        if case["tier"] == "red":
            s["emergency_red"] += bool(review.get("emergency"))
        elif case["tier"] == "green":
            s["false_emergency"] += bool(review.get("emergency"))

        if show_errors and (fn_codes or fp_codes or negations):
            print(f"    {case['id']} [{case['lang']}] missed {fn_codes} extra {fp_codes} negation-errors {negations}")

        if case["pathology"] is None:
            continue
        s["in_scope"] += 1
        truth = m._GROUP_MAP.get(case["pathology"], case["pathology"])
        valid_present = [e for e in r["present"] if e in m._feat_idx]
        valid_absent = [e for e in r["absent"] if e in m._feat_idx]
        for prefix, (ev, ab) in (("", (valid_present, valid_absent)),
                                 ("q_", simulate_questions(valid_present, valid_absent, r["age"], r["sex"],
                                                           set(case["record"])) if valid_present else ([], []))):
            dist = m.predict_distribution(ev, r["age"], r["sex"], ab) if ev else {}
            top = list(dist)
            s[prefix + "top1"] += bool(top) and top[0] == truth
            s[prefix + "top3"] += truth in top[:3]
            zone = triage.assess(dist, ev, review).zone if ev else "green"
            if case["tier"] == "red":
                s[prefix + "red_missed"] += zone != "red"
            elif case["tier"] == "green":
                s[prefix + "over"] += zone == "red"
        s["red"] += case["tier"] == "red"
        s["green"] += case["tier"] == "green"

    fixes = sum(results[c["id"]].get("fixes", 0) for c in done)
    for r in (results[c["id"]] for c in done):
        tokens_in += r["usage"]["input_tokens"]
        tokens_out += r["usage"]["output_tokens"]
        latency += r["latency_s"]

    n = len(done)
    # gold and regex have no safety review: show "—" rather than 0/9
    reviewed = any("emergency" in (results[c["id"]].get("review") or {}) for c in done)
    review_cell = (lambda hit, total: f"{hit}/{total}") if reviewed else (lambda hit, total: "—")
    precision, recall = _ratio(s["tp"], s["tp"] + s["fp"]), _ratio(s["tp"], s["tp"] + s["fn"])
    price_in, price_out = PRICES.get(backend, (0.0, 0.0))
    return {
        "backend": backend, "cases": n,
        "precision": precision, "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "recall_with_confirm": _ratio(s["tp"] + s["recovered"], s["tp"] + s["fn"]),
        "denials_caught": _ratio(s["absent_tp"], s["absent_gold"]),
        "negation_errors": s["negation_errors"],
        "redflag_recall": _ratio(s["redflag_tp"], s["redflag_gold"]),
        "f1_by_lang": {lang: _ratio(2 * tp, 2 * tp + fn + fp) for lang, (tp, fn, fp) in sorted(by_lang.items())},
        "top1": _ratio(s["top1"], s["in_scope"]), "top3": _ratio(s["top3"], s["in_scope"]),
        "q_top1": _ratio(s["q_top1"], s["in_scope"]), "q_top3": _ratio(s["q_top3"], s["in_scope"]),
        "red_missed": f"{s['red_missed']}/{s['red']}", "q_red_missed": f"{s['q_red_missed']}/{s['red']}",
        "over": f"{s['over']}/{s['green']}", "q_over": f"{s['q_over']}/{s['green']}",
        "oos_flagged": f"{s['oos_flagged']}/{s['oos']}" if reviewed or backend == "gold" else "—",
        "false_oos": f"{s['false_oos']}/{n - s['oos']}" if reviewed or backend == "gold" else "—",
        "emergency_red": review_cell(s["emergency_red"], sum(c["tier"] == "red" for c in done)),
        "false_emergency": review_cell(s["false_emergency"], sum(c["tier"] == "green" for c in done)),
        "fixes": fixes / n if any("fixes" in results[c["id"]] for c in done) else float("nan"),
        "tokens": f"{tokens_in / n:,.0f} in / {tokens_out / n:,.0f} out",
        "usd_per_1000": (tokens_in * price_in + tokens_out * price_out) / n / 1000,
        "latency_s": latency / n,
    }


ROWS = [
    ("cases", "Cases", "{}"),
    ("precision", "Findings precision", "{:.0%}"),
    ("recall", "Findings recall", "{:.0%}"),
    ("f1", "Findings F1", "{:.0%}"),
    ("recall_with_confirm", "Recall incl. 'please confirm'", "{:.0%}"),
    ("denials_caught", "Denials caught", "{:.0%}"),
    ("negation_errors", "Denied → marked present", "{}"),
    ("redflag_recall", "Red-flag recall", "{:.0%}"),
    ("f1_by_lang", "F1 by language", None),
    ("top1", "Top-1 (extraction only)", "{:.0%}"),
    ("top3", "Top-3 (extraction only)", "{:.0%}"),
    ("q_top1", "Top-1 (+ 2x3 questions)", "{:.0%}"),
    ("q_top3", "Top-3 (+ 2x3 questions)", "{:.0%}"),
    ("red_missed", "Red cases not sent red", "{}"),
    ("q_red_missed", "  … after questions", "{}"),
    ("over", "Green cases sent red", "{}"),
    ("q_over", "  … after questions", "{}"),
    ("emergency_red", "Emergency flagged (red cases)", "{}"),
    ("false_emergency", "Emergency flagged (green)", "{}"),
    ("fixes", "Chips patient must fix", "{:.1f}"),
    ("oos_flagged", "Out-of-scope flagged", "{}"),
    ("false_oos", "Out-of-scope false flags", "{}"),
    ("tokens", "Tokens / complaint", "{}"),
    ("usd_per_1000", "$ per 1,000 complaints", "${:.2f}"),
    ("latency_s", "Latency / complaint", "{:.1f}s"),
]


def print_table(scores: List[dict]) -> None:
    width, col = 32, 22
    print("\n" + "".ljust(width) + "".join(s["backend"].rjust(col) for s in scores))
    for key, label, fmt in ROWS:
        cells = []
        for s in scores:
            value = s[key]
            if key == "f1_by_lang":
                text = " ".join(f"{lang}:{v:.0%}" for lang, v in value.items())
            elif isinstance(value, float) and value != value:  # NaN
                text = "—"
            else:
                text = fmt.format(value)
            cells.append(text.rjust(col))
        print(label.ljust(width) + "".join(cells))


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=str(HERE / "cases.jsonl"))
    ap.add_argument("--backends", default="gold,regex")
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases")
    ap.add_argument("--fresh", action="store_true", help="ignore cached results")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds between API cases (pacing)")
    ap.add_argument("--errors", action="store_true", help="print per-case misses")
    ap.add_argument("--score-only", action="store_true", help="use cached results only; call no API")
    ap.add_argument("--confirm", action="store_true",
                    help="also score each backend after the patient's confirmation step (see confirmed())")
    ap.add_argument("--all-cases", action="store_true",
                    help="score each backend on every case it finished (default: only cases all backends finished)")
    args = ap.parse_args()

    # Ride out Groq's per-minute token limit instead of falling back to another model
    llm.MAX_RETRY_WAIT = 75.0
    llm.MAX_RETRIES = 6
    llm.TEXT_MODELS[:] = llm.TEXT_MODELS[:1]

    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        cases = cases[:args.limit]

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    results = {
        b: (load_cached(b) if args.score_only and b != "gold" else await run_backend(b, cases, args.fresh, args.pause))
        for b in backends
    }
    by_id = {c["id"]: c for c in cases}
    for b in backends:
        if b.startswith("clef"):
            results[b] = {i: rederive(r, by_id[i]) for i, r in results[b].items() if i in by_id}

    # A fair comparison scores every backend on the same cases
    if not args.all_cases:
        common = set.intersection(*(set(r) for r in results.values())) if results else set()
        if len(common) < len(cases):
            print(f"  scoring the {len(common)} case(s) finished by every backend (--all-cases to change)")
        cases = [c for c in cases if c["id"] in common]

    scores = []
    for backend in backends:
        if args.errors:
            print(f"\n  {backend} misses:")
        scored = score(backend, cases, results[backend], args.errors)
        if scored:
            scores.append(scored)
        if args.confirm and backend not in ("gold",):
            by_id = {c["id"]: c for c in cases}
            fixed = {i: confirmed(r, by_id[i]) for i, r in results[backend].items() if i in by_id}
            scored = score(f"{backend}+confirm", cases, fixed, False)
            if scored:
                scores.append(scored)
    if scores:
        print_table(scores)


if __name__ == "__main__":
    asyncio.run(main())
