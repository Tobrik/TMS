"""
Compare symptom-extraction backends end to end on eval/cases.jsonl:
free text → findings → naive Bayes differential → triage.

Backends:
  gold        the case's own labels (upper bound: perfect extraction)
  regex       v1 regex + BM25 fallback (no LLM)
  groq        extraction.extract — gpt-oss on Groq (GROQ_API_KEY)
  clef-flash  extraction_clef.extract — Cloudflare Clef-flash (CF_ACCOUNT_ID, CF_API_TOKEN)
  clef        the same with the 27B Clef
  <base>+net  the cached <base> results after extraction.apply_denial_net (rule-based
              denials from negation.py). Never calls an API, even without --score-only.
              Reads the base cache of the same --tag: groq+net with --tag v3 reads
              results/groq-v3.jsonl and is shown as groq-v3+net.

Results are cached per backend in eval/results/<backend>.jsonl (or <backend>-<tag>.jsonl
with --tag): a rerun only calls the API for cases not done yet, so free quotas are never
spent twice. A backend stops at its first failure (usually a used-up daily quota); rerun
later to resume.

Flags:
  --tag NAME       separate cache and column (<backend>-NAME), e.g. to compare prompt versions
  --no-review      extraction.extract(..., with_review=False): no safety-review call, review
                   cells show "—". Groq only (extraction_clef has no review switch).
  --only-denials   only cases whose gold labels include an explicit denial (non-empty absent)

Usage (from Backend/):
    python eval/compare_extraction.py --backends regex,groq,clef-flash [--limit 10] [--errors]
    python eval/compare_extraction.py --backends groq,groq+net --tag v3 --no-review --score-only

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
from extraction import Finding  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# $ per million tokens (input, output), 2026-10
PRICES = {"groq": (0.15, 0.60), "clef-flash": (0.038, 0.0), "clef": (0.24, 0.0)}

NET_SUFFIX = "+net"
LOCAL_BACKENDS = ("gold", "regex")  # computed without any API, so cached locally on demand


def cache_name(backend: str, tag: str = "") -> str:
    return f"{backend}-{tag}" if tag else backend


def split_net(name: str):
    """'groq-v3+net' -> ('groq-v3', True); 'groq' -> ('groq', False)."""
    return (name[:-len(NET_SUFFIX)], True) if name.endswith(NET_SUFFIX) else (name, False)


def price_key(name: str) -> str:
    """Price table key for a cache name such as 'clef-flash-v3+net'."""
    base = split_net(name)[0]
    for key in sorted(PRICES, key=len, reverse=True):
        if base == key or base.startswith(key + "-"):
            return key
    return ""


# ─── Running a backend ───────────────────────────────────────────────────────

async def run_case(backend: str, case: dict, no_review: bool = False) -> dict:
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
            ext = await extraction.extract(case["text"], with_review=not no_review)
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


def load_cached(name: str) -> Dict[str, dict]:
    """Cached records of one cache name (a backend, optionally with its --tag). No API calls."""
    path = RESULTS / f"{name}.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def apply_net(record: dict, case: dict) -> dict:
    """Cached base result after the rule-based denial net. Rebuilds the Finding objects
    from the saved present/absent/uncertain lists, so no API is called."""
    findings = ([Finding(c, "present") for c in record["present"]]
                + [Finding(c, "absent") for c in record["absent"]]
                + [Finding(c, "uncertain") for c in record["uncertain"]])
    findings = extraction.apply_denial_net(findings, case["text"])
    return {**record,
            "present": [f.code for f in findings if f.status == "present"],
            "absent": [f.code for f in findings if f.status == "absent"],
            "uncertain": [f.code for f in findings if f.status == "uncertain"]}


async def run_backend(backend: str, cases: List[dict], fresh: bool, pause: float,
                      tag: str = "", no_review: bool = False) -> Dict[str, dict]:
    RESULTS.mkdir(exist_ok=True)
    name = cache_name(backend, tag)
    path = RESULTS / f"{name}.jsonl"
    cached = {} if fresh else load_cached(name)
    if fresh and path.exists():
        path.unlink()
    todo = [c for c in cases if c["id"] not in cached]
    if todo:
        print(f"  {name}: running {len(todo)} case(s), {len(cached)} cached", flush=True)
    for i, case in enumerate(todo):
        try:
            record = await run_case(backend, case, no_review)
        except Exception as e:
            print(f"  {name}: stopped at {case['id']} — {e}. Rerun later to resume.", flush=True)
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
    price_in, price_out = PRICES.get(price_key(backend), (0.0, 0.0))
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
    ap.add_argument("--tag", default="", help="cache and column suffix: <backend>-<tag>")
    ap.add_argument("--no-review", action="store_true",
                    help="skip the safety-review call (groq); review cells show —")
    ap.add_argument("--only-denials", action="store_true",
                    help="only cases whose gold labels include an explicit denial")
    args = ap.parse_args()

    # Ride out Groq's per-minute token limit instead of falling back to another model
    llm.MAX_RETRY_WAIT = 75.0
    llm.MAX_RETRIES = 6
    llm.TEXT_MODELS[:] = llm.TEXT_MODELS[:1]

    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only_denials:
        cases = [c for c in cases if c["absent"]]
    if args.limit:
        cases = cases[:args.limit]
    by_id = {c["id"]: c for c in cases}

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    results: Dict[str, Dict[str, dict]] = {}  # display name -> {case id: record}
    for b in backends:
        base, is_net = split_net(b)
        display = cache_name(base, args.tag) + (NET_SUFFIX if is_net else "")
        if is_net:
            # Never an API call: the base comes from its cache (or runs locally for gold/regex)
            if base in LOCAL_BACKENDS:
                base_results = await run_backend(base, cases, False, 0.0, args.tag)
            else:
                base_results = load_cached(cache_name(base, args.tag))
            if base.startswith("clef"):
                base_results = {i: rederive(r, by_id[i]) for i, r in base_results.items() if i in by_id}
            results[display] = {i: apply_net(r, by_id[i]) for i, r in base_results.items() if i in by_id}
        else:
            if args.score_only and base != "gold":
                found = load_cached(display)
            else:
                found = await run_backend(base, cases, args.fresh, args.pause, args.tag, args.no_review)
            if base.startswith("clef"):
                found = {i: rederive(r, by_id[i]) for i, r in found.items() if i in by_id}
            results[display] = found

    # A fair comparison scores every backend on the same cases
    if not args.all_cases:
        common = set.intersection(*(set(r) for r in results.values())) if results else set()
        if len(common) < len(cases):
            print(f"  scoring the {len(common)} case(s) finished by every backend (--all-cases to change)")
        cases = [c for c in cases if c["id"] in common]

    scores = []
    for name in results:
        if args.errors:
            print(f"\n  {name} misses:")
        scored = score(name, cases, results[name], args.errors)
        if scored:
            scores.append(scored)
        if args.confirm and name != "gold":
            fixed = {i: confirmed(r, by_id[i]) for i, r in results[name].items() if i in by_id}
            scored = score(f"{name}+confirm", cases, fixed, False)
            if scored:
                scores.append(scored)
    if scores:
        print_table(scores)


if __name__ == "__main__":
    asyncio.run(main())
