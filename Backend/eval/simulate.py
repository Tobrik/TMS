"""
Offline end-to-end simulation of the diagnostic pipeline on DDXPlus test patients.
No LLM involved: measures question selection + model + triage.

Each simulated patient:
  1. volunteers what a patient typically says: their chief complaint
     (INITIAL_EVIDENCE), up to 2 other symptoms, and where it hurts if in pain;
  2. answers each follow-up question from their full DDXPlus record, but
     sometimes wrongly (--answer-error: yes/no flipped) or with "don't know"
     (--dont-know);
  3. gets a differential and a triage zone.

Usage (from Backend/):
    python eval/simulate.py <path/to/release_test_patients> [--n 3000] [--seed 7]
    python eval/simulate.py <path> --n 3000 --pipelines NB --calibration
    python eval/simulate.py <path> --noise 0.3 --answer-error 0.1 --dont-know 0.2 --calibration

Metrics: top-1 / top-3 accuracy, questions asked, don't-know answers, and triage
against the true disease's tier: red under-triage (true red, not sent red),
severe under-triage (true red, sent green) and over-triage (true green, sent red).
--calibration adds a reliability table, ECE and ECE after temperature scaling
of the final distribution (measurement only; questions and triage unchanged),
and accuracy by the engine's stop reason.
"""

import argparse
import ast
import csv
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ml_model as m  # noqa: E402
import qa_engine  # noqa: E402
import triage  # noqa: E402

EXTRA_VOLUNTEERED = 2
NOISE = 0.0          # probability of one false-positive symptom (set with --noise)
ANSWER_ERROR = 0.0   # probability a follow-up yes/no answer is flipped (--answer-error)
DONT_KNOW = 0.0      # probability a follow-up question is answered "don't know" (--dont-know)
TEMPERATURES = (0.5, 0.75, 1.0, 1.5, 2.0)
ASKABLE_SYMPTOMS = sorted(
    e for e, meta in m.EVIDENCES_META.items()
    if meta["data_type"] == "B" and not meta.get("is_antecedent", False) and e in m._feat_idx
)


# ─── v1 behaviour, reproduced for comparison ─────────────────────────────────

def legacy_zone(distribution: Dict[str, float], evidences: List[str]) -> str:
    """classify_zone() from v1: top-1 name + top-1 score as urgency."""
    ev_set = set(evidences)
    top1, score = next(iter(distribution.items()))
    if ev_set & triage.RED_FLAG_EVIDENCES or any(c <= ev_set for c in triage.RED_FLAG_COMBOS):
        return "red"
    has_yellow = bool(ev_set & triage.YELLOW_FLAG_EVIDENCES)
    if top1 in triage.CURATED_RED or score > 0.6:
        return "red"
    if top1 in triage.CURATED_YELLOW or score > 0.5 or has_yellow:
        return "yellow"
    return "green"


@dataclass
class Pipeline:
    name: str
    strategy: Optional[str]      # "jsd" | "eig" | None (no questions)
    rounds: int
    record_no: bool
    posterior: str               # "rf" | "nb"
    triage: str                  # "legacy" | "v2"
    oracle: bool = False         # give the model the full patient record
    floor: float = m._LIKELIHOOD_FLOOR  # naive Bayes smoothing
    demographics: bool = True    # patient states age and sex
    stop: float = qa_engine.STOP_CONFIDENCE  # qa_engine.STOP_CONFIDENCE, patched for this run
    max_rounds: int = qa_engine.MAX_ROUNDS   # qa_engine.MAX_ROUNDS, patched for this run


PIPELINES = [
    Pipeline("oracle: RF, full record", None, 0, False, "rf", "v2", oracle=True),
    Pipeline("oracle: NB, full record", None, 0, False, "nb", "v2", oracle=True),
    Pipeline("v1: jsd 1x3, no's dropped, RF", "jsd", 1, False, "rf", "legacy"),
    Pipeline("v1 questions + v2 triage", "jsd", 1, False, "rf", "v2"),
    Pipeline("eig 2x3, RF", "eig", 2, True, "rf", "v2"),
    Pipeline("eig 2x3, NB, age/sex unknown", "eig", 2, True, "nb", "v2", demographics=False),
    Pipeline("eig 2x3, NB floor .01", "eig", 2, True, "nb", "v2", floor=0.01),
    Pipeline("eig 2x3, NB floor .03 (default)", "eig", 2, True, "nb", "v2", floor=0.03),
    Pipeline("eig 2x3, NB floor .05", "eig", 2, True, "nb", "v2", floor=0.05),
    Pipeline("eig 2x3, NB, stop .80", "eig", 2, True, "nb", "v2", stop=0.80),
    Pipeline("eig 2x3, NB, stop .90", "eig", 2, True, "nb", "v2", stop=0.90),
    Pipeline("eig 3x3, NB", "eig", 3, True, "nb", "v2", max_rounds=3),
]


def select_pipelines(spec: str) -> List[Pipeline]:
    """Pipelines whose name contains any of the comma-separated substrings (case-insensitive)."""
    wanted = [s.strip().lower() for s in spec.split(",") if s.strip()]
    if not wanted:
        return PIPELINES
    return [p for p in PIPELINES if any(w in p.name.lower() for w in wanted)]


def is_default(p: Pipeline) -> bool:
    return (p.posterior, p.strategy, p.rounds, p.floor, p.demographics, p.stop, p.max_rounds) == \
        (m.POSTERIOR, "eig", 2, m._LIKELIHOOD_FLOOR, True, qa_engine.STOP_CONFIDENCE, qa_engine.MAX_ROUNDS)


def load_patients(path: str, n: int, seed: int) -> List[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["PATHOLOGY"] not in m._GEO_EXCLUDED]
    # travel-only diseases (Chagas): keep only patients who report that travel
    rows = [r for r in rows if r["PATHOLOGY"] not in m._TRAVEL_ONLY
            or m._TRAVEL_ONLY[r["PATHOLOGY"]] & set(ast.literal_eval(r["EVIDENCES"]))]
    return random.Random(seed).sample(rows, min(n, len(rows)))


def volunteered(patient: dict, record: set, rng: random.Random) -> List[str]:
    init = patient["INITIAL_EVIDENCE"]
    symptoms = sorted(
        e for e in record
        if e != init and e in m.EVIDENCES_META
        and m.EVIDENCES_META[e]["data_type"] == "B"
        and not m.EVIDENCES_META[e].get("is_antecedent", False)
    )
    said = [init] + rng.sample(symptoms, min(EXTRA_VOLUNTEERED, len(symptoms)))
    if "E_53" in said:  # in pain → patients say where (the main place, not all of them)
        locations = sorted(e for e in record if e.startswith("E_55_@_"))
        said += rng.sample(locations, min(1, len(locations)))
    if NOISE and rng.random() < NOISE:  # extraction error: one symptom they don't have
        said.append(rng.choice([e for e in ASKABLE_SYMPTOMS if e not in record]))
    return said


def answer_question(code: str, record: set, rng: random.Random) -> str:
    """The patient's answer to one follow-up question: "yes" | "no" | "dk".
    "dk" is decided first and is never flipped; otherwise the truth is flipped
    with probability ANSWER_ERROR. Draws happen only when a probability is set,
    so default runs consume the same random numbers as before."""
    if DONT_KNOW and rng.random() < DONT_KNOW:
        return "dk"
    truth = code in record
    if ANSWER_ERROR and rng.random() < ANSWER_ERROR:
        truth = not truth
    return "yes" if truth else "no"


def run(pipeline: Pipeline, patients: List[dict], seed: int) -> List[dict]:
    m.LIKELIHOODS[:] = m._build_likelihood_matrix(pipeline.floor)  # in place: engines hold a reference
    saved = (qa_engine.STOP_CONFIDENCE, qa_engine.MAX_ROUNDS)
    qa_engine.STOP_CONFIDENCE = pipeline.stop
    qa_engine.MAX_ROUNDS = pipeline.max_rounds
    try:
        return _run_patients(pipeline, patients, seed)
    finally:
        qa_engine.STOP_CONFIDENCE, qa_engine.MAX_ROUNDS = saved


def _run_patients(pipeline: Pipeline, patients: List[dict], seed: int) -> List[dict]:
    engine = None
    if pipeline.strategy:
        engine = m.make_qa_engine(pipeline.strategy, posterior=pipeline.posterior)
    out = []
    for i, p in enumerate(patients):
        record = set(ast.literal_eval(p["EVIDENCES"]))
        age, sex = (int(p["AGE"]), p["SEX"]) if pipeline.demographics else (None, None)
        rng = random.Random(seed * 100_003 + i)
        present = sorted(record) if pipeline.oracle else volunteered(p, record, rng)
        absent: List[str] = []
        unknown: List[str] = []
        asked = 0
        dont_know = 0
        stop_reason = "asked_all"  # the engine never stopped early

        for round_num in range(1, pipeline.rounds + 1):
            res = engine.get_questions(present, age, sex, round_num=round_num, n_questions=3,
                                       absent=absent if pipeline.record_no else (), unknown=unknown)
            if res.should_stop:
                stop_reason = res.stop_reason
                break
            for q in res.questions:
                asked += 1
                reply = answer_question(q.evidence_id, record, rng)
                if reply == "dk":
                    dont_know += 1
                    unknown.append(q.evidence_id)
                elif reply == "yes":
                    present.append(q.evidence_id)
                elif pipeline.record_no:
                    absent.append(q.evidence_id)

        dist = m.predict_distribution(present, age, sex, absent if pipeline.record_no else (),
                                      posterior=pipeline.posterior)
        assessed = triage.assess(dist, present)
        zone = legacy_zone(dist, present) if pipeline.triage == "legacy" else assessed.zone
        truth = m._GROUP_MAP.get(p["PATHOLOGY"], p["PATHOLOGY"])
        top = list(dist)
        out.append({
            "truth_tier": triage.tier_of(p["PATHOLOGY"]),
            "zone": zone,
            "top1": top[0] == truth,
            "top3": truth in top[:3],
            "conf": dist[top[0]],
            "probs": list(dist.values()),   # full final distribution, for temperature scaling
            "asked": asked,
            "dont_know": dont_know,
            "stop_reason": stop_reason,
            # for the threshold sweep: zone from symptom flags alone + risk masses
            "flag_zone": triage.assess({}, present).zone,
            "red_mass": assessed.red_mass,
            "yellow_mass": assessed.yellow_mass,
        })
    return out


# ─── calibration (measurement only) ──────────────────────────────────────────

def reliability(confidences: Sequence[float], correct: Sequence[bool], bins: int = 10) -> List[tuple]:
    """Equal-width bins over [0, 1]: (low, high, n, mean confidence, accuracy).
    Mean confidence and accuracy are None for an empty bin."""
    buckets: List[List[tuple]] = [[] for _ in range(bins)]
    for c, ok in zip(confidences, correct):
        buckets[min(int(c * bins), bins - 1)].append((c, bool(ok)))
    rows = []
    for i, b in enumerate(buckets):
        n = len(b)
        mean_conf = sum(c for c, _ in b) / n if n else None
        acc = sum(ok for _, ok in b) / n if n else None
        rows.append((i / bins, (i + 1) / bins, n, mean_conf, acc))
    return rows


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    """Sum over bins of (n_bin / N) * |accuracy - mean confidence|."""
    total = len(confidences)
    if not total:
        return 0.0
    return sum(n / total * abs(acc - mc) for _, _, n, mc, acc in reliability(confidences, correct, bins) if n)


def temperature_scale(probs: Sequence[float], T: float) -> List[float]:
    """p_i ** (1/T), renormalised. T=1 returns the input (up to float error)."""
    w = [p ** (1.0 / T) for p in probs]
    s = sum(w)
    return [x / s for x in w] if s > 0 else list(probs)


def calibration_report(results: List[dict]) -> str:
    confs = [r["conf"] for r in results]
    correct = [r["top1"] for r in results]
    lines = ["  reliability of top-1 confidence (10 equal bins):",
             "    bin           n  mean conf  accuracy"]
    for lo, hi, n, mc, acc in reliability(confs, correct):
        if n:
            lines.append(f"    {lo:.1f}-{hi:.1f} {n:6d}     {mc:6.3f}    {acc:6.3f}")
        else:
            lines.append(f"    {lo:.1f}-{hi:.1f} {0:6d}          —          —")
    lines.append(f"  ECE {expected_calibration_error(confs, correct):.4f}")
    temp_parts = []
    for T in TEMPERATURES:
        scaled = [max(temperature_scale(r["probs"], T)) for r in results]
        temp_parts.append(f"T={T:<4} {expected_calibration_error(scaled, correct):.4f}")
    lines.append("  ECE after temperature scaling: " + "  ".join(temp_parts))
    lines.append("  accuracy by stop reason:")
    for reason in sorted({r["stop_reason"] for r in results}):
        group = [r for r in results if r["stop_reason"] == reason]
        acc = sum(r["top1"] for r in group) / len(group)
        lines.append(f"    {reason:16} n={len(group):5d}  top1 {acc:6.1%}")
    return "\n".join(lines)


def triage_rates(results: List[dict]) -> Dict[str, float]:
    red = [r for r in results if r["truth_tier"] == "red"]
    green = [r for r in results if r["truth_tier"] == "green"]
    return {
        "red_under": sum(r["zone"] != "red" for r in red) / max(len(red), 1),
        "red_to_green": sum(r["zone"] == "green" for r in red) / max(len(red), 1),
        "over": sum(r["zone"] == "red" for r in green) / max(len(green), 1),
        "exact": sum(r["zone"] == r["truth_tier"] for r in results) / max(len(results), 1),
    }


def summarise(name: str, results: List[dict], seconds: float) -> None:
    n = len(results)
    t = triage_rates(results)
    print(f"{name:34} top1 {sum(r['top1'] for r in results) / n:6.1%}  top3 {sum(r['top3'] for r in results) / n:6.1%}"
          f"  conf {sum(r['conf'] for r in results) / n:5.2f}  asked {sum(r['asked'] for r in results) / n:4.1f}"
          f"  dk {sum(r['dont_know'] for r in results) / n:4.2f}"
          f" | red under-triage {t['red_under']:6.1%}  red→green {t['red_to_green']:5.1%}"
          f"  over-triage {t['over']:6.1%}  zone exact {t['exact']:6.1%}  ({seconds:4.0f}s)")


def sweep(results: List[dict]) -> None:
    print("\nThreshold sweep (default pipeline): RED_MASS / YELLOW_MASS")
    for red_t in (0.05, 0.10, 0.15, 0.20, 0.30):
        for yellow_t in (0.20, 0.30, 0.40):
            swept = []
            for r in results:
                zone = r["flag_zone"]
                if r["red_mass"] >= red_t:
                    zone = "red"
                elif r["yellow_mass"] >= yellow_t and zone == "green":
                    zone = "yellow"
                swept.append({**r, "zone": zone})
            t = triage_rates(swept)
            mark = "  <- current" if (red_t, yellow_t) == (triage.RED_MASS_THRESHOLD, triage.YELLOW_MASS_THRESHOLD) else ""
            print(f"  {red_t:.2f} / {yellow_t:.2f}: red under-triage {t['red_under']:6.1%}  red→green {t['red_to_green']:5.1%}"
                  f"  over-triage {t['over']:6.1%}  exact {t['exact']:6.1%}{mark}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("patients")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--noise", type=float, default=0.0, help="P(one false-positive symptom per patient)")
    ap.add_argument("--answer-error", type=float, default=0.0,
                    help="P(each follow-up yes/no answer is flipped)")
    ap.add_argument("--dont-know", type=float, default=0.0,
                    help="P(each follow-up question is answered 'don't know')")
    ap.add_argument("--pipelines", default="",
                    help="comma-separated substrings of pipeline names to run (default: all)")
    ap.add_argument("--calibration", action="store_true",
                    help="calibration report for every NB pipeline (the default pipeline always gets one)")
    args = ap.parse_args()
    for flag, value in (("--noise", args.noise), ("--answer-error", args.answer_error),
                        ("--dont-know", args.dont_know)):
        if not 0.0 <= value <= 1.0:
            ap.error(f"{flag} must be between 0 and 1")
    global NOISE, ANSWER_ERROR, DONT_KNOW
    NOISE = args.noise
    ANSWER_ERROR = args.answer_error
    DONT_KNOW = args.dont_know

    pipelines = select_pipelines(args.pipelines)
    if not pipelines:
        ap.error(f"--pipelines {args.pipelines!r} matches no pipeline")

    patients = load_patients(args.patients, args.n, args.seed)
    tiers = {t: sum(triage.tier_of(p["PATHOLOGY"]) == t for p in patients) for t in ("red", "yellow", "green")}
    print(f"{len(patients)} test patients (true tiers: {tiers}), extraction noise {NOISE:.0%}, "
          f"answer error {ANSWER_ERROR:.0%}, don't know {DONT_KNOW:.0%}\n")

    default_results = None
    for pipeline in pipelines:
        start = time.perf_counter()
        results = run(pipeline, patients, args.seed)
        summarise(pipeline.name, results, time.perf_counter() - start)
        if is_default(pipeline):
            default_results = results
        if is_default(pipeline) or (args.calibration and pipeline.posterior == "nb"):
            print(calibration_report(results))
    if default_results:
        sweep(default_results)


if __name__ == "__main__":
    main()
