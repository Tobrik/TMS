"""
Offline end-to-end simulation of the diagnostic pipeline on DDXPlus test patients.
No LLM involved: measures question selection + model + triage.

Each simulated patient:
  1. volunteers what a patient typically says: their chief complaint
     (INITIAL_EVIDENCE), up to 2 other symptoms, and where it hurts if in pain;
  2. answers every follow-up question truthfully from their full DDXPlus record;
  3. gets a differential and a triage zone.

Usage (from Backend/):
    python eval/simulate.py <path/to/release_test_patients> [--n 3000] [--seed 7]

Metrics: top-1 / top-3 accuracy, questions asked, and triage against the true
disease's tier — red under-triage (true red, not sent red), severe under-triage
(true red, sent green) and over-triage (true green, sent red).
"""

import argparse
import ast
import csv
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ml_model as m  # noqa: E402
import triage  # noqa: E402

EXTRA_VOLUNTEERED = 2
NOISE = 0.0  # probability of one false-positive symptom (set with --noise)
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
]


def load_patients(path: str, n: int, seed: int) -> List[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["PATHOLOGY"] not in m._GEO_EXCLUDED]
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


def run(pipeline: Pipeline, patients: List[dict], seed: int) -> List[dict]:
    m.LIKELIHOODS[:] = m._build_likelihood_matrix(pipeline.floor)  # in place: engines hold a reference
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
        asked = 0

        for round_num in range(1, pipeline.rounds + 1):
            res = engine.get_questions(present, age, sex, round_num=round_num, n_questions=3,
                                       absent=absent if pipeline.record_no else ())
            if res.should_stop:
                break
            for q in res.questions:
                asked += 1
                if q.evidence_id in record:
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
            "asked": asked,
            # for the threshold sweep: zone from symptom flags alone + risk masses
            "flag_zone": triage.assess({}, present).zone,
            "red_mass": assessed.red_mass,
            "yellow_mass": assessed.yellow_mass,
        })
    return out


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
    args = ap.parse_args()
    global NOISE
    NOISE = args.noise

    patients = load_patients(args.patients, args.n, args.seed)
    tiers = {t: sum(triage.tier_of(p["PATHOLOGY"]) == t for p in patients) for t in ("red", "yellow", "green")}
    print(f"{len(patients)} test patients (true tiers: {tiers}), extraction noise {NOISE:.0%}\n")

    default_results = None
    for pipeline in PIPELINES:
        start = time.perf_counter()
        results = run(pipeline, patients, args.seed)
        summarise(pipeline.name, results, time.perf_counter() - start)
        if (pipeline.posterior, pipeline.strategy, pipeline.floor, pipeline.demographics) == \
                (m.POSTERIOR, "eig", m._LIKELIHOOD_FLOOR, True):
            default_results = results
    if default_results:
        sweep(default_results)


if __name__ == "__main__":
    main()
