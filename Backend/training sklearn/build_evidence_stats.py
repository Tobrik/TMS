"""
Builds evidence_stats.json, counted from DDXPlus patients:
  - priors: P(disease)
  - p_evidence_given_disease: P(evidence token | disease)
  - p_age_bin_given_disease, p_sex_given_disease: demographics per disease

ml_model uses these as a naive Bayes model (the default posterior) and the
question engine uses them to compute expected information gain.

Usage:
    python build_evidence_stats.py <path/to/release_validate_patients>

Use the VALIDATE split: the RF is trained on the train split and the offline
simulation (eval/simulate.py) is run on the test split, so this keeps all three
disjoint.
"""

import ast
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).parent
OUT = BASE / "evidence_stats.json"
MIN_FREQ = 0.002  # drop rarer entries to keep the file small; readers floor them

AGE_BIN_EDGES = [5, 15, 30, 45, 60, 75]  # bins: <5, 5-14, 15-29, 30-44, 45-59, 60-74, 75+


def age_bin(age: int) -> int:
    return sum(age >= edge for edge in AGE_BIN_EDGES)


def main(csv_path: str) -> None:
    counts: dict = defaultdict(Counter)
    n_by_disease: Counter = Counter()
    ages: dict = defaultdict(Counter)
    sexes: dict = defaultdict(Counter)

    with open(csv_path, encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            disease = row["PATHOLOGY"]
            evidences = set(ast.literal_eval(row["EVIDENCES"]))
            n_by_disease[disease] += 1
            counts[disease].update(evidences)
            ages[disease][age_bin(int(row["AGE"]))] += 1
            sexes[disease][row["SEX"]] += 1
            if (i + 1) % 50_000 == 0:
                print(f"  {i + 1:,} rows")

    total = sum(n_by_disease.values())
    stats = {
        "source": Path(csv_path).name,
        "n_patients": total,
        "priors": {d: n / total for d, n in sorted(n_by_disease.items())},
        "age_bin_edges": AGE_BIN_EDGES,
        # Laplace-smoothed: a bin never seen for a disease is rare, not impossible
        "p_age_bin_given_disease": {
            d: [round((ages[d][b] + 1) / (n_by_disease[d] + len(AGE_BIN_EDGES) + 1), 4)
                for b in range(len(AGE_BIN_EDGES) + 1)]
            for d in sorted(n_by_disease)
        },
        "p_sex_given_disease": {
            d: {s: round((sexes[d][s] + 1) / (n_by_disease[d] + 2), 4) for s in ("M", "F")}
            for d in sorted(n_by_disease)
        },
        "p_evidence_given_disease": {
            d: {
                ev: round(c / n_by_disease[d], 4)
                for ev, c in sorted(counts[d].items())
                if c / n_by_disease[d] >= MIN_FREQ
            }
            for d in sorted(n_by_disease)
        },
    }
    OUT.write_text(json.dumps(stats, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{total:,} patients, {len(n_by_disease)} diseases -> {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
