"""Sanity checks for the extraction benchmark (eval/cases.jsonl) and its scorer."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "eval"))

import compare_extraction as ce  # noqa: E402
import ml_model  # noqa: E402

CASES = [json.loads(line) for line in (Path(__file__).parent / "eval" / "cases.jsonl").read_text(encoding="utf-8").splitlines()]


def test_cases_use_valid_codes():
    for case in CASES:
        codes = case["present"] + case["absent"] + [a for alts in case["equivalent"].values() for a in alts]
        codes += [c for c in case["also_ok"] if not c.endswith("*")]
        assert all(code in ml_model._feat_idx for code in codes), case["id"]
        assert not set(case["present"]) & set(case["absent"]), case["id"]


def test_gold_backend_scores_perfect_extraction():
    results = {c["id"]: asyncio.run(ce.run_case("gold", c)) for c in CASES}
    scored = ce.score("gold", CASES, results, show_errors=False)
    assert scored["precision"] == scored["recall"] == 1.0
    assert scored["negation_errors"] == 0
    assert scored["q_red_missed"].startswith("0/")


def test_confirmation_removes_extras_and_counts_taps():
    case = next(c for c in CASES if c["absent"])
    wrong = {"id": case["id"], "present": case["present"] + ["E_204_@_V_1", case["absent"][0]], "absent": [],
             "uncertain": [], "age": None, "sex": None, "review": None,
             "usage": {"input_tokens": 0, "output_tokens": 0}, "latency_s": 0}
    fixed = ce.confirmed(wrong, case)
    assert set(fixed["present"]) == set(case["present"])
    assert {"E_204_@_V_1", case["absent"][0]} <= set(fixed["absent"])
    assert fixed["fixes"] == 2


def test_scoring_counts_negation_errors_and_extras():
    case = next(c for c in CASES if c["absent"])
    wrong = {"id": case["id"], "present": case["present"] + case["absent"] + ["E_204_@_V_1"], "absent": [],
             "uncertain": [], "age": None, "sex": None, "review": None,
             "usage": {"input_tokens": 0, "output_tokens": 0}, "latency_s": 0}
    scored = ce.score("x", [case], {case["id"]: wrong}, show_errors=False)
    assert scored["negation_errors"] == len(case["absent"])
    assert scored["denials_caught"] == 0
    assert scored["precision"] < 1.0
