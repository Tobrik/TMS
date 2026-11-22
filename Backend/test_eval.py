"""Sanity checks for the extraction benchmark (eval/cases.jsonl) and its scorer."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent / "eval"))

import compare_extraction as ce  # noqa: E402
import ml_model  # noqa: E402
import simulate as sim  # noqa: E402

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


# ─── eval/simulate.py: answer noise, "don't know", calibration ───────────────

class FakeEngine:
    """Scripted question engine: questions per round, then high_confidence stop.
    Records what each call was given."""

    def __init__(self, questions_by_round):
        self.scripted = questions_by_round
        self.calls = []

    def get_questions(self, evidences, age=None, sex=None, round_num=1, n_questions=3, absent=(), unknown=()):
        self.calls.append({"evidences": list(evidences), "absent": list(absent), "unknown": list(unknown)})
        if round_num > len(self.scripted):
            return SimpleNamespace(should_stop=True, stop_reason="high_confidence", questions=[])
        return SimpleNamespace(
            should_stop=False, stop_reason="",
            questions=[SimpleNamespace(evidence_id=c) for c in self.scripted[round_num - 1]],
        )


def _fake_patient(record, init):
    return {"INITIAL_EVIDENCE": init, "EVIDENCES": repr(sorted(record)),
            "PATHOLOGY": ml_model.LABEL_CLASSES[0], "AGE": "40", "SEX": "M"}


def _run_with_fake(monkeypatch, fake, patient, error=0.0, dont_know=0.0):
    monkeypatch.setattr(sim.m, "make_qa_engine", lambda strategy, **kw: fake)
    monkeypatch.setattr(sim, "ANSWER_ERROR", error)
    monkeypatch.setattr(sim, "DONT_KNOW", dont_know)
    return sim.run(sim.Pipeline("test", "eig", 2, True, "nb", "v2"), [patient], seed=1)


def test_answer_error_one_flips_every_answer(monkeypatch):
    ask = sim.ASKABLE_SYMPTOMS
    patient = _fake_patient({ask[0]}, ask[0])  # record: only the chief complaint
    # questions ask about symptoms the patient does NOT have, so truth is "no"
    fake = FakeEngine([[ask[1], ask[2]]])
    _run_with_fake(monkeypatch, fake, patient, error=0.0)
    assert sorted(fake.calls[-1]["absent"]) == sorted([ask[1], ask[2]])
    assert sorted(fake.calls[-1]["evidences"]) == [ask[0]]

    fake = FakeEngine([[ask[1], ask[2]]])
    results = _run_with_fake(monkeypatch, fake, patient, error=1.0)
    assert sorted(fake.calls[-1]["evidences"]) == sorted([ask[0], ask[1], ask[2]])  # every "no" became "yes"
    assert fake.calls[-1]["absent"] == []
    assert results[0]["asked"] == 2 and results[0]["dont_know"] == 0


def test_dont_know_one_puts_every_asked_code_in_unknown(monkeypatch):
    ask = sim.ASKABLE_SYMPTOMS
    patient = _fake_patient({ask[0]}, ask[0])
    fake = FakeEngine([[ask[1], ask[2]]])
    # answer_error=1.0 too: "don't know" must never be flipped
    results = _run_with_fake(monkeypatch, fake, patient, error=1.0, dont_know=1.0)
    final = fake.calls[-1]
    assert sorted(final["unknown"]) == sorted([ask[1], ask[2]])
    assert final["absent"] == []
    assert sorted(final["evidences"]) == [ask[0]]
    assert results[0]["dont_know"] == 2
    assert results[0]["stop_reason"] == "high_confidence"


def test_expected_calibration_error_hand_example():
    # bin 0.9: 2 patients, mean conf 0.9, accuracy 0.5 -> gap 0.4, weight 1/2
    # bin 0.1: 2 patients, mean conf 0.1, accuracy 0.0 -> gap 0.1, weight 1/2
    ece = sim.expected_calibration_error([0.9, 0.9, 0.1, 0.1], [1, 0, 0, 0])
    assert ece == pytest.approx(0.25)
