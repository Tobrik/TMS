"""Metric objects, Wilson intervals and a tiny offline build of the accuracy report."""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "eval"))

import build_report as br  # noqa: E402
import compare_extraction as ce  # noqa: E402
import simulate  # noqa: E402
from metrics import metric, wilson  # noqa: E402


# ─── metrics ─────────────────────────────────────────────────────────────────

def test_wilson_known_values():
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(50, 100)
    assert round(lo, 3) == 0.404 and round(hi, 3) == 0.596
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and lo < 1.0
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and hi > 0.0


def test_metric_object_shape_and_rounding():
    assert metric(2706, 3000) == {"value": 0.902, "k": 2706, "n": 3000, "ci": [0.891, 0.912]}
    assert metric(0, 0) == {"value": None, "k": 0, "n": 0, "ci": None}
    m = metric(10, 10)
    assert m["value"] == 1.0 and m["ci"][1] == 1.0


# ─── build_report on a tiny fixture ──────────────────────────────────────────

PATIENTS = [  # DDXPlus-shaped rows: only the columns simulate.load_patients reads
    {"AGE": "30", "DIFFERENTIAL_DIAGNOSIS": "[]", "SEX": "M", "PATHOLOGY": "Influenza",
     "EVIDENCES": "['E_91', 'E_201']", "INITIAL_EVIDENCE": "E_91"},
    {"AGE": "45", "DIFFERENTIAL_DIAGNOSIS": "[]", "SEX": "F", "PATHOLOGY": "URTI",
     "EVIDENCES": "['E_97', 'E_201']", "INITIAL_EVIDENCE": "E_97"},
    {"AGE": "60", "DIFFERENTIAL_DIAGNOSIS": "[]", "SEX": "M", "PATHOLOGY": "Pneumonia",
     "EVIDENCES": "['E_91', 'E_201']", "INITIAL_EVIDENCE": "E_201"},
]

CASES = [  # hand-made: one clean extraction, one with an extra symptom, one out of scope
    {"id": "t1", "lang": "ru", "text": "тест", "present": ["E_91", "E_201"], "absent": [], "equivalent": {},
     "also_ok": [], "pathology": "URTI", "tier": "green", "out_of_scope": False, "likely_condition": None,
     "age": 30, "sex": "M", "record": ["E_91", "E_201", "E_97"]},
    {"id": "t2", "lang": "en", "text": "test", "present": ["E_91", "E_97"], "absent": [], "equivalent": {},
     "also_ok": [], "pathology": "Influenza", "tier": "green", "out_of_scope": False, "likely_condition": None,
     "age": 45, "sex": "F", "record": ["E_91", "E_97"]},
    {"id": "t3", "lang": "kk", "text": "тест", "present": [], "absent": [], "equivalent": {}, "also_ok": [],
     "pathology": None, "tier": None, "out_of_scope": True, "likely_condition": "Перелом", "age": None,
     "sex": None, "record": []},
]


def _review(out_of_scope=False):
    return {"emergency": False, "emergency_verified": False, "out_of_scope": out_of_scope}


CACHE = {  # cached extraction results, keyed by case id
    "t1": {"id": "t1", "present": ["E_91", "E_201"], "absent": [], "uncertain": [], "age": 30, "sex": "M",
           "review": _review(), "usage": {"input_tokens": 1, "output_tokens": 0}, "mode": "clef", "latency_s": 0.1},
    "t2": {"id": "t2", "present": ["E_91", "E_53"], "absent": [], "uncertain": [], "age": 45, "sex": "F",
           "review": _review(), "usage": {"input_tokens": 1, "output_tokens": 0}, "mode": "clef", "latency_s": 0.1},
    "t3": {"id": "t3", "present": [], "absent": [], "uncertain": [], "age": None, "sex": None,
           "review": _review(out_of_scope=True), "usage": {"input_tokens": 1, "output_tokens": 0}, "mode": "clef",
           "latency_s": 0.1},
}

TOP_KEYS = ["generated_at", "model_version", "extraction_backend", "synthetic", "written_cases", "caveats", "i18n"]
SYNTH_KEYS = ["source", "settings", "metrics", "questions_mean", "calibration", "low_confidence", "per_condition"]
WRITTEN_KEYS = ["source", "cases_file", "n_in_scope", "n_out_of_scope", "extraction", "patient_accepts",
                "patient_confirms", "by_language", "out_of_scope", "held_out"]


@pytest.fixture
def fixture_files(tmp_path, monkeypatch):
    patients = tmp_path / "patients.csv"
    with patients.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PATIENTS[0]))
        writer.writeheader()
        writer.writerows(PATIENTS)
    cases = tmp_path / "cases.jsonl"
    cases.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in CASES) + "\n", encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()
    (results / "clef-flash-test.jsonl").write_text(
        "\n".join(json.dumps(CACHE[c["id"]], ensure_ascii=False) for c in CASES) + "\n", encoding="utf-8")
    monkeypatch.setattr(ce, "RESULTS", results)
    return patients, cases


def test_build_report_shape_and_values(fixture_files):
    patients, cases = fixture_files
    report = br.build_report(str(patients), cases, "clef-flash", "test", n=3, seed=7)

    assert list(report) == TOP_KEYS
    assert report["model_version"] == "v2.1-nb" and report["extraction_backend"] == "clef-flash"
    assert 4 <= len(report["caveats"]) <= 6 and all("{backend}" not in c for c in report["caveats"])
    json.dumps(report, ensure_ascii=False)  # serialisable

    syn = report["synthetic"]
    assert list(syn) == SYNTH_KEYS
    assert syn["settings"] == {"n": 3, "seed": 7, "noise": 0.3, "answer_error": 0.1, "dont_know": 0.2}
    assert syn["metrics"]["top1"]["n"] == 3 and syn["metrics"]["red_sensitivity"]["n"] == 0
    assert sum(b["n"] for b in syn["calibration"]) == 3 and len(syn["calibration"]) == 10
    assert syn["per_condition"] == []  # no condition reaches n >= 20
    # the synthetic run restores the simulation globals it changed
    assert (simulate.NOISE, simulate.ANSWER_ERROR, simulate.DONT_KNOW) == (0.0, 0.0, 0.0)

    written = report["written_cases"]
    assert list(written) == WRITTEN_KEYS
    assert written["n_in_scope"] == 2 and written["n_out_of_scope"] == 1
    # gold present: t1 2 (both found), t2 2 (E_91 found, E_97 missed, E_53 extra)
    ext = written["extraction"]
    assert ext["precision"]["k"] == 3 and ext["precision"]["n"] == 4 and ext["precision"]["value"] == 0.75
    assert ext["recall"]["k"] == 3 and ext["recall"]["n"] == 4
    assert ext["denials"] == {"value": None, "k": 0, "n": 0, "ci": None}
    for key in ("top1", "top3", "red_sensitivity", "green_to_red"):
        assert set(written["patient_confirms"][key]) == {"value", "k", "n", "ci"}
    assert written["patient_confirms"]["top1"]["n"] == 2
    assert written["patient_accepts"]["top3"]["n"] == 2
    assert written["by_language"]["ru"]["n"] == 1 and written["by_language"]["en"]["n"] == 1
    assert written["by_language"]["kk"] == {"n": 0, "top1": {"value": None, "k": 0, "n": 0, "ci": None}}
    assert written["out_of_scope"]["flagged"] == {"value": 1.0, "k": 1, "n": 1, "ci": [0.207, 1.0]}
    assert written["out_of_scope"]["false_flags"]["k"] == 0 and written["out_of_scope"]["false_flags"]["n"] == 2


def test_missing_cache_gives_null_written_cases(fixture_files, monkeypatch, capsys):
    patients, cases = fixture_files
    monkeypatch.setattr(ce, "RESULTS", cases.parent / "empty")
    report = br.build_report(str(patients), cases, "clef-flash", "test", n=3, seed=7)
    assert report["written_cases"] is None and report["synthetic"]["metrics"]["top1"]["n"] == 3
    assert "written_cases is null" in capsys.readouterr().err


def test_main_writes_the_json_file(fixture_files, monkeypatch, tmp_path, capsys):
    patients, cases = fixture_files
    out = tmp_path / "out" / "report.json"
    monkeypatch.setattr(sys, "argv", ["build_report.py", "--patients", str(patients), "--cases", str(cases),
                                      "--backend", "clef-flash", "--tag", "test", "--out", str(out),
                                      "--n", "3", "--seed", "7"])
    br.main()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert list(saved) == TOP_KEYS
    assert "synthetic:" in capsys.readouterr().out


def test_simulate_run_records_the_canonical_truth(fixture_files):
    patients, _ = fixture_files
    rows = simulate.load_patients(str(patients), 3, 7)
    results = simulate.run(br.default_pipeline(), rows, 7)
    expected = [simulate.m._GROUP_MAP.get(p["PATHOLOGY"], p["PATHOLOGY"]) for p in rows]
    assert [r["truth"] for r in results] == expected
    assert all(set(r) >= {"truth", "truth_tier", "zone", "top1"} for r in results)


def test_backend_constants_come_from_the_module_source():
    constants = br.read_backend_constants()
    assert constants["MODEL_VERSION"] == "v2.1-nb"
    assert isinstance(constants["disease_labels"], dict) and len(constants["disease_labels"]) > 40


def test_localized_text_has_every_language_and_no_placeholders():
    text = br.localized_text("clef-flash", 3000, 120)
    assert set(text) == {"ru", "kk", "en"}
    for lang, t in text.items():
        assert len(t["caveats"]) == len(br.CAVEATS)
        for line in [t["synthetic_source"], t["written_source"], *t["caveats"]]:
            assert "{" not in line and "patient_confirms" not in line
