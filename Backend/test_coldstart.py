"""
Cold start: the app modules must run without scikit-learn, joblib and scipy.

Production installs requirements.txt only (naive Bayes, numpy). The checks run in
subprocesses so that a blocked import cannot leak into other tests.

Run:  python -m pytest -q test_coldstart.py
"""

import json
import os
import subprocess
import sys

BACKEND = os.path.dirname(os.path.abspath(__file__))
HEAVY = ("sklearn", "joblib", "scipy")

BLOCK_BOOTSTRAP = """
import sys
for _name in %r:
    sys.modules[_name] = None   # any `import <name>` now raises ImportError
import ml_model, qa_engine, extraction, triage  # noqa: E401

# imports still work
d = ml_model.predict_distribution(["E_91", "E_201"])
assert d, "empty differential"
top, p = next(iter(d.items()))
assert isinstance(top, str) and 0.0 < p <= 1.0, (top, p)

q = ml_model.qa_engine.get_questions(["E_91", "E_201"], round_num=1)
assert q.stop_reason != "model_unavailable", q.stop_reason
assert q.round_type == "eig", q.round_type
assert q.questions and q.questions[0].evidence_id, "eig asked no question"

# the RF-only paths degrade instead of crashing
assert ml_model.get_rf() is None
jsd = ml_model.make_qa_engine("jsd", posterior="nb").get_questions(["E_91"], round_num=1)
assert jsd.stop_reason == "model_unavailable", jsd.stop_reason

assert triage.tier_of(top) in ("red", "yellow", "green")
print("OK", top, round(p, 3), q.questions[0].evidence_id)
""" % (HEAVY,)


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=BACKEND, capture_output=True, text=True, timeout=180,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def test_app_modules_import_and_predict_without_sklearn_joblib_scipy():
    proc = _run(BLOCK_BOOTSTRAP)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().startswith("OK"), proc.stdout


def test_importing_ml_model_does_not_import_heavy_packages():
    code = (
        "import json, sys\n"
        "import ml_model, qa_engine, extraction, triage  # noqa: E401\n"
        "print(json.dumps(sorted(m for m in sys.modules "
        f"if m.split('.')[0] in {HEAVY!r})))\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr[-2000:]
    loaded = json.loads(proc.stdout.strip().splitlines()[-1])
    assert loaded == [], f"heavy packages imported at module load: {loaded}"
