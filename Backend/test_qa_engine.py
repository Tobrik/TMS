"""
Tests for the two-round QAEngine.

`engine` is the legacy "jsd" strategy (exploration-first, which avoided the
confirmation bias of the old find_discriminative_evidences); `eig_engine` is the
default expected-information-gain strategy.

Run:  python -m pytest test_qa_engine.py -v
"""

import sys
import os
import pytest

# Ensure the Backend directory is on the path so imports resolve
sys.path.insert(0, os.path.dirname(__file__))

from qa_engine import QAResult, STOP_CONFIDENCE, MAX_ROUNDS


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    """
    Build a QAEngine from the real trained model artefacts.
    Skips the entire module if artefacts are missing.
    """
    try:
        import ml_model
    except Exception as exc:
        pytest.skip(f"Cannot import ml_model (missing artefacts?): {exc}")

    if ml_model._clf is None:
        pytest.skip("RF model not loaded")

    return ml_model.make_qa_engine("jsd", posterior="rf")


@pytest.fixture(scope="module")
def eig_engine():
    import ml_model
    return ml_model.make_qa_engine("eig")


# Headache + nausea input — typical neurological presentation
HEADACHE_EVIDENCES = [
    "E_55_@_V_89",   # headache (pain location = head)
    "E_148",          # nausea
]


# ─── Test 1: Exploration round does NOT ask cardiac questions ────────────────

def test_exploration_excludes_cardiac_questions(engine):
    """
    Round 1 (exploration) on a headache+nausea patient must NOT produce
    cardiac-specific questions that the old biased system used to select.
    E_155 = palpitations, E_220 = pleuritic pain, E_217 = worse lying/better sitting.
    """
    result: QAResult = engine.get_questions(
        evidences=HEADACHE_EVIDENCES,
        age=30,
        sex="M",
        round_num=1,
        n_questions=5,
    )
    asked_ids = {q.evidence_id for q in result.questions}

    cardiac_noise = {"E_155", "E_220", "E_217"}
    overlap = asked_ids & cardiac_noise
    assert overlap == set(), (
        f"Exploration round asked cardiac questions {overlap} for a "
        f"headache+nausea patient — confirmation bias not fixed"
    )
    assert result.round_type == "exploration"


# ─── Test 2: JSD score for E_91 > old shift score for E_220 ─────────────────

def test_jsd_beats_old_shift_metric(engine):
    """
    E_91 (fever/temperature) is a broad, informative symptom.
    E_220 (pain on inspiration) is a narrow, cardiac-biased symptom.
    With a headache+nausea input the JSD score for E_91 must exceed the
    old probability-shift score for E_220 — proving the new metric selects
    more informative questions.
    """
    import numpy as np

    base_vec = engine._build_feature_vector(HEADACHE_EVIDENCES, age=30, sex="M")
    base_proba = engine._get_proba(base_vec)

    # New metric: JSD for E_91
    jsd_scores = engine._score_exploration(["E_91"], base_vec, base_proba)
    jsd_e91 = jsd_scores[0][1] if jsd_scores else 0.0

    # Old metric: abs shift of top-1 probability for E_220
    top1_idx = int(np.argmax(base_proba))
    base_top1_p = float(base_proba[top1_idx])

    vec_220 = base_vec.copy()
    if "E_220" in engine._feat_idx:
        vec_220[engine._feat_idx["E_220"]] = 1.0
    proba_220 = engine._get_proba(vec_220)
    old_shift_e220 = abs(float(proba_220[top1_idx]) - base_top1_p)

    assert jsd_e91 > old_shift_e220, (
        f"JSD(E_91)={jsd_e91:.6f} should be > old_shift(E_220)={old_shift_e220:.6f}"
    )


# ─── Test 3: Round 2 questions come from top-3 relevant evidences ────────────

def test_discrimination_uses_relevant_evidences(engine):
    """
    After round 1 fills in more symptoms, round 2 (discrimination) must
    only return questions from the disease_relevant_evidences of the
    current top-3 diagnoses.
    """
    # Simulate a more complete symptom vector (round 1 answers added)
    extended_evidences = HEADACHE_EVIDENCES + [
        "E_91",   # fever
        "E_97",   # sore throat
        "E_181",  # nasal congestion
        "E_201",  # cough
        "E_175",  # malaise
    ]

    result: QAResult = engine.get_questions(
        evidences=extended_evidences,
        age=30,
        sex="M",
        round_num=2,
        n_questions=5,
    )

    if result.should_stop:
        pytest.skip("Model already confident enough — no round 2 questions")

    assert result.round_type == "discrimination"

    # Collect the set of evidences relevant to the current top-3
    top3_names = {name for name, _ in result.top3}
    relevant = set()
    for d_name in top3_names:
        relevant |= engine.disease_relevant_evidences.get(d_name, set())

    asked_ids = {q.evidence_id for q in result.questions}
    outside = asked_ids - relevant
    assert outside == set(), (
        f"Discrimination round asked {outside} which are not relevant "
        f"to top-3 {top3_names}"
    )


# ─── Test 4: High confidence → stop ─────────────────────────────────────────

def test_high_confidence_stops(engine):
    """
    When RF confidence on top-1 is ≥ 0.70, get_questions must return
    an empty list with should_stop=True.
    """
    import numpy as np

    # Build a maximally-informative evidence list by collecting the top
    # relevant evidences for the most likely disease on a base vector.
    base_vec = engine._build_feature_vector([], age=30, sex="M")
    base_proba = engine._get_proba(base_vec)
    top1_name = engine.label_classes[int(np.argmax(base_proba))]

    # Grab the top relevant evidences for that disease and activate them
    relevant = list(engine.disease_relevant_evidences.get(top1_name, set()))
    # Use many evidences to push confidence high
    heavy_evidences = relevant[:20]

    result: QAResult = engine.get_questions(
        evidences=heavy_evidences,
        age=30,
        sex="M",
        round_num=1,
        n_questions=3,
    )

    if result.confidence < STOP_CONFIDENCE:
        # If even 20 top-relevant evidences don't push above 0.70,
        # the model is inherently uncertain — test the logic directly
        # by mocking a high-confidence scenario.
        pytest.skip(
            f"Model confidence {result.confidence:.2f} < {STOP_CONFIDENCE} "
            f"even with 20 relevant evidences — skipping live-model test"
        )

    assert result.should_stop is True
    assert result.questions == []
    assert result.stop_reason == "high_confidence"


# ─── Test 5: round_num=3 → max_rounds stop ──────────────────────────────────

def test_max_rounds_stops(engine):
    """
    When round_num exceeds MAX_ROUNDS (2), the engine must stop
    with reason 'max_rounds' regardless of confidence.
    """
    result: QAResult = engine.get_questions(
        evidences=HEADACHE_EVIDENCES,
        age=30,
        sex="M",
        round_num=3,
        n_questions=3,
    )
    assert result.should_stop is True
    assert result.stop_reason == "max_rounds"
    assert result.questions == []


# ─── Expected information gain (default strategy) ───────────────────────────

FLU_LIKE = ["E_91", "E_201", "E_181"]  # fever, cough, runny nose


def test_eig_returns_translated_questions(eig_engine):
    result = eig_engine.get_questions(FLU_LIKE, round_num=1, n_questions=3)
    assert result.round_type == "eig"
    assert len(result.questions) == 3
    for q in result.questions:
        assert q.question_en and q.question_ru and q.question_kk
        assert q.evidence_id not in FLU_LIKE


def test_eig_never_reasks_a_denied_question(eig_engine):
    first = eig_engine.get_questions(FLU_LIKE, round_num=1, n_questions=3)
    denied = [q.evidence_id for q in first.questions]
    second = eig_engine.get_questions(FLU_LIKE, round_num=2, n_questions=3, absent=denied)
    assert not {q.evidence_id for q in second.questions} & set(denied)


def test_no_answers_change_the_differential():
    """v1 dropped "no" answers; now a denial is evidence too."""
    import ml_model
    before = ml_model.predict_distribution(FLU_LIKE)
    after = ml_model.predict_distribution(FLU_LIKE, absent=["E_94", "E_144", "E_77"])
    assert before != after


def test_eig_scores_are_nonnegative_and_sorted(eig_engine):
    import ml_model
    posterior = ml_model.class_posterior(FLU_LIKE)
    candidates = eig_engine._base_candidates(set(FLU_LIKE))
    scores = [s for _, s in eig_engine._score_eig(candidates, posterior)]
    assert scores == sorted(scores, reverse=True)
    assert min(scores) >= -1e-9


def test_eig_asks_translated_antecedents_only(eig_engine):
    import ml_model
    candidates = eig_engine._base_candidates(set())
    antecedents = [c for c in candidates if ml_model.EVIDENCES_META[c].get("is_antecedent")]
    assert antecedents, "risk-factor questions should be askable"
    assert all(c in ml_model._QUESTION_TRANSLATIONS for c in antecedents)


def test_eig_stops_when_confident(eig_engine):
    import ml_model
    # GERD's six most frequent yes/no evidences in the DDXPlus validation split
    stats = ml_model.EVIDENCE_STATS["p_evidence_given_disease"]["GERD"]
    typical = [e for e, _ in sorted(stats.items(), key=lambda x: -x[1])
               if ml_model.EVIDENCES_META.get(e, {}).get("data_type") == "B"][:6]
    result = eig_engine.get_questions(typical)
    assert result.should_stop and result.stop_reason == "high_confidence"
    assert result.top3[0][0] == "GERD"
    assert result.confidence >= STOP_CONFIDENCE


def test_eig_max_rounds(eig_engine):
    result = eig_engine.get_questions(FLU_LIKE, round_num=MAX_ROUNDS + 1)
    assert result.should_stop and result.stop_reason == "max_rounds"
