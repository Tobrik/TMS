"""Tests for triage.py: risk-mass triage, red flags, escalate-only review."""

import triage


def test_confident_benign_diagnosis_is_not_urgent():
    # v1 sent any top-1 score > 0.6 to the red zone, whatever the disease
    assert triage.assess({"URTI": 0.9, "Viral pharyngitis": 0.1}, ["E_181"]).zone == "green"


def test_dangerous_runner_up_makes_it_red():
    # v1 looked at the top-1 name only; a 15% pulmonary embolism went unnoticed
    result = triage.assess({"URTI": 0.85, "Pulmonary embolism": 0.15}, ["E_181"])
    assert result.zone == "red"
    assert result.red_mass == 0.15
    assert {"zone": "red", "code": "red_mass", "value": 0.15} in result.reasons


def test_red_flag_symptom_overrides_model():
    assert triage.assess({"GERD": 1.0}, ["E_14"]).zone == "red"


def test_red_flag_combination():
    assert triage.assess({"Bronchitis": 1.0}, ["E_91", "E_220"]).zone == "red"


def test_yellow_flag():
    assert triage.assess({"URTI": 1.0}, ["E_91"]).zone == "yellow"


def test_review_escalates_but_never_lowers():
    benign = {"URTI": 1.0}
    assert triage.assess(benign, [], {"emergency": True, "emergency_verified": True}).zone == "red"
    assert triage.assess(benign, [], {"emergency": True, "emergency_verified": False}).zone == "yellow"
    assert triage.assess(benign, [], {"out_of_scope": True}).zone == "yellow"
    assert triage.assess(benign, ["E_14"], {"emergency": False, "out_of_scope": False}).zone == "red"


def test_tiers_include_ddxplus_severity():
    assert triage.tier_of("Larygospasm") == "red"      # severity 1, green in v1
    assert triage.tier_of("Croup") == "yellow"         # severity 2, green in v1
    assert triage.tier_of("Pulmonary embolism") == "red"
    assert triage.tier_of("Stable angina") == "yellow"
    assert triage.tier_of("URTI") == "green"


def test_critical_alert():
    assert "СТРИДОР" in triage.critical_alert(["E_112"])["message"]
    assert triage.critical_alert(["E_14", "E_50"])["level"] == "red"
    assert triage.critical_alert(["E_181"]) is None
