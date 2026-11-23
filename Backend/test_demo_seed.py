"""Demo patients: every day must come out of the model as the case says.

The prediction mirrors the /analys endpoint (analys_endpoint in `backend TMS.py`):
evidences and absent are cleaned with validate_evidences (dedupe, drop unknown
codes), absent loses anything also present, then predict_distribution gives the
differential, its first key is diseaseName, and triage.assess gives the zone with
no LLM review (seeded days have no intake review).
"""

import re

import pytest

import triage
from demo_seed import DEMO_CASES
from ml_model import predict_distribution, validate_evidences

KAZAKH_LETTERS = set("әғқңөұүһі")


def analys_like(evidences, absent, age, sex):
    """The prediction /analys stores for one day, without the database."""
    evidences = validate_evidences(list(dict.fromkeys(evidences)))[0]
    absent = [e for e in validate_evidences(list(dict.fromkeys(absent)))[0] if e not in set(evidences)]
    notes: list = []
    distribution = predict_distribution(evidences, age, sex, absent, notes=notes)
    top1 = next(iter(distribution))
    assessed = triage.assess(distribution, evidences, None)
    return top1, distribution[top1], assessed.zone


def last_day(case):
    return case["days"][-1]


@pytest.mark.parametrize("case", DEMO_CASES, ids=lambda c: c["key"])
def test_every_day_matches_expected_prediction(case):
    for day in case["days"]:
        top1, score, zone = analys_like(day["evidences"], day["absent"], case["age"], case["sex"])
        assert top1 == day["expect_top1"], f"{case['key']} {day['minutes_ago']} min: top-1 {top1} ({score:.3f})"
        assert zone == day["expect_zone"], f"{case['key']} {day['minutes_ago']} min: zone {zone}"


@pytest.mark.parametrize("case", DEMO_CASES, ids=lambda c: c["key"])
def test_evidence_codes_are_valid_and_not_contradictory(case):
    for day in case["days"]:
        for field in ("evidences", "absent", "unknown"):
            valid, invalid = validate_evidences(day[field])
            assert not invalid, f"{case['key']} {field}: unknown codes {invalid}"
        assert not set(day["evidences"]) & set(day["absent"]), f"{case['key']}: code in both lists"
        assert day["evidences"], f"{case['key']}: no evidences"


def test_six_cases_with_unique_keys():
    assert len(DEMO_CASES) == 6
    keys = [c["key"] for c in DEMO_CASES]
    assert len(set(keys)) == 6
    for key in keys:
        assert re.fullmatch(r"demo-[a-z-]+", key), key


def test_zone_spread_is_two_two_two():
    zones = [last_day(c)["expect_zone"] for c in DEMO_CASES]
    assert sorted(zones) == ["green", "green", "red", "red", "yellow", "yellow"]


def test_language_spread_on_last_day():
    langs = [last_day(c)["lang"] for c in DEMO_CASES]
    assert langs.count("ru") == 4
    assert langs.count("kk") == 1
    assert langs.count("en") == 1


def test_kazakh_complaint_uses_kazakh_letters():
    for case in DEMO_CASES:
        for day in case["days"]:
            if day["lang"] == "kk":
                assert KAZAKH_LETTERS & set(day["complaint"].lower()), case["key"]


def test_ages_and_sexes_vary():
    ages = [c["age"] for c in DEMO_CASES]
    assert any(a > 60 for a in ages)
    assert any(18 <= a <= 30 for a in ages)
    assert {c["sex"] for c in DEMO_CASES} == {"M", "F"}
    assert all(c["sex"] in ("M", "F") for c in DEMO_CASES)


def test_names_and_cities_are_filled():
    for case in DEMO_CASES:
        assert len(case["name"].split()) == 2, case["key"]
        assert case["city"], case["key"]
        assert all(day["complaint"].strip() for day in case["days"]), case["key"]


def test_timing_latest_recent_and_distinct_older_days_within_range():
    latest = [last_day(c)["minutes_ago"] for c in DEMO_CASES]
    assert all(5 <= m <= 180 for m in latest)
    assert len(set(latest)) == len(latest)
    two_day = [c for c in DEMO_CASES if len(c["days"]) > 1]
    assert 1 <= len(two_day) <= 2
    for case in DEMO_CASES:
        mins = [d["minutes_ago"] for d in case["days"]]
        assert mins == sorted(mins, reverse=True), f"{case['key']}: days must be oldest first"
        for m in mins[:-1]:
            assert 3000 <= m <= 5000, case["key"]
        assert all(d["lang"] in ("ru", "kk", "en") for d in case["days"])
