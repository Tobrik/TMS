"""
End-to-end API test on a throwaway PostgreSQL (pip install pgserver) with a fake
Groq: register → extract → questions → analyse → explain → history → doctor.
"""

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

pgserver = pytest.importorskip("pgserver")

COMPLAINT = "Третий день температура 39, сильный кашель и болит горло. Насморка нет."
EXTRACTION = {
    "findings": [
        {"code": "E_91", "status": "present", "quote": "температура 39"},
        {"code": "E_201", "status": "present", "quote": "сильный кашель"},
        {"code": "E_97", "status": "present", "quote": "болит горло"},
        {"code": "E_181", "status": "absent", "quote": "Насморка нет"},
    ],
    "pain": {"locations": ["V_148"], "radiates_to": [], "character": [], "intensity": None,
             "onset_speed": None, "quote": "болит горло"},
    "travel": None, "age": None, "sex": None,
}
REVIEW = {"emergency": False, "emergency_quote": "", "reason": "", "in_scope": True, "likely_condition": None}


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    server = pgserver.get_server(tmp_path_factory.mktemp("pg"), cleanup_mode="stop")
    from cryptography.fernet import Fernet
    os.environ.update({
        "DATABASE_URL": server.get_uri(),
        "DB_ENCRYPTION_KEY": Fernet.generate_key().decode(),
        "JWT_SECRET_KEY": "test-secret",
        "DOCTOR_SEEDS": "Dr Test|Терапевт|Doctor123",
    })
    module = importlib.import_module("backend TMS")  # creates tables, seeds the doctor
    module.limiter.enabled = False  # the tests register more patients than 10/minute
    from fastapi.testclient import TestClient
    with TestClient(module.app) as client:
        yield client, module
    server.cleanup()


def register(client, name="Айгерим"):
    resp = client.post("/regist_as_patient", json={"name": name, "city": "Алматы", "password": "secret123"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    return body["patient_id"], {"Authorization": f"Bearer {body['access_token']}"}


def doctor_headers(client):
    resp = client.post("/login_doctor", json={"doctor_id": 1, "password": "Doctor123"})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def test_full_patient_flow(api, fake_groq):
    client, module = api
    fake_groq.replies.update({
        "extract": EXTRACTION,
        "review": REVIEW,
        "text": lambda p: ("**Clinical reasoning** fever, cough, sore throat"
                           if "DOCTOR" in p["messages"][0]["content"] else "Похоже на инфекцию горла."),
    })
    patient_id, headers = register(client)

    # 1. Free text → findings
    resp = client.post("/extract_symptoms", json={"text": COMPLAINT}, headers=headers)
    assert resp.status_code == 200, resp.text
    extracted = resp.json()
    assert {"E_91", "E_201", "E_97", "E_53", "E_55_@_V_148"} <= set(extracted["evidences"])
    assert extracted["absent"] == ["E_181"]
    assert extracted["mode"] == "llm" and extracted["intake_id"] > 0
    assert all(f["label"]["ru"] for f in extracted["findings"])

    # 2. Follow-up questions: English text present, denied symptom never asked
    resp = client.post("/next_questions", headers=headers, json={
        "evidences": extracted["evidences"], "absent": extracted["absent"], "round_num": 1, "n": 3})
    assert resp.status_code == 200, resp.text
    questions = resp.json()["questions"]
    assert questions and all(q["question_en"] for q in questions)
    assert "E_181" not in {q["evidence_id"] for q in questions}

    # 3. Analysis, linked to the intake
    absent = extracted["absent"] + [q["evidence_id"] for q in questions]  # patient said "no" to all
    resp = client.post("/analys", headers=headers, json={
        "evidences": extracted["evidences"], "absent": absent, "intake_id": extracted["intake_id"]})
    assert resp.status_code == 200, resp.text
    result = resp.json()
    assert result["diseaseName"] in module.LABEL_CLASSES
    assert result["zone"] in ("green", "yellow", "red")
    assert any(r["code"] == "yellow_flag" for r in result["reasons"])  # fever
    assert result["out_of_scope"] is None

    # 4. Explanations generated and stored server-side
    resp = client.post("/explain", headers=headers, json={"day_id": result["day"], "lang": "ru"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["patientExplanation"] == "Похоже на инфекцию горла."
    doctor_prompt_facts = fake_groq.calls[-1][2]["messages"][1]["content"]
    assert "Influence:" in doctor_prompt_facts and COMPLAINT in doctor_prompt_facts

    # 5. History shows the top-1 disease, its zone and the stored explanation
    entry = client.post("/get_history", headers=headers, json={"patient_id": patient_id}).json()["history"][0]
    assert entry["disease_predict"] == result["diseaseName"]
    assert entry["zone"] == result["zone"]
    assert entry["patient_explanation"] == "Похоже на инфекцию горла."

    # 6. Denied symptoms are stored (value 0) alongside reported ones
    symptoms = client.post("/get_day_symptoms", headers=headers, json={"day_id": result["day"]}).json()["symptoms"]
    assert symptoms["E_91"] == 1 and symptoms["E_181"] == 0

    # 7. The doctor sees the same zone; the doctor's name is decrypted in history
    doc = doctor_headers(client)
    triaged = {p["patient_id"]: p for p in client.post("/list_patients_triage", headers=doc).json()["patients"]}
    assert triaged[patient_id]["zone"] == result["zone"]
    assert triaged[patient_id]["last_disease"] == result["diseaseName"]
    client.post("/update_by_doctor", headers=doc, json={
        "patient_id": patient_id, "day_id": result["day"], "doctor_id": 1, "recept": "Покой"})
    entry = client.post("/get_history", headers=headers, json={"patient_id": patient_id}).json()["history"][0]
    assert entry["doctor_name"] == "Dr Test"


def test_patient_cannot_use_another_patients_intake(api, fake_groq):
    client, _ = api
    fake_groq.replies.update({"extract": EXTRACTION, "review": REVIEW})
    _, owner = register(client, "Owner")
    _, other = register(client, "Other")
    intake_id = client.post("/extract_symptoms", json={"text": COMPLAINT}, headers=owner).json()["intake_id"]
    resp = client.post("/analys", headers=other, json={"evidences": ["E_91"], "intake_id": intake_id})
    assert resp.status_code == 403


def test_out_of_scope_review_raises_zone_and_warns(api, fake_groq):
    client, _ = api
    fake_groq.replies.update({
        "extract": {**EXTRACTION, "findings": [{"code": "E_181", "status": "present", "quote": "Насморка"}],
                    "pain": None},
        "review": {**REVIEW, "in_scope": False, "likely_condition": "Appendicitis"},
    })
    _, headers = register(client, "Scope")
    extracted = client.post("/extract_symptoms", json={"text": COMPLAINT}, headers=headers).json()
    result = client.post("/analys", headers=headers, json={
        "evidences": extracted["evidences"], "intake_id": extracted["intake_id"]}).json()
    assert result["out_of_scope"] == {"likely_condition": "Appendicitis"}
    assert result["zone"] in ("yellow", "red")
    assert "Appendicitis" in result["recommendation"]


def test_extraction_without_llm_uses_regex_fallback(api, fake_groq):
    client, _ = api
    fake_groq.replies.update({"extract": 500, "review": 500})
    _, headers = register(client, "Fallback")
    extracted = client.post("/extract_symptoms", json={"text": COMPLAINT}, headers=headers).json()
    assert extracted["mode"] == "fallback"
    assert "E_91" in extracted["evidences"]


def test_v1_rows_are_triaged_by_full_disease_name(api):
    """v1 stored the space-joined top-3; the dashboard used split(" ")[0] and
    looked up "Possible" instead of "Possible NSTEMI / STEMI"."""
    client, module = api
    patient_id, _ = register(client, "Legacy")
    module.insert_disease(patient_id, ["E_201"], "Possible NSTEMI / STEMI URTI Bronchitis", 0.3)
    triaged = {p["patient_id"]: p for p in
               client.post("/list_patients_triage", headers=doctor_headers(client)).json()["patients"]}
    assert triaged[patient_id]["last_disease"] == "Possible NSTEMI / STEMI"
    assert triaged[patient_id]["zone"] == "red"


def test_save_explanation_is_gone(api):
    client, _ = api
    _, headers = register(client, "Spoof")
    resp = client.post("/save_explanation", headers=headers,
                       json={"day_id": 1, "patient_explanation": "x", "doctor_explanation": "fake AI analysis"})
    assert resp.status_code == 404


def test_clef_backend_end_to_end(api, fake_clef, monkeypatch):
    """Clef mode: confidence-banded findings; its emergency score alone never makes a case red."""
    from conftest import choice, status
    client, _ = api
    monkeypatch.setenv("EXTRACTION_BACKEND", "clef")
    fake_clef["answers"].update({
        "E_91": status(reported=0.95),
        "E_201": status(reported=0.9),
        "E_97": status(reported=0.6),            # "?" chip
        "E_181": status(denied=0.9),
        "pain_location": choice({"none": 0.1, "V_148": 0.9}),
        "emergency": {"type": "noul", "noul": 0.97},
    })
    _, headers = register(client, "Clef")
    extracted = client.post("/extract_symptoms", json={"text": COMPLAINT}, headers=headers).json()
    assert extracted["mode"] == "clef"
    assert {"E_91", "E_201", "E_53", "E_55_@_V_148"} <= set(extracted["evidences"])
    assert extracted["uncertain"] == ["E_97"] and extracted["absent"] == ["E_181"]
    assert next(f for f in extracted["findings"] if f["code"] == "E_97")["confidence"] == 0.6
    assert extracted["critical_alert"]["level"] == "yellow"  # soft warning, not red

    result = client.post("/analys", headers=headers, json={
        "evidences": extracted["evidences"], "absent": extracted["absent"], "intake_id": extracted["intake_id"]}).json()
    llm_reasons = [r for r in result["reasons"] if r["code"] == "llm_emergency"]
    assert llm_reasons and all(r["zone"] == "yellow" for r in llm_reasons)


def _pipeline_records(caplog):
    return [json.loads(r.getMessage().split("PIPELINE ", 1)[1]) for r in caplog.records if r.name == "tms.pipeline"]


def _run_case(client, headers):
    extracted = client.post("/extract_symptoms", json={"text": COMPLAINT}, headers=headers).json()
    questions = client.post("/next_questions", headers=headers, json={
        "intake_id": extracted["intake_id"], "evidences": extracted["evidences"],
        "absent": extracted["absent"], "round_num": 1}).json()["questions"]
    result = client.post("/analys", headers=headers, json={
        "evidences": extracted["evidences"], "absent": extracted["absent"] + [q["evidence_id"] for q in questions],
        "intake_id": extracted["intake_id"]}).json()
    client.post("/explain", headers=headers, json={"day_id": result["day"], "lang": "en"})
    return extracted, result


def test_pipeline_logs_follow_a_case_without_patient_text(api, fake_groq, caplog, monkeypatch):
    import logging
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "eval"))
    import trace as pipeline_trace

    client, _ = api
    monkeypatch.delenv("LOG_PATIENT_TEXT", raising=False)
    fake_groq.replies.update({"extract": EXTRACTION, "review": REVIEW, "text": "Explanation."})
    caplog.set_level(logging.INFO, logger="tms.pipeline")
    _, headers = register(client, "Logs")
    extracted, result = _run_case(client, headers)

    records = _pipeline_records(caplog)
    assert [r["stage"] for r in records] == ["extract", "questions", "analysis", "explain"]
    assert {r.get("intake") for r in records[:3]} == {extracted["intake_id"]}
    assert records[3]["day"] == records[2]["day"] == result["day"]

    extract, questions, analysis, explain = records
    assert extract["lang"] == "ru" and extract["backend"] == "llm"
    assert {"code": "E_181", "label": extract["findings"][3]["label"], "status": "absent", "source": "llm"} in extract["findings"]
    assert extract["review"]["available"] and "reason" not in extract["review"]
    assert questions["round"] == 1 and questions["asked"] and questions["top"]
    assert analysis["top"][0][0] == result["diseaseName"] and analysis["zone"] == result["zone"]
    assert set(analysis["fit"]) == {s["name"] for s in result["slices"]}
    assert explain["patient_model"] != "fallback" and explain["attributions"]

    # Privacy: no complaint text, quotes or LLM reason anywhere in the pipeline logs
    logged = "\n".join(r.getMessage() for r in caplog.records if r.name == "tms.pipeline")
    for fragment in ("температура 39", "сильный кашель", "Насморка нет"):
        assert fragment not in logged

    timeline = pipeline_trace.render(pipeline_trace.parse(f"INFO PIPELINE {json.dumps(r)}" for r in records))
    assert f"intake {extracted['intake_id']}" in timeline and "ANALYSIS" in timeline and "round 1" in timeline


def test_pipeline_logs_include_text_when_enabled(api, fake_groq, caplog, monkeypatch):
    import logging
    client, _ = api
    monkeypatch.setenv("LOG_PATIENT_TEXT", "true")
    fake_groq.replies.update({"extract": EXTRACTION, "review": REVIEW})
    caplog.set_level(logging.INFO, logger="tms.pipeline")
    _, headers = register(client, "LogsText")
    client.post("/extract_symptoms", json={"text": COMPLAINT}, headers=headers)
    extract = _pipeline_records(caplog)[0]
    assert extract["text"] == COMPLAINT
    assert any(f.get("quote") == "температура 39" for f in extract["findings"])


def test_failed_review_is_visible_in_the_logs(api, fake_groq, caplog):
    import logging
    client, _ = api
    fake_groq.replies.update({"extract": EXTRACTION, "review": 429})  # rate-limited review
    caplog.set_level(logging.INFO, logger="tms.pipeline")
    _, headers = register(client, "LogsNoReview")
    client.post("/extract_symptoms", json={"text": COMPLAINT}, headers=headers)
    assert _pipeline_records(caplog)[0]["review"] == {"available": False}


# ─── "Don't know", symptom picker, doctor-confirmed diagnoses ───

def _diagnosed_day(client, headers):
    """Extract the complaint and analyse it; returns (intake extraction, analysis result)."""
    extracted = client.post("/extract_symptoms", json={"text": COMPLAINT}, headers=headers).json()
    result = client.post("/analys", headers=headers, json={
        "evidences": extracted["evidences"], "absent": extracted["absent"],
        "intake_id": extracted["intake_id"]}).json()
    return extracted, result


def test_next_questions_never_asks_again_a_dont_know(api, fake_groq):
    client, _ = api
    fake_groq.replies.update({"extract": EXTRACTION, "review": REVIEW})
    _, headers = register(client, "DontKnow")
    extracted = client.post("/extract_symptoms", json={"text": COMPLAINT}, headers=headers).json()
    base = {"evidences": extracted["evidences"], "absent": extracted["absent"], "round_num": 1, "n": 3}
    first = client.post("/next_questions", headers=headers, json=base).json()["questions"]
    assert first
    unknown = [q["evidence_id"] for q in first]

    resp = client.post("/next_questions", headers=headers, json={**base, "unknown": unknown})
    assert resp.status_code == 200, resp.text
    again = {q["evidence_id"] for q in resp.json()["questions"]}
    assert again and not (again & set(unknown))


def test_analys_stores_unknown_and_model_version(api, fake_groq):
    client, module = api
    fake_groq.replies.update({"extract": EXTRACTION, "review": REVIEW})
    _, headers = register(client, "Unknown")
    extracted = client.post("/extract_symptoms", json={"text": COMPLAINT}, headers=headers).json()
    dont_know = client.post("/next_questions", headers=headers, json={
        "evidences": extracted["evidences"], "absent": extracted["absent"], "round_num": 1}).json()["questions"][0]["evidence_id"]
    resp = client.post("/analys", headers=headers, json={
        "evidences": extracted["evidences"], "absent": extracted["absent"],
        "unknown": [dont_know], "intake_id": extracted["intake_id"]})
    assert resp.status_code == 200, resp.text
    day = resp.json()["day"]

    conn = module.connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT analysis_json FROM diary_days WHERE day_id = %s", (day,))
        analysis = json.loads(module.decrypt_field(cur.fetchone()[0]))
        cur.execute("SELECT symptom_code FROM diary_symptoms WHERE day_id = %s", (day,))
        stored_symptoms = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()
    assert analysis["unknown"] == [dont_know]
    assert analysis["model_version"] == module.MODEL_VERSION == "v2.1-nb"
    assert dont_know not in stored_symptoms  # "don't know" is not evidence


def test_conditions_list_matches_the_model(api):
    client, module = api
    _, headers = register(client, "Conditions")
    assert client.get("/conditions").status_code in (401, 403)  # no token
    resp = client.get("/conditions", headers=headers)
    assert resp.status_code == 200, resp.text
    conditions = resp.json()["conditions"]

    names = [c["name"] for c in conditions]
    assert set(names) == set(module.predict_distribution(["E_91"]).keys())
    assert len(names) == len(set(names))
    assert "Unstable angina" not in names and "Ebola" not in names  # pooled / geo-excluded
    assert all(c["label_ru"] and c["label_ru"] != c["name"] for c in conditions)
    assert all(c["tier"] in ("red", "yellow", "green") for c in conditions)
    assert [c["label_ru"] for c in conditions] == sorted(c["label_ru"] for c in conditions)


def test_evidence_catalog_shape(api):
    client, module = api
    _, headers = register(client, "Catalog")
    assert client.get("/evidence_catalog").status_code in (401, 403)  # no token
    resp = client.get("/evidence_catalog", headers=headers)
    assert resp.status_code == 200, resp.text
    evidences = resp.json()["evidences"]

    # 96 binary model features minus E_211, which is in ml_model._NEVER_ASK
    assert len(evidences) == 95
    codes = [e["code"] for e in evidences]
    assert "E_91" in codes and "E_211" not in codes
    assert len(set(codes)) == len(codes)
    for e in evidences:
        assert e["label"]["en"] and e["label"]["ru"] and e["label"]["kk"], e
    assert [e["label"]["ru"] for e in evidences] == sorted(e["label"]["ru"] for e in evidences)


def test_doctor_feedback_happy_path_and_validation(api, fake_groq):
    client, module = api
    fake_groq.replies.update({"extract": EXTRACTION, "review": REVIEW})
    patient_id, headers = register(client, "Feedback")
    doc = doctor_headers(client)
    _, result = _diagnosed_day(client, headers)
    day, top1, top3 = result["day"], result["diseaseName"], [s["name"] for s in result["slices"]]
    url = "/doctor_feedback"

    resp = client.post(url, headers=doc, json={"day_id": day, "diagnosis": top1, "zone": result["zone"],
                                              "note": "flu test positive"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True and body["agrees_top1"] is True and body["in_top3"] is True

    resp = client.post(url, headers=doc, json={"day_id": day, "diagnosis": top3[1], "zone": "yellow"})
    assert resp.json()["agrees_top1"] is False and resp.json()["in_top3"] is True

    resp = client.post(url, headers=doc, json={"day_id": day, "diagnosis": "other", "zone": "red",
                                              "other_text": "Бронхит с осложнением"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["agrees_top1"] is False and resp.json()["in_top3"] is False
    other_id = resp.json()["feedback_id"]

    # the note is encrypted at rest (when a key is configured)
    conn = module.connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT model_top1, model_version, model_zone, note FROM diagnosis_feedback WHERE feedback_id = %s",
                    (body["feedback_id"],))
        model_top1, model_version, model_zone, note = cur.fetchone()
    finally:
        conn.close()
    assert model_top1 == top1 and model_version == "v2.1-nb" and model_zone == result["zone"]
    assert note != "flu test positive" and module.decrypt_field(note) == "flu test positive"

    # validation
    assert client.post(url, headers=doc, json={"day_id": day, "diagnosis": "other", "zone": "red"}).status_code == 422
    assert client.post(url, headers=doc, json={"day_id": day, "diagnosis": "other", "zone": "red",
                                              "other_text": "   "}).status_code == 422
    assert client.post(url, headers=doc, json={"day_id": day, "diagnosis": "Flu", "zone": "red"}).status_code == 422
    assert client.post(url, headers=doc, json={"day_id": day, "diagnosis": top1, "zone": "purple"}).status_code == 422
    assert client.post(url, headers=headers, json={"day_id": day, "diagnosis": top1, "zone": "red"}).status_code == 403
    assert client.post(url, json={"day_id": day, "diagnosis": top1, "zone": "red"}).status_code in (401, 403)
    assert client.post(url, headers=doc, json={"day_id": 999999, "diagnosis": top1, "zone": "red"}).status_code == 404

    # the latest row is what /get_history shows; other_text is decrypted for the doctor's view
    history = client.post("/get_history", headers=headers, json={"patient_id": patient_id}).json()["history"]
    entry = next(h for h in history if h["day_id"] == day)
    assert entry["feedback"]["diagnosis"] == "other"
    assert entry["feedback"]["label_ru"] is None
    assert entry["feedback"]["other_text"] == "Бронхит с осложнением"
    assert entry["feedback"]["zone"] == "red"
    assert other_id > body["feedback_id"]


def test_history_feedback_is_null_until_confirmed(api, fake_groq):
    client, _ = api
    fake_groq.replies.update({"extract": EXTRACTION, "review": REVIEW})
    patient_id, headers = register(client, "NoFeedback")
    _, result = _diagnosed_day(client, headers)
    history = client.post("/get_history", headers=headers, json={"patient_id": patient_id}).json()["history"]
    assert history[0]["day_id"] == result["day"] and history[0]["feedback"] is None


def test_get_lab_results_patient_with_empty_body(api):
    client, _ = api
    _, headers = register(client, "Labs")
    resp = client.post("/get_lab_results", json={}, headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["results"] == []
    assert client.post("/get_lab_results", headers=headers).status_code == 200  # no body at all
    # doctors must name the patient
    assert client.post("/get_lab_results", json={}, headers=doctor_headers(client)).status_code == 400


# ─── Wave 2: triage pending flag, feedback accuracy, real-case export ───

def _triage_entry(client, doc, patient_id):
    patients = client.post("/list_patients_triage", headers=doc).json()["patients"]
    return next(p for p in patients if p["patient_id"] == patient_id)


def test_triage_zone_null_without_days_and_feedback_pending(api, fake_groq):
    client, _ = api
    fake_groq.replies.update({"extract": EXTRACTION, "review": REVIEW})
    doc = doctor_headers(client)
    patient_id, headers = register(client, "Pending")
    entry = _triage_entry(client, doc, patient_id)
    assert entry["zone"] is None and entry["last_day_id"] is None and entry["feedback_pending"] is False

    _, result = _diagnosed_day(client, headers)
    entry = _triage_entry(client, doc, patient_id)
    assert entry["zone"] == result["zone"]
    assert entry["last_day_id"] == result["day"] and entry["feedback_pending"] is True

    resp = client.post("/doctor_feedback", headers=doc, json={
        "day_id": result["day"], "diagnosis": result["diseaseName"], "zone": result["zone"]})
    assert resp.status_code == 200, resp.text
    assert _triage_entry(client, doc, patient_id)["feedback_pending"] is False

    _, newer = _diagnosed_day(client, headers)  # a new day is pending again
    entry = _triage_entry(client, doc, patient_id)
    assert entry["last_day_id"] == newer["day"] and entry["feedback_pending"] is True


def test_feedback_stats_latest_row_per_day(api, fake_groq):
    client, module = api
    fake_groq.replies.update({"extract": EXTRACTION, "review": REVIEW})
    doc = doctor_headers(client)
    # Stats cover the whole table, so start from an empty one. Later tests in this
    # module do not depend on the feedback rows removed here.
    conn = module.connect()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM diagnosis_feedback")
        conn.commit()
    finally:
        conn.close()

    empty = client.get("/feedback_stats", headers=doc)
    assert empty.status_code == 200, empty.text
    zeros = {d: {"red": 0, "yellow": 0, "green": 0} for d in ("red", "yellow", "green")}
    assert empty.json() == {"n": 0, "by_version": {}, "zone_matrix": zeros, "confusions": []}

    assert client.get("/feedback_stats").status_code in (401, 403)
    _, patient_headers = register(client, "StatsPatient")
    assert client.get("/feedback_stats", headers=patient_headers).status_code == 403

    _, headers = register(client, "Stats")
    _, a = _diagnosed_day(client, headers)
    _, b = _diagnosed_day(client, headers)
    _, c = _diagnosed_day(client, headers)
    top1, top3 = a["diseaseName"], [s["name"] for s in a["slices"]]
    model_zone = a["zone"]
    assert model_zone != "red"  # so a red doctor zone is an under-triage
    url = "/doctor_feedback"

    # day a: an earlier wrong answer that must be ignored, then the agreeing one (doctor says red)
    assert client.post(url, headers=doc, json={"day_id": a["day"], "diagnosis": top3[2], "zone": "green"}).status_code == 200
    assert client.post(url, headers=doc, json={"day_id": a["day"], "diagnosis": top1, "zone": "red"}).status_code == 200
    # day b: in the model's top 3 only
    assert client.post(url, headers=doc, json={"day_id": b["day"], "diagnosis": top3[1], "zone": model_zone}).status_code == 200
    # day c: "other"
    assert client.post(url, headers=doc, json={"day_id": c["day"], "diagnosis": "other", "zone": model_zone,
                                              "other_text": "Бронхит"}).status_code == 200

    resp = client.get("/feedback_stats", headers=doc)
    assert resp.status_code == 200, resp.text
    stats = resp.json()

    zone_matrix = {d: {"red": 0, "yellow": 0, "green": 0} for d in ("red", "yellow", "green")}
    zone_matrix["red"][model_zone] += 1          # day a: doctor red, model not red
    zone_matrix[model_zone][model_zone] += 2     # days b and c agree on zone
    assert stats["n"] == 3
    assert stats["by_version"] == {"v2.1-nb": {
        "n": 3, "other": 1, "top1": round(1 / 3, 3), "top3": round(2 / 3, 3),
        "zone_exact": round(2 / 3, 3), "under_triage": 1, "over_triage": 0}}
    assert stats["zone_matrix"] == zone_matrix
    assert stats["confusions"] == [{"model": top1, "doctor": top3[1], "count": 1}]


def test_export_cases_from_doctor_confirmed_days(api, fake_groq, monkeypatch):
    client, module = api
    sys.path.insert(0, str(Path(__file__).parent / "eval"))
    export = __import__("export_real_cases")
    fake_groq.replies.update({"extract": EXTRACTION, "review": REVIEW})
    doc = doctor_headers(client)
    _, headers = register(client, "Export")
    _, confirmed = _diagnosed_day(client, headers)
    _, unconfirmed = _diagnosed_day(client, headers)
    _, other = _diagnosed_day(client, headers)
    top1 = confirmed["diseaseName"]
    assert client.post("/doctor_feedback", headers=doc, json={
        "day_id": confirmed["day"], "diagnosis": top1, "zone": "yellow"}).status_code == 200
    assert client.post("/doctor_feedback", headers=doc, json={
        "day_id": other["day"], "diagnosis": "other", "zone": "red",
        "other_text": "Бронхит с осложнением"}).status_code == 200

    conn = module.connect()
    try:
        without_text = {c["id"]: c for c in export.export_cases(conn, with_text=False, since=None)}
        with_text = {c["id"]: c for c in export.export_cases(conn, with_text=True, since=None)}
        assert export.export_cases(conn, with_text=True, since="2100-01-01") == []
    finally:
        conn.close()

    assert f"r{unconfirmed['day']}" not in without_text  # no feedback, no case
    case_id = f"r{confirmed['day']}"
    case = without_text[case_id]
    assert case["text"] == "" and with_text[case_id]["text"] == COMPLAINT
    assert case["lang"] == "ru"
    assert "E_91" in case["present"] and "E_181" not in case["present"]
    assert case["absent"] == ["E_181"]
    assert case["pathology"] == top1 and case["tier"] == module.triage.tier_of(top1)
    assert case["out_of_scope"] is False and case["likely_condition"] is None
    assert case["doctor_zone"] == "yellow" and case["model_top1"] == top1
    assert case["model_zone"] == confirmed["zone"] and case["model_version"] == "v2.1-nb"

    other_case = without_text[f"r{other['day']}"]
    assert other_case["pathology"] is None and other_case["tier"] is None
    assert other_case["out_of_scope"] is True
    assert other_case["likely_condition"] == "Бронхит с осложнением"
    assert other_case["text"] == ""


# ─── Round 2: accuracy report endpoint ───

def test_accuracy_report_endpoint_real_patients_and_missing_file(api, fake_groq, monkeypatch, tmp_path):
    from datetime import date
    client, module = api
    fake_groq.replies.update({"extract": EXTRACTION, "review": REVIEW})
    doc = doctor_headers(client)
    _, patient_headers = register(client, "AccuracyPatient")
    assert client.get("/accuracy_report", headers=patient_headers).status_code == 403
    assert client.get("/accuracy_report").status_code in (401, 403)

    # start from one feedback row so the real-patient numbers are exact
    conn = module.connect()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM diagnosis_feedback")
        conn.commit()
    finally:
        conn.close()
    _, headers = register(client, "Accuracy")
    _, result = _diagnosed_day(client, headers)
    resp = client.post("/doctor_feedback", headers=doc, json={
        "day_id": result["day"], "diagnosis": result["diseaseName"], "zone": result["zone"]})
    assert resp.status_code == 200, resp.text

    # a report file next to the endpoint's reader
    report_path = tmp_path / "accuracy_report.json"
    report_path.write_text(json.dumps({"model_version": "v2.1-nb", "synthetic": {"metrics": {}}}), encoding="utf-8")
    monkeypatch.setattr(module, "ACCURACY_REPORT_PATH", str(report_path))
    resp = client.get("/accuracy_report", headers=doc)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model_version"] == "v2.1-nb" and "report_missing" not in body
    real = body["real_patients"]
    assert real["n"] == 1 and real["other"] == 0
    assert real["top1"] == {"value": 1.0, "k": 1, "n": 1, "ci": real["top1"]["ci"]}
    assert set(real["top3"]) == {"value", "k", "n", "ci"} and real["top3"]["k"] == 1
    assert real["by_version"]["v2.1-nb"]["n"] == 1
    assert date.fromisoformat(real["since"])

    # missing file: 200, report_missing, real numbers still there
    monkeypatch.setattr(module, "ACCURACY_REPORT_PATH", str(tmp_path / "missing.json"))
    resp = client.get("/accuracy_report", headers=doc)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["report_missing"] is True and body["real_patients"]["n"] == 1
    assert set(body) == {"report_missing", "real_patients"}
