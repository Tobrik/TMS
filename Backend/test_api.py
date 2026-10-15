"""
End-to-end API test on a throwaway PostgreSQL (pip install pgserver) with a fake
Groq: register → extract → questions → analyse → explain → history → doctor.
"""

import importlib
import os

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
