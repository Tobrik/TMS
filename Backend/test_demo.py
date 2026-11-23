"""Demo mode: /demo/* endpoints, seeding, the queue additions, and the real/demo isolation rule.
Uses a small fake DEMO_CASES list so these tests do not depend on demo_seed.py's data."""

import pytest
from test_api import api, register, doctor_headers, COMPLAINT, EXTRACTION, REVIEW  # noqa: F401

LONG_COMPLAINT = "Кашляю уже неделю, жёлтая мокрота, температура, справа в груди колет. " * 4  # > 160 chars
RED = {"evidences": ["E_66", "E_214"], "absent": ["E_201", "E_91"]}
PNEUMONIA = {"evidences": ["E_201", "E_77", "E_91", "E_129", "E_53", "E_55_@_V_159"], "absent": ["E_181"]}
OTITIS = {"evidences": ["E_53", "E_55_@_V_130", "E_114", "E_181", "E_201"], "absent": ["E_161"]}

FAKE_CASES = [
    {"key": "t-green", "name": "Мадина Ахметова", "city": "Алматы", "sex": "F", "age": 30,
     "days": [{"minutes_ago": 60, "lang": "ru", "complaint": "Болит ухо.", **OTITIS, "unknown": []}]},
    {"key": "t-yellow-old", "name": "Ерлан Нурланов", "city": "Астана", "sex": "M", "age": 19,
     "days": [{"minutes_ago": 3000, "lang": "ru", "complaint": "Кашель второй день.", **PNEUMONIA, "unknown": []},
              {"minutes_ago": 120, "lang": "ru", "complaint": LONG_COMPLAINT, **PNEUMONIA, "unknown": []}]},
    {"key": "t-red", "name": "Айгуль Сеитова", "city": "Шымкент", "sex": "F", "age": 34,
     "days": [{"minutes_ago": 30, "lang": "kk", "complaint": "Тыныс алу қиын.", **RED, "unknown": []}]},
    {"key": "t-yellow-new", "name": "Данияр Бекенов", "city": "Караганда", "sex": "M", "age": 19,
     "days": [{"minutes_ago": 10, "lang": "en", "complaint": "Cough and fever.", **PNEUMONIA, "unknown": []}]},
]
SEEDED_ORDER = ["Айгуль Сеитова", "Данияр Бекенов", "Ерлан Нурланов", "Мадина Ахметова"]  # red, yellow new/old, green
MISSING = 99999999


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def demo(api, monkeypatch):
    client, module = api
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setattr(module, "_demo_cases", lambda: FAKE_CASES)
    _sql(module, "DELETE FROM patients WHERE is_demo; DELETE FROM doctors WHERE is_demo")
    return client, module


def _sql(module, query, params=()):
    conn = module.connect()
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
    finally:
        conn.close()


def _rows(module, query, params=()):
    conn = module.connect()
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        return cur.fetchall()
    finally:
        conn.close()


def demo_doctor(client):
    resp = client.post("/demo/doctor")
    assert resp.status_code == 200, resp.text
    return resp.json()


def queue(client, headers):
    resp = client.post("/list_patients_triage", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["patients"]


# ─── switch ───

@pytest.mark.parametrize("value", [None, "0", "false", ""])
def test_demo_disabled(api, monkeypatch, value):
    client, _ = api
    if value is None:
        monkeypatch.delenv("DEMO_MODE", raising=False)
    else:
        monkeypatch.setenv("DEMO_MODE", value)
    assert client.get("/demo/status").json() == {"enabled": False}
    for path in ("/demo/doctor", "/demo/patient"):
        resp = client.post(path)
        assert resp.status_code == 404 and resp.json() == {"detail": "demo_disabled"}


def test_demo_status_enabled(demo):
    client, _ = demo
    assert client.get("/demo/status").json() == {"enabled": True}


# ─── /demo/doctor ───

def test_demo_doctor_token_and_profile(demo):
    client, module = demo
    import auth
    body = demo_doctor(client)
    claims = auth.decode_token(body["access_token"])
    assert claims["role"] == "doctor" and claims["demo"] is True
    assert body["doctor"]["username"] == "demo" and body["doctor"]["is_demo"] is True
    assert body["doctor"]["full_name"] == "Демо-врач" and body["doctor"]["specialty"] == "Терапевт"
    # the demo doctor cannot log in with a password
    assert client.post("/login_doctor", json={"username": "demo", "password": "anything1"}).status_code == 401


def test_demo_doctor_seeds_once(demo):
    client, module = demo
    demo_doctor(client)
    demo_doctor(client)
    assert _rows(module, "SELECT COUNT(*) FROM patients WHERE demo_seed IS NOT NULL")[0][0] == len(FAKE_CASES)
    assert _rows(module, "SELECT COUNT(*) FROM doctors WHERE is_demo")[0][0] == 1
    # one day per seeded day, not doubled
    assert _rows(module, "SELECT COUNT(*) FROM diary_days d JOIN patients p USING(patient_id) WHERE p.is_demo")[0][0] == 5


def test_seed_is_safe_against_a_race(demo):
    """Two requests passing the 'not there yet' check: the unique partial index lets one win."""
    client, module = demo
    case = FAKE_CASES[0]
    assert module._seed_demo_case(case) is True
    assert module._seed_demo_case(case) is False
    assert _rows(module, "SELECT COUNT(*) FROM patients WHERE demo_seed = %s", (case["key"],))[0][0] == 1
    with pytest.raises(Exception):  # the index itself, not only the ON CONFLICT clause
        _sql(module, "INSERT INTO patients(full_name, password_hash, is_demo, demo_seed) VALUES ('x','x',TRUE,%s)",
             (case["key"],))


def test_demo_queue_content_and_order(demo):
    client, _ = demo
    hdr = _bearer(demo_doctor(client)["access_token"])
    rows = queue(client, hdr)
    assert [r["full_name"] for r in rows] == SEEDED_ORDER
    assert [r["zone"] for r in rows] == ["red", "yellow", "yellow", "green"]
    assert all(r["is_demo"] is True for r in rows)
    red, yellow_new, yellow_old, green = rows
    assert red["complaint"] == "Тыныс алу қиын."
    assert red["last_disease"].startswith(("Anaphylaxis", "Bronchospasm")) and red["reasons"]
    assert isinstance(green["reasons"], list) and green["low_confidence"] in (True, False)
    # the latest day is what the queue shows; a long complaint is cut with an ellipsis
    assert yellow_old["complaint"].endswith("…") and len(yellow_old["complaint"]) <= 160
    assert yellow_old["complaint"].startswith("Кашляю уже неделю")
    assert yellow_new["complaint"] == "Cough and fever."
    assert yellow_new["diag_date"] > yellow_old["diag_date"]


def test_complaint_is_null_without_an_intake_and_reasons_default(demo):
    client, module = demo
    resp = client.post("/demo/patient", json={"name": "Без жалобы"})
    pid = resp.json()["patient_id"]
    hdr = _bearer(demo_doctor(client)["access_token"])
    row = next(r for r in queue(client, hdr) if r["patient_id"] == pid)
    assert row["complaint"] is None and row["reasons"] == [] and row["low_confidence"] is False
    assert row["zone"] is None
    assert queue(client, hdr)[-1]["patient_id"] == pid  # not assessed: last


# ─── /demo/patient ───

def test_demo_patient_names_and_token(demo):
    client, module = demo
    import auth
    resp = client.post("/demo/patient", json={"name": "Айгерим", "lang": "kk"})
    assert resp.status_code == 200
    body = resp.json()
    claims = auth.decode_token(body["access_token"])
    assert claims == {**claims, "role": "patient", "patient_id": body["patient_id"], "demo": True}
    row = _rows(module, "SELECT full_name, is_demo, demo_seed FROM patients WHERE patient_id=%s", (body["patient_id"],))[0]
    assert module.decrypt_field(row[0]) == "Айгерим" and row[1] is True and row[2] is None
    for lang, prefix in (("ru", "Гость"), ("kk", "Қонақ"), ("en", "Guest")):
        pid = client.post("/demo/patient", json={"lang": lang}).json()["patient_id"]
        name = module.decrypt_field(_rows(module, "SELECT full_name FROM patients WHERE patient_id=%s", (pid,))[0][0])
        assert name == f"{prefix} {pid}"
    assert client.post("/demo/patient").status_code == 200  # no body at all


def test_live_demo_patient_flow_reaches_the_demo_queue(demo, fake_groq):
    client, module = demo
    fake_groq.replies.update({"extract": EXTRACTION, "review": REVIEW})
    body = client.post("/demo/patient", json={"name": "Гость-тест"}).json()
    hdr = _bearer(body["access_token"])
    extracted = client.post("/extract_symptoms", json={"text": COMPLAINT}, headers=hdr)
    assert extracted.status_code == 200, extracted.text
    result = client.post("/analys", headers=hdr, json={
        "evidences": extracted.json()["evidences"], "absent": extracted.json()["absent"],
        "intake_id": extracted.json()["intake_id"]})
    assert result.status_code == 200, result.text
    doc = _bearer(demo_doctor(client)["access_token"])
    row = next(r for r in queue(client, doc) if r["patient_id"] == body["patient_id"])
    assert row["complaint"] == COMPLAINT and row["zone"] == result.json()["zone"]
    assert row["is_demo"] is True and row["last_day_id"] == result.json()["day"]
    # a real doctor never sees it
    assert body["patient_id"] not in [r["patient_id"] for r in queue(client, doctor_headers(client))]


# ─── cleanup ───

def test_cleanup_removes_old_live_patients_and_keeps_seeded(demo):
    client, module = demo
    demo_doctor(client)
    old = client.post("/demo/patient", json={"name": "Старый"}).json()["patient_id"]
    fresh = client.post("/demo/patient", json={"name": "Новый"}).json()["patient_id"]
    day = module.insert_disease(old, ["E_91"], "URTI", 0.5)
    _sql(module, "UPDATE patients SET created_at = NOW() - INTERVAL '49 hours' WHERE patient_id = %s", (old,))
    _sql(module, "UPDATE patients SET created_at = NOW() - INTERVAL '90 days' WHERE demo_seed IS NOT NULL")
    demo_doctor(client)
    ids = {r[0] for r in _rows(module, "SELECT patient_id FROM patients WHERE is_demo")}
    assert old not in ids and fresh in ids
    assert _rows(module, "SELECT COUNT(*) FROM patients WHERE demo_seed IS NOT NULL")[0][0] == len(FAKE_CASES)
    assert _rows(module, "SELECT COUNT(*) FROM diary_days WHERE day_id = %s", (day,))[0][0] == 0  # cascade


# ─── isolation ───

def _real_patient_with_day(client, module):
    pid, hdr = register(client, "Реальный Пациент")
    day = client.post("/analys", headers=hdr, json={"evidences": PNEUMONIA["evidences"], "age": 40, "sex": "M"}).json()["day"]
    module.save_lab_result(pid, "ОАК", "2026-01-01", [{"name": "Hb", "value": "1"}], "ok", "f.jpg")
    _sql(module, "INSERT INTO diary_symptoms(day_id, symptom_code, value) VALUES (%s,'FEVER',1)", (day,))
    return pid, day


def _calls(client, headers, pid, day):
    """One request per doctor-reachable endpoint that takes a patient_id or day_id."""
    return {
        "get_history": client.post("/get_history", headers=headers, json={"patient_id": pid}),
        "get_symptoms": client.post("/get_symptoms", headers=headers, json={"patient_id": pid, "symptom_str": "FEVER"}),
        "get_patient_info": client.post("/get_patient_info", headers=headers, json={"patient_id": pid}),
        "get_day_symptoms": client.post("/get_day_symptoms", headers=headers, json={"day_id": day}),
        "get_lab_results": client.post("/get_lab_results", headers=headers, json={"patient_id": pid}),
        "update_by_doctor": client.post("/update_by_doctor", headers=headers, json={
            "patient_id": pid, "day_id": day, "doctor_id": 1, "recept": "pwned"}),
        "doctor_feedback": client.post("/doctor_feedback", headers=headers, json={
            "day_id": day, "diagnosis": "other", "other_text": "x", "zone": "green"}),
    }


def _same_as_missing(client, headers, pid, day):
    hidden = _calls(client, headers, pid, day)
    missing = _calls(client, headers, MISSING, MISSING)
    for name in hidden:
        assert (hidden[name].status_code, hidden[name].json()) == (missing[name].status_code, missing[name].json()), name


def test_real_doctor_cannot_reach_demo_patients_and_vice_versa(demo):
    client, module = demo
    demo_hdr = _bearer(demo_doctor(client)["access_token"])
    real_hdr = doctor_headers(client)
    seeded_pid = next(r["patient_id"] for r in queue(client, demo_hdr) if r["zone"] == "red")
    seeded_day = _rows(module, "SELECT MAX(day_id) FROM diary_days WHERE patient_id = %s", (seeded_pid,))[0][0]
    # /get_symptoms still reads the legacy v1 codes, so plant one row to have something to hide
    _sql(module, "INSERT INTO diary_symptoms(day_id, symptom_code, value) VALUES (%s,'FEVER',1)", (seeded_day,))
    module.save_lab_result(seeded_pid, "ОАК", "2026-01-01", [{"name": "Hb", "value": "1"}], "ok", "f.jpg")
    real_pid, real_day = _real_patient_with_day(client, module)

    # same answers as for ids that do not exist, in both directions
    _same_as_missing(client, real_hdr, seeded_pid, seeded_day)
    _same_as_missing(client, demo_hdr, real_pid, real_day)

    # positive controls: each doctor does see their own side
    own = _calls(client, demo_hdr, seeded_pid, seeded_day)
    assert own["get_history"].json()["history"] and own["get_symptoms"].json()["symptoms_arr"]
    assert own["get_patient_info"].json()["found"] is True and own["get_day_symptoms"].json()["symptoms"]
    assert own["get_lab_results"].json()["results"] and own["update_by_doctor"].json() == {"ok": True}
    assert own["doctor_feedback"].status_code == 200
    own_real = _calls(client, real_hdr, real_pid, real_day)
    assert own_real["get_history"].json()["history"] and own_real["get_patient_info"].json()["found"] is True
    assert own_real["doctor_feedback"].status_code == 200

    # nothing was written across the line
    assert _rows(module, "SELECT COUNT(*) FROM diagnosis_feedback WHERE day_id = %s", (real_day,))[0][0] == 1  # only the real doctor's own control call

    # queues are disjoint
    real_ids = {r["patient_id"] for r in queue(client, real_hdr)}
    demo_ids = {r["patient_id"] for r in queue(client, demo_hdr)}
    assert real_pid in real_ids and real_pid not in demo_ids
    assert seeded_pid in demo_ids and seeded_pid not in real_ids
    assert all(r["is_demo"] is False for r in queue(client, real_hdr))


def test_hidden_day_with_a_visible_patient_id_is_still_hidden(demo):
    client, module = demo
    demo_hdr = _bearer(demo_doctor(client)["access_token"])
    real_pid, real_day = _real_patient_with_day(client, module)
    seeded_pid = queue(client, demo_hdr)[0]["patient_id"]
    resp = client.post("/update_by_doctor", headers=demo_hdr, json={
        "patient_id": seeded_pid, "day_id": real_day, "doctor_id": 1, "recept": "pwned"})
    assert resp.json() == {"ok": False}


def test_patients_keep_their_own_data_checks(demo):
    client, module = demo
    demo_patient = client.post("/demo/patient").json()
    real_pid, _ = register(client)
    hdr = _bearer(demo_patient["access_token"])
    assert client.post("/get_history", headers=hdr, json={"patient_id": real_pid}).status_code == 403
    assert client.post("/get_history", headers=hdr, json={"patient_id": demo_patient["patient_id"]}).status_code == 200


def test_feedback_on_demo_days_does_not_change_the_stats(demo):
    client, module = demo
    demo_hdr = _bearer(demo_doctor(client)["access_token"])
    real_hdr = doctor_headers(client)
    before = client.get("/feedback_stats", headers=real_hdr).json()
    before_report = client.get("/accuracy_report", headers=real_hdr).json()["real_patients"]
    row = queue(client, demo_hdr)[0]
    resp = client.post("/doctor_feedback", headers=demo_hdr, json={
        "day_id": row["last_day_id"], "diagnosis": "Pneumonia", "zone": "yellow"})
    assert resp.status_code == 200
    assert _rows(module, "SELECT COUNT(*) FROM diagnosis_feedback")[0][0] >= 1
    assert client.get("/feedback_stats", headers=real_hdr).json() == before
    assert client.get("/feedback_stats", headers=demo_hdr).json() == before
    assert client.get("/accuracy_report", headers=real_hdr).json()["real_patients"] == before_report


def test_real_demo_seed_data_seeds_and_matches_expectations(api, monkeypatch):
    """With B3's real DEMO_CASES: every case seeds, and the queue shows each case's expected zone."""
    demo_seed = pytest.importorskip("demo_seed")
    client, module = api
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setattr(module, "_demo_cases", lambda: demo_seed.DEMO_CASES)
    _sql(module, "DELETE FROM patients WHERE is_demo; DELETE FROM doctors WHERE is_demo")
    rows = queue(client, _bearer(demo_doctor(client)["access_token"]))
    assert len(rows) == len(demo_seed.DEMO_CASES)
    by_name = {r["full_name"]: r for r in rows}
    for case in demo_seed.DEMO_CASES:
        row = by_name[case["name"]]
        last = case["days"][-1]
        assert row["zone"] == last["expect_zone"], case["key"]
        assert row["last_disease"] == last["expect_top1"], case["key"]
    order = [{"red": 0, "yellow": 1, "green": 2}[r["zone"]] for r in rows]
    assert order == sorted(order)


def test_queue_and_history_carry_top3_and_the_full_complaint(demo):
    """The doctor's view needs the model's whole top-3 and the patient's own words per day."""
    client, _ = demo
    hdr = _bearer(demo_doctor(client)["access_token"])
    rows = queue(client, hdr)
    for row in rows:
        assert 1 <= len(row["top3"]) <= 3
        first = row["top3"][0]
        assert set(first) == {"name", "label_ru", "p"} and 0 < first["p"] <= 1
        assert [t["p"] for t in row["top3"]] == sorted((t["p"] for t in row["top3"]), reverse=True)
    yellow_old = rows[2]
    history = client.post("/get_history", json={"patient_id": yellow_old["patient_id"]}, headers=hdr).json()["history"]
    latest = history[0]
    assert latest["complaint"].startswith("Кашляю уже неделю") and not latest["complaint"].endswith("…")
    assert latest["top3"] and latest["top3"][0]["name"] == yellow_old["top3"][0]["name"]
    assert isinstance(latest["reasons"], list) and latest["low_confidence"] in (True, False)
