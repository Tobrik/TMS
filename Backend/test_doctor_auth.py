"""Doctor accounts: username login, invites, registration, password change, demo token limits."""

import pytest
from test_api import api  # noqa: F401  (module-scoped throwaway Postgres + app; seeds "Dr Test|…|Doctor123")


def _login(client, **kw):
    return client.post("/login_doctor", json=kw)


def _hdr(resp):
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _admin(client):
    resp = _login(client, username="doctor1", password="Doctor123")
    assert resp.status_code == 200, resp.text
    return resp


def _new_invite(client):
    resp = client.post("/doctor/invites", headers=_hdr(_admin(client)))
    assert resp.status_code == 200, resp.text
    return resp.json()["code"]


def _register(client, code, username="aliya.k", password="Secret123"):
    return client.post("/register_doctor", json={
        "invite_code": code, "username": username, "full_name": "Алия Касымова",
        "specialty": "Терапевт", "password": password})


def _sql(module, query, params=()):
    conn = module.connect()
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
    finally:
        conn.close()


# ─── login ───

def test_login_by_username_is_case_insensitive(api):
    client, _ = api
    resp = _login(client, username="  DOCTOR1 ", password="Doctor123")
    assert resp.status_code == 200
    body = resp.json()
    assert body["login"] is True and body["token_type"] == "bearer"
    doc = body["doctor"]
    assert doc["username"] == "doctor1" and doc["full_name"] == "Dr Test"
    assert doc["specialty"] == "Терапевт" and doc["is_demo"] is False


def test_login_by_legacy_doctor_id(api):
    client, _ = api
    doctor_id = _admin(client).json()["doctor"]["doctor_id"]
    resp = _login(client, doctor_id=doctor_id, password="Doctor123")
    assert resp.status_code == 200 and resp.json()["doctor"]["username"] == "doctor1"


def test_wrong_password_and_unknown_user_look_the_same(api):
    client, _ = api
    wrong = _login(client, username="doctor1", password="nope-nope1")
    unknown = _login(client, username="nobody", password="Doctor123")
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json() == {"detail": "Invalid credentials"}


def test_username_and_doctor_id_together_or_neither_is_422(api):
    client, _ = api
    assert _login(client, username="doctor1", doctor_id=1, password="Doctor123").status_code == 422
    assert _login(client, password="Doctor123").status_code == 422


def test_demo_doctor_cannot_log_in_with_a_password(api):
    client, module = api
    _sql(module, "INSERT INTO doctors(full_name, specialty, password_hash, username, is_demo) VALUES (%s,%s,%s,%s,TRUE)",
         (module.encrypt_field("Демо-врач"), module.encrypt_field("Терапевт"), module.hash_password("Whatever123"), "demo"))
    try:
        resp = _login(client, username="demo", password="Whatever123")
        assert resp.status_code == 401 and resp.json() == {"detail": "Invalid credentials"}
    finally:
        _sql(module, "DELETE FROM doctors WHERE username = 'demo'")


# ─── seeding ───

@pytest.fixture
def clean_doctors(api):
    """Empty doctors table for the test, original seed restored afterwards (ids change, usernames do not)."""
    _, module = api
    _sql(module, "DELETE FROM doctors; ALTER SEQUENCE doctors_doctor_id_seq RESTART WITH 1")
    yield module
    _sql(module, "DELETE FROM doctors; ALTER SEQUENCE doctors_doctor_id_seq RESTART WITH 1")
    import os
    os.environ["DOCTOR_SEEDS"] = "Dr Test|Терапевт|Doctor123"
    module.seed_doctors()


def test_seed_four_part_and_legacy_three_part(clean_doctors, monkeypatch):
    module = clean_doctors
    monkeypatch.setenv("DOCTOR_SEEDS", "Admin.One|Dr Admin|Хирург|Admin1234; Old Name|Кардиолог|Old12345 ;bad|entry")
    module.seed_doctors()
    conn = module.connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT doctor_id, username, full_name, specialty, is_demo FROM doctors ORDER BY doctor_id")
        rows = cur.fetchall()
    finally:
        conn.close()
    assert len(rows) == 2
    assert rows[0][1] == "admin.one"
    assert module.decrypt_field(rows[0][2]) == "Dr Admin" and module.decrypt_field(rows[0][3]) == "Хирург"
    assert rows[1][1] == f"doctor{rows[1][0]}"
    assert module.decrypt_field(rows[1][2]) == "Old Name" and rows[1][4] is False


def test_seed_skips_when_a_real_doctor_exists(clean_doctors, monkeypatch):
    module = clean_doctors
    monkeypatch.setenv("DOCTOR_SEEDS", "first|One|Терапевт|Passw0rd1")
    module.seed_doctors()
    monkeypatch.setenv("DOCTOR_SEEDS", "second|Two|Терапевт|Passw0rd2")
    module.seed_doctors()
    conn = module.connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT username FROM doctors")
        assert [r[0] for r in cur.fetchall()] == ["first"]
    finally:
        conn.close()


# ─── invites and registration ───

def test_invite_flow_create_register_login(api):
    client, _ = api
    code = _new_invite(client)
    assert len(code) == 13 and code.startswith("TMS-") and code[8] == "-"
    reg = _register(client, code.lower().replace("-", " "))  # normalised before hashing
    assert reg.status_code == 200, reg.text
    doc = reg.json()["doctor"]
    assert doc["username"] == "aliya.k" and doc["full_name"] == "Алия Касымова" and doc["is_demo"] is False
    assert _login(client, username="Aliya.K", password="Secret123").status_code == 200
    me = client.get("/doctor/me", headers=_hdr(reg))
    assert me.status_code == 200 and me.json()["username"] == "aliya.k"


def test_invite_is_single_use(api):
    client, _ = api
    code = _new_invite(client)
    assert _register(client, code, username="first.user").status_code == 200
    again = _register(client, code, username="second.user")
    assert again.status_code == 400 and again.json() == {"detail": "invalid_invite"}
    assert _login(client, username="second.user", password="Secret123").status_code == 401


def test_unknown_and_expired_invites_are_400(api):
    client, module = api
    assert _register(client, "TMS-AAAA-BBBB", username="ghost.one").json() == {"detail": "invalid_invite"}
    code = _new_invite(client)
    _sql(module, "UPDATE doctor_invites SET expires_at = NOW() - INTERVAL '1 minute' WHERE code_hash = %s",
         (module._invite_hash(code),))
    resp = _register(client, code, username="late.one")
    assert resp.status_code == 400 and resp.json() == {"detail": "invalid_invite"}


def test_taken_username_is_409_and_keeps_the_invite(api):
    client, _ = api
    code = _new_invite(client)
    resp = _register(client, code, username="DOCTOR1")
    assert resp.status_code == 409 and resp.json() == {"detail": "username_taken"}
    assert _register(client, code, username="free.name").status_code == 200


def test_register_validation_is_422(api):
    client, _ = api
    code = _new_invite(client)
    assert _register(client, code, username="A B").status_code == 422
    assert _register(client, code, username="ab").status_code == 422
    assert _register(client, code, username="okname", password="short1").status_code == 422
    assert _register(client, code, username="okname", password="onlyletters").status_code == 422
    assert _register(client, code, username="okname", password="12345678").status_code == 422
    assert _register(client, code, username="okname").status_code == 200  # code was not burnt by the 422s


# ─── change password ───

def test_change_password(api):
    client, _ = api
    code = _new_invite(client)
    reg = _register(client, code, username="pw.changer")
    hdr = _hdr(reg)
    bad = client.post("/doctor/change_password", headers=hdr, json={"old_password": "Wrong1234", "new_password": "NewSecret9"})
    assert bad.status_code == 400 and bad.json() == {"detail": "wrong_password"}
    weak = client.post("/doctor/change_password", headers=hdr, json={"old_password": "Secret123", "new_password": "weak"})
    assert weak.status_code == 422
    ok = client.post("/doctor/change_password", headers=hdr, json={"old_password": "Secret123", "new_password": "NewSecret9"})
    assert ok.status_code == 200 and ok.json() == {"ok": True}
    assert _login(client, username="pw.changer", password="Secret123").status_code == 401
    assert _login(client, username="pw.changer", password="NewSecret9").status_code == 200


# ─── list_doctor and demo tokens ───

def test_list_doctor_requires_a_doctor_token(api):
    client, _ = api
    assert client.post("/list_doctor").status_code in (401, 403)
    resp = client.post("/list_doctor", headers=_hdr(_admin(client)))
    assert resp.status_code == 200
    doctors = resp.json()["doctors"]
    assert any(d["username"] == "doctor1" for d in doctors)
    # a patient token is not enough
    from test_api import register
    _, patient_hdr = register(client)
    assert client.post("/list_doctor", headers=patient_hdr).status_code == 403


def test_demo_token_is_refused_by_real_doctor_endpoints(api):
    client, _ = api
    import auth
    token = auth.create_access_token({"role": "doctor", "doctor_id": 1, "demo": True})
    hdr = {"Authorization": f"Bearer {token}"}
    inv = client.post("/doctor/invites", headers=hdr)
    pw = client.post("/doctor/change_password", headers=hdr, json={"old_password": "Doctor123", "new_password": "Another123"})
    assert inv.status_code == pw.status_code == 403
    assert inv.json() == pw.json() == {"detail": "demo_forbidden"}
    # but /doctor/me is open to any doctor token
    assert client.get("/doctor/me", headers=hdr).status_code == 200


# ─── reserved username, list_doctor and the demo flag ───

def test_demo_is_a_reserved_username(api):
    client, _ = api
    code = _new_invite(client)
    resp = _register(client, code, username="DEMO")
    assert resp.status_code == 409 and resp.json() == {"detail": "username_taken"}
    assert _register(client, code, username="not.demo").status_code == 200  # the invite was not used up


def test_seed_with_the_reserved_username_is_skipped(clean_doctors, monkeypatch):
    module = clean_doctors
    monkeypatch.setenv("DOCTOR_SEEDS", "demo|Fake|Терапевт|Passw0rd1;real.one|Real|Терапевт|Passw0rd2")
    module.seed_doctors()
    conn = module.connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT username FROM doctors")
        assert [r[0] for r in cur.fetchall()] == ["real.one"]
    finally:
        conn.close()


def test_list_doctor_only_shows_doctors_with_the_callers_demo_flag(api):
    client, module = api
    import auth
    _sql(module, "INSERT INTO doctors(full_name, specialty, password_hash, username, is_demo) "
                 "VALUES (%s,%s,'x','demo.list',TRUE) ON CONFLICT DO NOTHING",
         (module.encrypt_field("Демо"), module.encrypt_field("Терапевт")))
    real = client.post("/list_doctor", headers=_hdr(_admin(client))).json()["doctors"]
    assert real and all(d["username"] != "demo.list" for d in real)
    demo_hdr = {"Authorization": "Bearer " + auth.create_access_token({"role": "doctor", "doctor_id": 1, "demo": True})}
    demo = client.post("/list_doctor", headers=demo_hdr).json()["doctors"]
    assert [d["username"] for d in demo] == ["demo.list"]
