"""
Security helpers: field encryption (crypto_utils) and JWT tokens (auth).

Run:  python -m pytest -q test_security.py
"""

import importlib
import logging
import os
import sys
from datetime import datetime, timezone

import dotenv
import pytest
from cryptography.fernet import Fernet
from jose import jwt as jose_jwt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def modules():
    """Import crypto_utils and auth here, and forget them afterwards. Both read the
    environment at import time (DB_ENCRYPTION_KEY, JWT_SECRET_KEY); test_api.py sets
    those and must not find a module cached by this file."""
    import auth
    import crypto_utils
    yield crypto_utils, auth
    for name in ("crypto_utils", "auth"):
        sys.modules.pop(name, None)


@pytest.fixture
def crypto(modules, monkeypatch):
    crypto_utils, _ = modules
    monkeypatch.setattr(crypto_utils, "_KEY", Fernet.generate_key().decode())
    return crypto_utils


# ─── crypto_utils.decrypt_field ─────────────────────────────────────────────

def test_round_trip(crypto):
    enc = crypto.encrypt_field("Иванов Иван")
    assert enc != "Иванов Иван"
    assert enc.startswith("gAAAAA")
    assert crypto.decrypt_field(enc) == "Иванов Иван"


def test_legacy_plain_text_passes_through(crypto):
    assert crypto.decrypt_field("Иванов Иван") == "Иванов Иван"
    assert crypto.decrypt_field("") == ""


def test_token_from_another_key_returns_none(crypto, caplog):
    other = Fernet.generate_key()
    secret = "Иванов Иван"
    token = Fernet(other).encrypt(secret.encode("utf-8")).decode("utf-8")
    with caplog.at_level(logging.WARNING, logger="tms.crypto"):
        assert crypto.decrypt_field(token) is None
    # the warning names the problem but never the value
    assert any("could not be decrypted" in r.getMessage() for r in caplog.records)
    assert token not in caplog.text
    assert secret not in caplog.text


def test_none_stays_none(crypto):
    assert crypto.decrypt_field(None) is None
    assert crypto.encrypt_field(None) is None


def test_no_key_returns_value_unchanged(crypto, monkeypatch):
    monkeypatch.setattr(crypto, "_KEY", "")
    assert crypto.decrypt_field("gAAAAAnot-really") == "gAAAAAnot-really"
    assert crypto.encrypt_field("abc") == "abc"


# ─── auth.create_access_token ───────────────────────────────────────────────

def test_token_exp_is_aware_utc_and_round_trips(modules):
    _, auth = modules
    before = datetime.now(timezone.utc)
    token = auth.create_access_token({"sub": "7", "role": "patient"})
    claims = jose_jwt.get_unverified_claims(token)

    exp = datetime.fromtimestamp(claims["exp"], tz=timezone.utc)
    expected = before.timestamp() + auth.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    assert abs(claims["exp"] - expected) < 120  # within two minutes of the configured lifetime
    assert exp > before

    decoded = auth.decode_token(token)
    assert decoded["role"] == "patient"
    assert decoded["sub"] == "7"


def test_missing_jwt_secret_logs_one_error(modules, monkeypatch, caplog):
    _, auth = modules
    # keep load_dotenv from putting a configured key back into the environment
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    try:
        with caplog.at_level(logging.ERROR, logger="tms.auth"):
            importlib.reload(auth)
        errors = [r for r in caplog.records if r.name == "tms.auth" and r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "JWT_SECRET_KEY" in errors[0].getMessage()
        assert len(auth.SECRET_KEY) == 64  # random fallback still in place
    finally:
        monkeypatch.undo()
        importlib.reload(auth)  # back to the environment's configuration
