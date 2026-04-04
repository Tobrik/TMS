"""
Field-level encryption for sensitive medical data in SQLite.
Uses Fernet (AES-128-CBC + HMAC-SHA256) from the cryptography library.
Backwards compatible: decrypt_field() handles unencrypted legacy data gracefully.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from cryptography.fernet import Fernet, InvalidToken

_KEY = os.environ.get("DB_ENCRYPTION_KEY", "")


def get_fernet() -> Fernet | None:
    if not _KEY:
        return None
    return Fernet(_KEY.encode())


def encrypt_field(value: str | None) -> str | None:
    """Encrypt a string value. Returns original if no key configured."""
    if value is None:
        return None
    f = get_fernet()
    if f is None:
        return value
    return f.encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_field(value: str | None) -> str | None:
    """Decrypt a string value. Returns original if not encrypted (backwards compat)."""
    if value is None:
        return None
    f = get_fernet()
    if f is None:
        return value
    try:
        return f.decrypt(value.encode("utf-8")).decode("utf-8")
    except (InvalidToken, Exception):
        # Data was stored before encryption was enabled — return as-is
        return value
