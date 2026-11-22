"""
Field-level encryption for sensitive medical data in SQLite.
Uses Fernet (AES-128-CBC + HMAC-SHA256) from the cryptography library.
Backwards compatible: decrypt_field() handles unencrypted legacy data gracefully.
"""

import logging
import os
from dotenv import load_dotenv

load_dotenv()

from cryptography.fernet import Fernet

logger = logging.getLogger("tms.crypto")

_KEY = os.environ.get("DB_ENCRYPTION_KEY", "")
_TOKEN_PREFIX = "gAAAAA"  # every Fernet token starts with this (version byte 0x80 + timestamp)


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
    """Decrypt a string value. Plain text (stored before encryption was enabled) is
    returned as it is. A value that looks like a Fernet token but does not decrypt
    (wrong or rotated key, corruption) returns None, so ciphertext is never shown
    as data."""
    if value is None:
        return None
    f = get_fernet()
    if f is None:
        return value
    try:
        return f.decrypt(value.encode("utf-8")).decode("utf-8")
    except Exception:
        if value.startswith(_TOKEN_PREFIX):
            logger.warning("encrypted field could not be decrypted (wrong or rotated DB_ENCRYPTION_KEY?); returning None")
            return None
        # Legacy plain-text value, stored before encryption was enabled
        return value
