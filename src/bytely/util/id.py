"""Stable content identifiers and text normalization."""

import hashlib
import re


def content_hash(text: str) -> str:
    """Return the full hex SHA-256 of the text's UTF-8 bytes.

    Matches the reference implementation's content hash, so `body_hash`
    values are identical to its values for the same source.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_body(text: str) -> str:
    """Normalize text by collapsing whitespace."""
    return re.sub(r"\s+", " ", text).strip()
