"""Text normalization and fuzzy similarity, shared by the ASR and OCR
dialogue matchers so both apply identical robustness rules (see DESIGN.md
section 6): case, punctuation, and whitespace are normalized away, and
comparisons are never exact-string-equality since both ASR and OCR output
is expected to contain occasional recognition errors.

Pure functions, no I/O, no external dependencies — trivially unit testable.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_PUNCTUATION_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = _PUNCTUATION_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    """Fuzzy similarity in [0, 1] between two strings, after normalization.

    Uses difflib's ratio (Ratcliff/Obershelp), which tolerates the kind of
    small insertions/substitutions typical of ASR and OCR errors without
    requiring an extra dependency.
    """
    a_norm = normalize(a)
    b_norm = normalize(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def word_count(text: str) -> int:
    """Word count of the normalized text, used to size sliding windows."""
    norm = normalize(text)
    return len(norm.split()) if norm else 0
