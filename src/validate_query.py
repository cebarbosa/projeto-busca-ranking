from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any
import unicodedata

from .config import load_config

DEFAULT_MAX_CHARS = 512
DEFAULT_MAX_TOKENS = 64

# Strip control characters (Cc) except standard whitespace, then collapse whitespace.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")
# Keep letters, numbers, spaces, hyphens, and slashes; replace other symbols/punctuation with space.
_PUNCT_SYMBOL_RE = re.compile(r"[^\w\s\-/]+", flags=re.UNICODE)


@dataclass(frozen=True)
class ValidatedQuery:
    """Normalized query text along with validation and truncation metadata."""

    text: str
    raw_text: str
    was_truncated: bool
    stripped_special_chars: bool
    original_char_count: int
    final_char_count: int


def _truncate_at_word_boundary(text: str, max_chars: int) -> tuple[str, bool]:
    """Truncate text to at most max_chars without cutting mid-word when practical."""
    if len(text) <= max_chars:
        return text, False
    if max_chars <= 0:
        return "", True

    slice_candidate = text[:max_chars]
    last_space = slice_candidate.rfind(" ")
    # Avoid chopping right at the start of a long first token.
    if last_space >= max(1, max_chars // 2):
        return slice_candidate[:last_space].strip(), True
    return slice_candidate.strip(), True


def _cap_tokens(text: str, max_tokens: int | None) -> tuple[str, bool]:
    """Cap whitespace-delimited token count if max_tokens is specified."""
    if max_tokens is None or max_tokens <= 0:
        return text, False
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text, False
    return " ".join(tokens[:max_tokens]), True


def _resolve_query_limits(
    *,
    max_chars: int | None,
    max_tokens: int | None,
    strip_special_chars: bool | None,
    allow_punctuation_only: bool | None,
    cfg: dict[str, Any] | None,
) -> tuple[int, int | None, bool, bool]:
    qv_cfg = ((cfg or load_config()).get("query_validation") or {})

    resolved_max_chars = (
        DEFAULT_MAX_CHARS
        if max_chars is None and "max_chars" not in qv_cfg
        else (int(qv_cfg["max_chars"]) if max_chars is None else int(max_chars))
    )

    if max_tokens is not None:
        resolved_max_tokens: int | None = int(max_tokens)
    elif "max_tokens" in qv_cfg:
        cfg_tokens = qv_cfg["max_tokens"]
        resolved_max_tokens = None if cfg_tokens is None else int(cfg_tokens)
    else:
        resolved_max_tokens = DEFAULT_MAX_TOKENS

    resolved_strip_special = (
        bool(qv_cfg.get("strip_special_chars", False))
        if strip_special_chars is None
        else bool(strip_special_chars)
    )
    resolved_allow_punct = (
        bool(qv_cfg.get("allow_punctuation_only", False))
        if allow_punctuation_only is None
        else bool(allow_punctuation_only)
    )

    return (
        resolved_max_chars,
        resolved_max_tokens,
        resolved_strip_special,
        resolved_allow_punct,
    )


def inspect_query(
    query: object,
    *,
    max_chars: int | None = None,
    max_tokens: int | None = None,
    strip_special_chars: bool | None = None,
    allow_punctuation_only: bool | None = None,
    cfg: dict[str, Any] | None = None,
) -> ValidatedQuery:
    """Validate and normalize a raw search query, returning text and diagnostic flags."""
    if query is None or (isinstance(query, float) and math.isnan(query)):
        raise ValueError("Search query cannot be null or empty.")
    if not isinstance(query, str):
        raise ValueError(f"Search query must be a string, got {type(query).__name__}.")

    raw_text = query
    original_len = len(raw_text)

    normalized = unicodedata.normalize("NFKC", raw_text)
    normalized = _CONTROL_CHARS_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()

    if not normalized:
        raise ValueError("Search query cannot be empty or whitespace-only.")

    effective_max_chars, effective_max_tokens, effective_strip_special, effective_allow_punct = (
        _resolve_query_limits(
            max_chars=max_chars,
            max_tokens=max_tokens,
            strip_special_chars=strip_special_chars,
            allow_punctuation_only=allow_punctuation_only,
            cfg=cfg,
        )
    )

    cleaned = normalized
    stripped_special = False
    if effective_strip_special:
        after_punct = _PUNCT_SYMBOL_RE.sub(" ", cleaned)
        after_punct = _WHITESPACE_RE.sub(" ", after_punct).strip()
        if after_punct:
            stripped_special = after_punct != cleaned
            cleaned = after_punct
        elif not effective_allow_punct:
            raise ValueError("Search query contains no searchable terms.")

    was_truncated = False
    cleaned, char_trunc = _truncate_at_word_boundary(cleaned, max_chars=effective_max_chars)
    was_truncated = was_truncated or char_trunc

    cleaned, tok_trunc = _cap_tokens(cleaned, max_tokens=effective_max_tokens)
    was_truncated = was_truncated or tok_trunc

    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    if not cleaned:
        raise ValueError("Search query became empty after normalization.")

    return ValidatedQuery(
        text=cleaned,
        raw_text=raw_text,
        was_truncated=was_truncated,
        stripped_special_chars=stripped_special,
        original_char_count=original_len,
        final_char_count=len(cleaned),
    )


def validate_query(
    query: object,
    *,
    max_chars: int | None = None,
    max_tokens: int | None = None,
    strip_special_chars: bool | None = None,
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return a clean, validated search string or raise ValueError for invalid input."""
    return inspect_query(
        query,
        max_chars=max_chars,
        max_tokens=max_tokens,
        strip_special_chars=strip_special_chars,
        cfg=cfg,
    ).text