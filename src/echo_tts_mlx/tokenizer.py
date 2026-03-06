"""Echo-TTS UTF-8 byte tokenizer with upstream normalization rules."""

from __future__ import annotations

import warnings


def normalize_text(text: str) -> str:
    """Apply the exact punctuation normalization rules from the spec."""
    text = text.replace("…", "...")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\n", " ")
    text = text.replace(":", ",")
    text = text.replace(";", ",")
    text = text.replace("—", ", ")
    return text


def apply_speaker_prefix(text: str) -> str:
    """Prepend `[S1] ` when no explicit speaker marker/prefix is present."""
    if (not text.startswith("[") and not text.startswith("(") and "S1" not in text and "S2" not in text):
        return "[S1] " + text
    return text


def tokenize(text: str, *, max_length: int = 768, warn_on_truncate: bool = True) -> list[int]:
    """Tokenize text into raw UTF-8 bytes with BOS token 0."""
    normalized = normalize_text(text)
    normalized = apply_speaker_prefix(normalized)

    tokens = [0] + list(normalized.encode("utf-8"))
    if len(tokens) > max_length:
        if warn_on_truncate:
            warnings.warn(
                f"Tokenized text length {len(tokens)} exceeds max_length={max_length}; truncating.",
                UserWarning,
                stacklevel=2,
            )
        tokens = tokens[:max_length]

    return tokens
