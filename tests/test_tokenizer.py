from __future__ import annotations

import pytest

from echo_tts_mlx.tokenizer import apply_speaker_prefix, normalize_text, tokenize


def test_tokenizer_normalization_rules_exact() -> None:
    raw = "Hello… \u201cworld\u201d \u2019test\u2019:\nline;A—B"
    normalized = normalize_text(raw)
    assert normalized == 'Hello... "world" \'test\', line,A, B'


def test_tokenizer_auto_prefix_s1() -> None:
    assert apply_speaker_prefix("Hello there") == "[S1] Hello there"
    assert apply_speaker_prefix("[S2] Hello there") == "[S2] Hello there"
    assert apply_speaker_prefix("(aside) Hello there") == "(aside) Hello there"
    assert apply_speaker_prefix("contains S1 marker") == "contains S1 marker"
    assert apply_speaker_prefix("contains S2 marker") == "contains S2 marker"


def test_tokenizer_bos_and_utf8_bytes() -> None:
    tokens = tokenize("Hi")
    assert tokens[0] == 0
    assert tokens[1:] == list("[S1] Hi".encode("utf-8"))


def test_tokenizer_truncation_to_max_length() -> None:
    text = "a" * 2000
    with pytest.warns(UserWarning):
        tokens = tokenize(text, max_length=768)
    assert len(tokens) == 768
