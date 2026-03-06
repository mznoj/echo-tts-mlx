from __future__ import annotations

from pathlib import Path

import pytest

from echo_tts_mlx._conversion_utils import (
    DEFAULT_DAC_CHECKPOINT,
    load_and_fold_dac_state,
    read_safetensor_header,
)


@pytest.mark.skipif(not DEFAULT_DAC_CHECKPOINT.exists(), reason="upstream DAC checkpoint not present")
def test_dac_checkpoint_header_contains_expected_keys() -> None:
    header = read_safetensor_header(DEFAULT_DAC_CHECKPOINT)

    assert "encoder.block.0.conv.parametrizations.weight.original1" in header
    assert "decoder.model.0.conv.parametrizations.weight.original1" in header
    assert "quantizer.pre_module.layers.0.attention.wqkv.weight" in header
    assert "quantizer.semantic_quantizer.quantizers.0.codebook.weight" in header


@pytest.mark.skipif(not DEFAULT_DAC_CHECKPOINT.exists(), reason="upstream DAC checkpoint not present")
def test_fold_weight_norm_eliminates_raw_weight_norm_keys() -> None:
    folded, stats = load_and_fold_dac_state(DEFAULT_DAC_CHECKPOINT)

    assert stats.folded_new_style == 60
    assert stats.folded_old_style == 20

    # New-style folded convs should exist.
    assert "encoder.block.0.conv.weight" in folded
    assert "decoder.model.0.conv.weight" in folded

    # Old-style VQ projections should exist.
    assert "quantizer.semantic_quantizer.quantizers.0.in_proj.weight" in folded
    assert "quantizer.semantic_quantizer.quantizers.0.out_proj.weight" in folded
    assert "quantizer.quantizer.quantizers.0.in_proj.weight" in folded

    # Raw weight norm keys should be removed.
    assert not any("parametrizations.weight.original" in k for k in folded)
    assert not any(k.endswith(".weight_v") or k.endswith(".weight_g") for k in folded)


@pytest.mark.skipif(not DEFAULT_DAC_CHECKPOINT.exists(), reason="upstream DAC checkpoint not present")
def test_folded_shapes_match_expected() -> None:
    folded, _ = load_and_fold_dac_state(DEFAULT_DAC_CHECKPOINT)

    assert tuple(folded["encoder.block.0.conv.weight"].shape) == (64, 1, 7)
    assert tuple(folded["decoder.model.1.block.1.conv.weight"].shape) == (1536, 768, 16)
    assert tuple(folded["quantizer.pre_module.layers.0.attention.wqkv.weight"].shape) == (3072, 1024)
    assert tuple(folded["quantizer.quantizer.quantizers.0.codebook.weight"].shape) == (1024, 8)
