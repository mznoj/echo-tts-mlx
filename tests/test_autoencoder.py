"""Tests for echo_tts_mlx.autoencoder — target ≥92% coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from echo_tts_mlx.autoencoder import (
    resolve_converted_dac_paths,
    load_converted_dac_state,
)


# ── resolve_converted_dac_paths ────────────────────────────────────────────


def test_resolve_raises_on_file_path(tmp_path: Path):
    f = tmp_path / "weights.safetensors"
    f.write_text("dummy")
    with pytest.raises(ValueError, match="Expected converted weights directory"):
        resolve_converted_dac_paths(f)


def test_resolve_raises_missing_config(tmp_path: Path):
    (tmp_path / "dac_weights.safetensors").write_text("dummy")
    with pytest.raises(FileNotFoundError, match="Missing converted config"):
        resolve_converted_dac_paths(tmp_path)


def test_resolve_raises_missing_dac_weights(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="Missing converted DAC weights"):
        resolve_converted_dac_paths(tmp_path)


def test_resolve_success(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "dac_weights.safetensors").write_text("dummy")
    cfg, dac = resolve_converted_dac_paths(tmp_path)
    assert cfg.name == "config.json"
    assert dac.name == "dac_weights.safetensors"


# ── MlxFishS1DAC ──────────────────────────────────────────────────────────


def test_from_pretrained_rejects_non_float32():
    from echo_tts_mlx.autoencoder import MlxFishS1DAC

    with pytest.raises(ValueError, match="float32-only"):
        MlxFishS1DAC.from_pretrained("weights/converted", dtype="float16")


def test_to_mx_array_from_numpy():
    """Test the _to_mx_array internal method handles numpy input."""
    pytest.importorskip("mlx.core")
    from echo_tts_mlx.autoencoder import MlxFishS1DAC

    # We need a minimal instance — load from the real weights
    weights_dir = Path("weights/converted")
    if not (weights_dir / "dac_weights.safetensors").exists():
        pytest.skip("Converted weights not available")

    dac = MlxFishS1DAC.from_pretrained(weights_dir)

    import mlx.core as mx

    # Test numpy path
    arr_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = dac._to_mx_array(arr_np)
    assert result.dtype == mx.float32

    # Test MLX array path (already mlx)
    result2 = dac._to_mx_array(result)
    assert result2.dtype == mx.float32


# ── load_converted_dac_state ──────────────────────────────────────────────


def test_load_converted_dac_state_casts_non_float32():
    """Verify that non-float32 tensors are cast to float32."""
    pytest.importorskip("safetensors")

    weights_dir = Path("weights/converted")
    if not (weights_dir / "dac_weights.safetensors").exists():
        pytest.skip("Converted weights not available")

    state = load_converted_dac_state(weights_dir)
    for key, val in state.items():
        if np.issubdtype(val.dtype, np.floating):
            assert val.dtype == np.float32, f"Key {key} has dtype {val.dtype}, expected float32"
