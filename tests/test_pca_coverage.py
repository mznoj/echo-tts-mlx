"""Additional PCA tests to bring coverage ≥92%."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from echo_tts_mlx.pca import (
    PCAState,
    load_pca_state,
    pca_decode_np,
    pca_encode_np,
    resolve_converted_pca_path,
)


def _make_pca_state() -> PCAState:
    rng = np.random.default_rng(42)
    return PCAState(
        pca_components=rng.standard_normal((80, 1024)).astype(np.float32),
        pca_mean=rng.standard_normal((1024,)).astype(np.float32),
        latent_scale=0.5,
    )


# ── resolve_converted_pca_path ─────────────────────────────────────────────


def test_resolve_raises_on_file(tmp_path: Path):
    f = tmp_path / "dummy.safetensors"
    f.write_text("data")
    with pytest.raises(ValueError, match="Expected converted weights directory"):
        resolve_converted_pca_path(f)


def test_resolve_raises_missing_pca(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Missing converted PCA state"):
        resolve_converted_pca_path(tmp_path)


# ── pca_encode_np ──────────────────────────────────────────────────────────


def test_pca_encode_wrong_shape_2d():
    state = _make_pca_state()
    with pytest.raises(ValueError, match="shape.*B, 1024, T"):
        pca_encode_np(np.zeros((1024, 10), dtype=np.float32), state)


def test_pca_encode_wrong_dim():
    state = _make_pca_state()
    with pytest.raises(ValueError, match="shape.*B, 1024, T"):
        pca_encode_np(np.zeros((1, 512, 10), dtype=np.float32), state)


def test_pca_encode_output_shape():
    state = _make_pca_state()
    z_q = np.random.default_rng(0).standard_normal((1, 1024, 50)).astype(np.float32)
    result = pca_encode_np(z_q, state)
    assert result.shape == (1, 50, 80)


# ── pca_decode_np ──────────────────────────────────────────────────────────


def test_pca_decode_wrong_shape_2d():
    state = _make_pca_state()
    with pytest.raises(ValueError, match="shape.*B, T, 80"):
        pca_decode_np(np.zeros((50, 80), dtype=np.float32), state)


def test_pca_decode_wrong_dim():
    state = _make_pca_state()
    with pytest.raises(ValueError, match="shape.*B, T, 80"):
        pca_decode_np(np.zeros((1, 50, 40), dtype=np.float32), state)


def test_pca_decode_output_shape():
    state = _make_pca_state()
    z = np.random.default_rng(0).standard_normal((1, 50, 80)).astype(np.float32)
    result = pca_decode_np(z, state)
    assert result.shape == (1, 1024, 50)


# ── encode/decode roundtrip ───────────────────────────────────────────────


def test_pca_roundtrip_approximate():
    state = _make_pca_state()
    z_q = np.random.default_rng(99).standard_normal((1, 1024, 20)).astype(np.float32)
    encoded = pca_encode_np(z_q, state)
    decoded = pca_decode_np(encoded, state)
    assert decoded.shape == z_q.shape
    # Not exact since PCA projects to 80 < 1024, but shape should match


# ── load_pca_state from real weights ──────────────────────────────────────


def test_load_pca_state_real():
    """Verify real PCA state loads correctly."""
    weights_dir = Path("weights/converted")
    if not (weights_dir / "pca_state.safetensors").exists():
        pytest.skip("Converted weights not available")

    state = load_pca_state(weights_dir)
    assert state.pca_components.shape == (80, 1024)
    assert state.pca_mean.shape == (1024,)
    assert isinstance(state.latent_scale, float)
    assert state.latent_scale > 0
