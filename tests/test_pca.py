from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from echo_tts_mlx.pca import load_pca_state, pca_decode_np, pca_encode_np, resolve_converted_pca_path


WEIGHTS_DIR = Path("weights/converted")
PCA_PATH = WEIGHTS_DIR / "pca_state.safetensors"


@pytest.mark.skipif(not PCA_PATH.exists(), reason="converted PCA state not present")
def test_resolve_converted_pca_path() -> None:
    assert resolve_converted_pca_path(WEIGHTS_DIR) == PCA_PATH


@pytest.mark.skipif(not PCA_PATH.exists(), reason="converted PCA state not present")
def test_pca_encode_decode_formulas_match_manual() -> None:
    state = load_pca_state(WEIGHTS_DIR)
    z_q = np.linspace(-1.0, 1.0, num=1 * 1024 * 5, dtype=np.float32).reshape(1, 1024, 5)

    encoded = pca_encode_np(z_q, state)
    manual_encoded = (
        (np.transpose(z_q, (0, 2, 1)) - state.pca_mean.reshape(1, 1, 1024)) @ state.pca_components.T
    ) * np.float32(state.latent_scale)
    assert np.allclose(encoded, manual_encoded, atol=1e-6)

    decoded = pca_decode_np(encoded, state)
    manual_decoded = np.transpose(
        (encoded / np.float32(state.latent_scale)) @ state.pca_components + state.pca_mean.reshape(1, 1, 1024),
        (0, 2, 1),
    )
    assert np.allclose(decoded, manual_decoded, atol=1e-6)
