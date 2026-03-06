from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


WEIGHTS_DIR = Path("weights/converted")
CONFIG_PATH = WEIGHTS_DIR / "config.json"
DAC_PATH = WEIGHTS_DIR / "dac_weights.safetensors"


def _mlx_runtime_available() -> bool:
    if importlib.util.find_spec("mlx") is None:
        return False
    proc = subprocess.run(
        [sys.executable, "-c", "import mlx.core as mx; _ = mx.array([0], dtype=mx.float32)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


HAS_MLX = _mlx_runtime_available()


def _known_waveform(samples: int = 16_384, sample_rate: int = 44_100) -> np.ndarray:
    """Deterministic harmonic fixture used for round-trip quality testing."""
    t = np.arange(samples, dtype=np.float32) / np.float32(sample_rate)
    env = 0.5 * (1.0 + np.sin(2.0 * np.pi * 2.0 * t))
    x = env * (
        0.6 * np.sin(2.0 * np.pi * 130.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 260.0 * t)
        + 0.1 * np.sin(2.0 * np.pi * 390.0 * t)
    )
    x = np.tanh(x).astype(np.float32)
    return x.reshape(1, 1, samples)


def _best_aligned_metrics(reference: np.ndarray, candidate: np.ndarray, max_lag: int = 64) -> tuple[float, float, int]:
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float32).reshape(-1)
    best_snr = float("-inf")
    best_corr = float("-inf")
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            ref_seg = ref[lag:]
            cand_seg = cand[: len(ref_seg)]
        else:
            cand_seg = cand[-lag:]
            ref_seg = ref[: len(cand_seg)]

        n = min(len(ref_seg), len(cand_seg))
        if n < 32:
            continue

        ref_seg = ref_seg[:n]
        cand_seg = cand_seg[:n]

        err = ref_seg - cand_seg
        num = float(np.sum(ref_seg * ref_seg) + 1e-12)
        den = float(np.sum(err * err) + 1e-12)
        snr = 10.0 * np.log10(num / den)
        corr = float(np.corrcoef(ref_seg, cand_seg)[0, 1])

        if snr > best_snr:
            best_snr = float(snr)
            best_corr = corr
            best_lag = lag

    return best_snr, best_corr, best_lag


def test_dac_float32_only_policy() -> None:
    from echo_tts_mlx.autoencoder import MlxFishS1DAC

    with pytest.raises(ValueError):
        MlxFishS1DAC.from_pretrained(WEIGHTS_DIR, dtype="float16")


@pytest.mark.skipif(not CONFIG_PATH.exists() or not DAC_PATH.exists(), reason="converted DAC weights not present")
def test_resolve_converted_dac_paths() -> None:
    from echo_tts_mlx.autoencoder import resolve_converted_dac_paths

    config_path, dac_path = resolve_converted_dac_paths(WEIGHTS_DIR)
    assert config_path == CONFIG_PATH
    assert dac_path == DAC_PATH


@pytest.mark.skipif(not CONFIG_PATH.exists() or not DAC_PATH.exists(), reason="converted DAC weights not present")
@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed or runtime unavailable")
def test_mlx_encode_decode_smoke() -> None:
    import mlx.core as mx

    from echo_tts_mlx.autoencoder import MlxFishS1DAC

    model = MlxFishS1DAC.from_pretrained(WEIGHTS_DIR, dtype="float32")
    audio_np = _known_waveform(samples=4096)

    z_q, codes, latents = model.encode_zq(audio_np)
    recon = model.decode_zq(z_q)
    mx.eval(z_q, codes, latents, recon)

    assert tuple(z_q.shape) == (1, 1024, 2)
    assert tuple(codes.shape) == (10, 1, 2)
    assert tuple(latents.shape) == (1, 80, 2)
    assert tuple(recon.shape) == tuple(audio_np.shape)
    assert bool(np.isfinite(np.asarray(recon)).all())


@pytest.mark.skipif(not CONFIG_PATH.exists() or not DAC_PATH.exists(), reason="converted DAC weights not present")
@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed or runtime unavailable")
def test_mlx_roundtrip_quality_gate() -> None:
    import mlx.core as mx

    from echo_tts_mlx.autoencoder import MlxFishS1DAC

    model = MlxFishS1DAC.from_pretrained(WEIGHTS_DIR, dtype="float32")
    audio_np = _known_waveform(samples=16_384)

    z_q, _, _ = model.encode_zq(audio_np)
    recon = model.decode_zq(z_q)
    mx.eval(z_q, recon)
    recon_np = np.asarray(recon).astype(np.float32)

    snr_db, corr, lag = _best_aligned_metrics(audio_np, recon_np, max_lag=64)

    assert snr_db >= 5.0, f"mlx round-trip SNR too low: {snr_db:.3f} dB (lag={lag})"
    assert corr >= 0.85, f"mlx round-trip correlation too low: {corr:.4f} (lag={lag})"
