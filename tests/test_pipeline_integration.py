from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from echo_tts_mlx.pipeline import resolve_quality_preset
from echo_tts_mlx.sampler import build_timestep_schedule, find_flattening_point


WEIGHTS_DIR = Path("weights/converted")
CONFIG_PATH = WEIGHTS_DIR / "config.json"
DIT_PATH = WEIGHTS_DIR / "dit_weights.safetensors"
DAC_PATH = WEIGHTS_DIR / "dac_weights.safetensors"
PCA_PATH = WEIGHTS_DIR / "pca_state.safetensors"


def _mlx_runtime_available() -> bool:
    if importlib.util.find_spec("mlx") is None:
        return False
    proc = subprocess.run(
        [sys.executable, "-c", "import mlx.core as mx; _ = mx.array([0], dtype=mx.float16)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


HAS_MLX = _mlx_runtime_available()
HAS_CONVERTED = CONFIG_PATH.exists() and DIT_PATH.exists() and DAC_PATH.exists() and PCA_PATH.exists()


def test_schedule_shape_and_scale() -> None:
    t = build_timestep_schedule(32)
    assert t.shape == (33,)
    assert np.isclose(float(t[0]), 0.999, atol=1e-6)
    assert np.isclose(float(t[-1]), 0.0, atol=1e-6)


def test_flattening_heuristic_detects_tail() -> None:
    active = np.ones((20, 80), dtype=np.float32) * 0.3
    flat = np.zeros((30, 80), dtype=np.float32)
    latents = np.concatenate([active, flat], axis=0)
    idx = find_flattening_point(latents, window_size=10, std_threshold=0.01, mean_threshold=0.05)
    assert idx == 20


def test_quality_preset_values() -> None:
    steps, trunc = resolve_quality_preset("balanced", sequence_length=200)
    assert steps == 16
    assert trunc == pytest.approx(0.8)


@pytest.mark.skipif(not HAS_CONVERTED, reason="converted weights not present")
@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed or runtime unavailable")
def test_prepare_speaker_latents_direct_trims_to_patch_multiple() -> None:
    from echo_tts_mlx.pipeline import EchoTTS

    pipeline = EchoTTS.from_pretrained(WEIGHTS_DIR, dtype="float16")
    latents = np.zeros((1, 11, 80), dtype=np.float32)
    out_latents, out_mask = pipeline.prepare_speaker_latents(speaker_latents=latents)

    assert tuple(out_latents.shape) == (1, 8, 80)
    assert tuple(out_mask.shape) == (1, 8)
