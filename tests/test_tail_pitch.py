from __future__ import annotations

import numpy as np
import pytest

from echo_tts_mlx.pipeline import resolve_adaptive_truncation, resolve_quality_preset
from echo_tts_mlx.sampler import find_content_boundary


def _latents_with_tail(*, active_frames: int, total_frames: int) -> np.ndarray:
    latents = np.zeros((total_frames, 80), dtype=np.float32)
    latents[:active_frames, :] = 0.25
    return latents


def test_find_content_boundary_latent_mode_matches_flattening() -> None:
    latents = _latents_with_tail(active_frames=40, total_frames=100)
    audio = np.ones((100 * 2048,), dtype=np.float32) * 0.1

    idx = find_content_boundary(
        latents,
        audio,
        energy_enabled=False,
        f0_enabled=False,
        latent_window=10,
        latent_std_threshold=0.01,
        latent_mean_threshold=0.05,
    )
    assert idx == 40


def test_find_content_boundary_uses_energy_when_within_guardrail() -> None:
    latents = _latents_with_tail(active_frames=100, total_frames=120)
    audio = np.zeros((120 * 2048,), dtype=np.float32)
    audio[: 80 * 2048] = 0.3
    audio[80 * 2048 :] = 1e-5

    idx = find_content_boundary(
        latents,
        audio,
        energy_enabled=True,
        energy_threshold_db=-20.0,
        energy_hop_samples=2048,
        f0_enabled=False,
        min_retained_ratio=0.5,
    )
    assert idx == 80


def test_find_content_boundary_guardrail_falls_back_to_latent() -> None:
    latents = _latents_with_tail(active_frames=100, total_frames=120)
    audio = np.zeros((120 * 2048,), dtype=np.float32)
    audio[: 20 * 2048] = 0.3
    audio[20 * 2048 :] = 1e-5

    idx = find_content_boundary(
        latents,
        audio,
        energy_enabled=True,
        energy_threshold_db=-20.0,
        energy_hop_samples=2048,
        f0_enabled=False,
        min_retained_ratio=0.5,
    )
    assert idx == 100


def test_find_content_boundary_f0_onset_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    latents = _latents_with_tail(active_frames=100, total_frames=120)
    audio = np.ones((120 * 2048,), dtype=np.float32) * 0.1

    monkeypatch.setattr(
        "echo_tts_mlx.sampler._analyze_f0_variance",
        lambda **_kwargs: {"onset_latent_frame": 70},
    )
    idx = find_content_boundary(
        latents,
        audio,
        energy_enabled=False,
        f0_enabled=True,
        min_retained_ratio=0.5,
    )
    assert idx == 70


def test_find_content_boundary_f0_requires_librosa(monkeypatch: pytest.MonkeyPatch) -> None:
    latents = _latents_with_tail(active_frames=100, total_frames=120)
    audio = np.ones((120 * 2048,), dtype=np.float32) * 0.1

    def _raise_import_error(**_kwargs):
        raise ImportError("missing")

    monkeypatch.setattr("echo_tts_mlx.sampler._analyze_f0_variance", _raise_import_error)
    with pytest.raises(ImportError, match="pip install librosa"):
        find_content_boundary(latents, audio, energy_enabled=False, f0_enabled=True)


def test_resolve_adaptive_truncation_interpolates_and_clamps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "echo_tts_mlx.pipeline.ADAPTIVE_TRUNCATION",
        {100: 0.7, 200: 0.9},
    )
    assert resolve_adaptive_truncation(50) == pytest.approx(0.7)
    assert resolve_adaptive_truncation(300) == pytest.approx(0.9)
    assert resolve_adaptive_truncation(150) == pytest.approx(0.8)


def test_resolve_quality_preset_auto_uses_adaptive_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "echo_tts_mlx.pipeline.ADAPTIVE_TRUNCATION",
        {100: 0.74, 640: 0.88},
    )
    steps, trunc = resolve_quality_preset("balanced", sequence_length=100)
    assert steps == 16
    assert trunc == pytest.approx(0.74)
