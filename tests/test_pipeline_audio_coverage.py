"""More pipeline tests for coverage."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path


pytestmark = pytest.mark.skipif(
    not Path("weights/converted/dit_weights.safetensors").exists(),
    reason="Converted weights not available",
)


@pytest.fixture(scope="module")
def pipeline():
    pytest.importorskip("mlx.core")
    from echo_tts_mlx.pipeline import EchoTTS
    return EchoTTS.from_pretrained("weights/converted", dtype="float16")


# ── _prepare_speaker_latents_from_audio ─────────────────────────────────────


def test_prepare_speaker_from_audio(pipeline):
    """Test standard audio extraction to latents."""
    mx = pytest.importorskip("mlx.core")
    
    # 2 seconds of audio (2 * 44100 = 88200 samples)
    audio = np.random.default_rng(0).uniform(-1, 1, (88200,)).astype(np.float32)
    lat, mask = pipeline._prepare_speaker_latents_from_audio(audio)
    
    # Check outputs are properly trimmed to patch_size (4)
    # 88200 / 2048 (downsample factor) = 43.06 frames -> 43 frames
    # 43 % 4 = 3 -> trims down to 40 frames
    assert lat.shape[0] == 1
    assert lat.shape[1] == 40
    assert lat.shape[2] == 80
    assert mask.shape[1] == 40


def test_prepare_speaker_from_audio_too_short(pipeline):
    # Only 100 samples (less than 1 frame downsample factor of 2048)
    audio = np.random.default_rng(0).uniform(-1, 1, (100,)).astype(np.float32)
    with pytest.raises(ValueError, match="too short: need at least 2048"):
        pipeline._prepare_speaker_latents_from_audio(audio)


def test_prepare_speaker_from_audio_pad_needed(pipeline):
    """Test chunk padding path when audio is not a multiple of chunk size."""
    mx = pytest.importorskip("mlx.core")
    # max_latent_length = 640
    # chunk_samples = 640 * 512 = 327680
    # Provide 400000 samples, so second chunk needs padding
    audio = np.random.default_rng(0).uniform(-1, 1, (400000,)).astype(np.float32)
    lat, mask = pipeline._prepare_speaker_latents_from_audio(audio)
    assert lat.shape[0] == 1
    assert mask.shape[1] > 0


def test_prepare_speaker_from_audio_too_short_after_trim(pipeline):
    """Test case where valid samples result in 0 patches."""
    mx = pytest.importorskip("mlx.core")
    # Provide exactly enough for 3 frames (3 * 2048 = 6144)
    # But patch size is 4, so 3 frames trims down to 0 frames
    audio = np.random.default_rng(0).uniform(-1, 1, (6144,)).astype(np.float32)
    with pytest.raises(ValueError, match="fewer than one patch"):
        pipeline._prepare_speaker_latents_from_audio(audio)


# ── Error conditions in generate ──────────────────────────────────────────


def test_generate_latents_without_speaker(pipeline):
    """Test the unconditional path in generation."""
    mx = pytest.importorskip("mlx.core")
    # Quick 1-step dummy generation just to hit the code path
    latents = pipeline.generate_latents(
        text="[S1] Hello.",
        num_steps=1,
        seed=42,
        sequence_length=16,
    )
    assert latents.shape == (1, 16, 80)


def test_decode_latents_without_trim(pipeline):
    """Test decode path without tail trimming."""
    mx = pytest.importorskip("mlx.core")
    latents = mx.zeros((1, 16, 80), dtype=mx.float32)
    audio = pipeline.decode_latents(latents, trim_latents=False)
    # 16 frames * 2048 downsample = 32768 samples
    assert audio.shape == (1, 1, 32768)


def test_save_audio_wrapper(pipeline, tmp_path):
    """Test pipeline.save_audio() integration."""
    audio = np.zeros((1, 1, 100), dtype=np.float32)
    out_file = tmp_path / "test.wav"
    pipeline.save_audio(audio, out_file)
    assert out_file.exists()

