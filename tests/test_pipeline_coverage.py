"""Additional pipeline tests to improve coverage — validation paths, error handling."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


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


# Skip everything if MLX isn't available or weights aren't present
pytestmark = pytest.mark.skipif(
    (not Path("weights/converted/dit_weights.safetensors").exists()) or (not HAS_MLX),
    reason="Converted weights not available or MLX runtime unavailable",
)


@pytest.fixture(scope="module")
def pipeline():
    """Load the pipeline once for all tests in this module."""
    mx = pytest.importorskip("mlx.core")
    from echo_tts_mlx.pipeline import EchoTTS

    return EchoTTS.from_pretrained("weights/converted", dtype="float16")


# ── _validate_quantize_mode ────────────────────────────────────────────────


def test_validate_quantize_invalid():
    from echo_tts_mlx.pipeline import _validate_quantize_mode

    with pytest.raises(ValueError, match="Invalid quantize mode"):
        _validate_quantize_mode("q3")


def test_validate_quantize_case_insensitive():
    from echo_tts_mlx.pipeline import _validate_quantize_mode

    assert _validate_quantize_mode("None") == "none"
    assert _validate_quantize_mode(" 8BIT ") == "8bit"
    assert _validate_quantize_mode(" MXFP4 ") == "mxfp4"


def test_from_pretrained_accepts_quantize_mode(monkeypatch: pytest.MonkeyPatch):
    from echo_tts_mlx.pipeline import EchoTTS

    sentinel_config = object()
    sentinel_model = object()
    sentinel_autoencoder = object()
    sentinel_pca = object()
    seen: dict[str, object] = {}

    class DummyEchoTTS(EchoTTS):
        def __init__(self, *, model, autoencoder, config, pca_state, quantize, weights_dir=None):  # type: ignore[no-untyped-def]
            seen["model"] = model
            seen["autoencoder"] = autoencoder
            seen["config"] = config
            seen["pca_state"] = pca_state
            seen["quantize"] = quantize

    monkeypatch.setattr("echo_tts_mlx.pipeline.load_model_config", lambda _weights_dir: sentinel_config)
    monkeypatch.setattr(
        "echo_tts_mlx.pipeline.MlxEchoDiT.from_pretrained",
        lambda _weights_dir, dtype="float16", quantize="none": sentinel_model,
    )
    monkeypatch.setattr(
        "echo_tts_mlx.pipeline.MlxFishS1DAC.from_pretrained",
        lambda _weights_dir, dtype="float32": sentinel_autoencoder,
    )
    monkeypatch.setattr("echo_tts_mlx.pipeline.load_pca_state", lambda _weights_dir: sentinel_pca)

    out = DummyEchoTTS.from_pretrained("weights/converted", quantize="4bit")
    assert isinstance(out, DummyEchoTTS)
    assert seen["model"] is sentinel_model
    assert seen["autoencoder"] is sentinel_autoencoder
    assert seen["config"] is sentinel_config
    assert seen["pca_state"] is sentinel_pca
    assert seen["quantize"] == "4bit"


# ── _is_mlx_array ─────────────────────────────────────────────────────────


def test_is_mlx_array():
    mx = pytest.importorskip("mlx.core")
    from echo_tts_mlx.pipeline import _is_mlx_array

    assert _is_mlx_array(mx.array([1.0]))
    assert not _is_mlx_array(np.array([1.0]))


# ── _to_mx_array ──────────────────────────────────────────────────────────


def test_to_mx_array_numpy(pipeline):
    mx = pytest.importorskip("mlx.core")
    arr = np.array([1.0, 2.0], dtype=np.float32)
    result = pipeline._to_mx_array(arr, dtype=mx.float32)
    assert result.dtype == mx.float32


def test_to_mx_array_already_mlx(pipeline):
    mx = pytest.importorskip("mlx.core")
    arr = mx.array([1.0, 2.0], dtype=mx.float16)
    result = pipeline._to_mx_array(arr, dtype=mx.float32)
    assert result.dtype == mx.float32


def test_to_mx_array_no_dtype(pipeline):
    mx = pytest.importorskip("mlx.core")
    arr = mx.array([1.0, 2.0], dtype=mx.float16)
    result = pipeline._to_mx_array(arr)
    assert result.dtype == mx.float16  # unchanged


# ── prepare_text ───────────────────────────────────────────────────────────


def test_prepare_text(pipeline):
    mx = pytest.importorskip("mlx.core")
    ids, mask = pipeline.prepare_text("[S1] Hello world.")
    assert ids.shape[0] == 1
    assert mask.shape[0] == 1
    assert ids.shape[1] == mask.shape[1]


# ── _normalize_audio ───────────────────────────────────────────────────────


def test_normalize_audio_1d(pipeline):
    audio = np.random.default_rng(42).uniform(-1, 1, (44100,)).astype(np.float32)
    result = pipeline._normalize_audio(audio)
    assert result.shape == (1, 1, 44100)
    assert result.max() <= 1.0


def test_normalize_audio_2d_batch(pipeline):
    audio = np.random.default_rng(0).uniform(-0.5, 0.5, (1, 44100)).astype(np.float32)
    result = pipeline._normalize_audio(audio)
    assert result.shape == (1, 1, 44100)


def test_normalize_audio_2d_column(pipeline):
    audio = np.random.default_rng(0).uniform(-0.5, 0.5, (44100, 1)).astype(np.float32)
    result = pipeline._normalize_audio(audio)
    assert result.shape == (1, 1, 44100)


def test_normalize_audio_2d_stereo_wide(pipeline):
    # (2, 44100) → rows <= cols → mean(axis=0)
    audio = np.random.default_rng(0).uniform(-0.5, 0.5, (2, 44100)).astype(np.float32)
    result = pipeline._normalize_audio(audio)
    assert result.shape == (1, 1, 44100)


def test_normalize_audio_2d_stereo_tall(pipeline):
    # (44100, 2) → rows > cols → mean(axis=1)
    audio = np.random.default_rng(0).uniform(-0.5, 0.5, (44100, 2)).astype(np.float32)
    result = pipeline._normalize_audio(audio)
    assert result.shape == (1, 1, 44100)


def test_normalize_audio_3d(pipeline):
    audio = np.random.default_rng(0).uniform(-0.5, 0.5, (1, 1, 44100)).astype(np.float32)
    result = pipeline._normalize_audio(audio)
    assert result.shape == (1, 1, 44100)


def test_normalize_audio_3d_multichannel(pipeline):
    audio = np.random.default_rng(0).uniform(-0.5, 0.5, (1, 2, 44100)).astype(np.float32)
    result = pipeline._normalize_audio(audio)
    assert result.shape == (1, 1, 44100)


def test_normalize_audio_3d_batch_gt1_raises(pipeline):
    audio = np.zeros((2, 1, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="batch=1"):
        pipeline._normalize_audio(audio)


def test_normalize_audio_4d_raises(pipeline):
    audio = np.zeros((1, 1, 1, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="Unsupported speaker_audio shape"):
        pipeline._normalize_audio(audio)


# ── pca_encode / pca_decode ────────────────────────────────────────────────


def test_pca_encode_wrong_shape(pipeline):
    mx = pytest.importorskip("mlx.core")
    with pytest.raises(ValueError, match="shape"):
        pipeline.pca_encode(mx.zeros((1, 512, 10), dtype=mx.float32))


def test_pca_decode_wrong_shape(pipeline):
    mx = pytest.importorskip("mlx.core")
    with pytest.raises(ValueError, match="shape"):
        pipeline.pca_decode(mx.zeros((1, 10, 40), dtype=mx.float32))


# ── prepare_speaker_latents ───────────────────────────────────────────────


def test_prepare_speaker_both_raises(pipeline):
    with pytest.raises(ValueError, match="not both"):
        pipeline.prepare_speaker_latents(
            speaker_latents=np.zeros((1, 100, 80), dtype=np.float32),
            speaker_audio=np.zeros(44100, dtype=np.float32),
        )


def test_prepare_speaker_neither_raises(pipeline):
    with pytest.raises(ValueError, match="must be provided"):
        pipeline.prepare_speaker_latents()


def test_prepare_speaker_latents_direct_2d(pipeline):
    mx = pytest.importorskip("mlx.core")
    latents = np.random.default_rng(0).standard_normal((100, 80)).astype(np.float32)
    result_lat, result_mask = pipeline.prepare_speaker_latents(speaker_latents=latents)
    assert result_lat.ndim == 3
    assert result_lat.shape[0] == 1
    assert result_mask.ndim == 2


def test_prepare_speaker_latents_with_mask(pipeline):
    mx = pytest.importorskip("mlx.core")
    latents = np.random.default_rng(0).standard_normal((1, 100, 80)).astype(np.float32)
    mask = np.ones(100, dtype=bool)
    result_lat, result_mask = pipeline.prepare_speaker_latents(
        speaker_latents=latents, speaker_mask=mask
    )
    assert result_lat.ndim == 3


def test_prepare_speaker_latents_too_short_raises(pipeline):
    mx = pytest.importorskip("mlx.core")
    # Only 1 frame — shorter than patch_size (4)
    latents = np.random.default_rng(0).standard_normal((1, 1, 80)).astype(np.float32)
    with pytest.raises(ValueError, match="too short"):
        pipeline.prepare_speaker_latents(speaker_latents=latents)


# ── generate_latents validation ───────────────────────────────────────────


def test_generate_latents_speaker_mask_without_audio_raises(pipeline):
    mx = pytest.importorskip("mlx.core")
    with pytest.raises(ValueError, match="speaker_mask was provided"):
        pipeline.generate_latents(
            text="[S1] Hello.",
            speaker_mask=mx.ones((1, 10), dtype=mx.bool_),
        )


def test_generate_latents_speaker_kv_scale_without_speaker_raises(pipeline):
    mx = pytest.importorskip("mlx.core")
    with pytest.raises(ValueError, match="speaker_kv_scale requires"):
        pipeline.generate_latents(
            text="[S1] Hello.",
            speaker_kv_scale=0.5,
        )


def test_generate_latents_bad_noise_shape_raises(pipeline):
    mx = pytest.importorskip("mlx.core")
    with pytest.raises(ValueError, match="noise must have shape"):
        pipeline.generate_latents(
            text="[S1] Hello.",
            noise=mx.zeros((2, 10, 80), dtype=mx.float32),
        )


# ── _repeat_kv_cache / _scale_kv_cache ────────────────────────────────────


def test_repeat_kv_cache(pipeline):
    mx = pytest.importorskip("mlx.core")
    kv = [(mx.ones((1, 5, 64), dtype=mx.float32), mx.ones((1, 5, 64), dtype=mx.float32))]
    repeated = pipeline._repeat_kv_cache(kv, repeats=3)
    assert len(repeated) == 1
    assert repeated[0][0].shape[0] == 3
    assert repeated[0][1].shape[0] == 3


def test_scale_kv_cache(pipeline):
    mx = pytest.importorskip("mlx.core")
    kv = [
        (mx.ones((1, 5, 64), dtype=mx.float32), mx.ones((1, 5, 64), dtype=mx.float32)),
        (mx.ones((1, 5, 64), dtype=mx.float32), mx.ones((1, 5, 64), dtype=mx.float32)),
    ]
    pipeline._scale_kv_cache(kv, 2.0, max_layers=1)
    mx.eval(kv[0][0])
    np.testing.assert_allclose(np.asarray(kv[0][0]), 2.0)
    np.testing.assert_allclose(np.asarray(kv[1][0]), 1.0)  # Unchanged (max_layers=1)
