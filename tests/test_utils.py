"""Tests for echo_tts_mlx.utils — target ≥92% coverage."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from echo_tts_mlx.utils import (
    _to_mono,
    _to_numpy,
    duration_seconds,
    flatten_audio_for_write,
    load_audio,
    peak_amplitude,
    save_audio,
)


# ── _to_numpy ──────────────────────────────────────────────────────────────


def test_to_numpy_from_ndarray():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = _to_numpy(arr)
    np.testing.assert_array_equal(result, arr)


def test_to_numpy_from_list():
    result = _to_numpy([1.0, 2.0])
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1.0, 2.0])


# ── _to_mono ───────────────────────────────────────────────────────────────


def test_to_mono_already_mono():
    mono = np.array([0.5, -0.5, 0.1], dtype=np.float32)
    result = _to_mono(mono)
    np.testing.assert_array_equal(result, mono)


def test_to_mono_stereo():
    stereo = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = _to_mono(stereo)
    assert result.ndim == 1
    assert result.shape == (2,)
    np.testing.assert_allclose(result, [0.5, 0.5])


def test_to_mono_unsupported_ndim():
    with pytest.raises(ValueError, match="Unsupported audio shape"):
        _to_mono(np.zeros((2, 3, 4), dtype=np.float32))


# ── flatten_audio_for_write ────────────────────────────────────────────────


def test_flatten_1d():
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    result = flatten_audio_for_write(x)
    np.testing.assert_array_equal(result, x)


def test_flatten_2d_batch_1():
    x = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)  # (1, 3)
    result = flatten_audio_for_write(x)
    assert result.shape == (3,)
    np.testing.assert_allclose(result, [0.1, 0.2, 0.3], atol=1e-7)


def test_flatten_2d_column():
    x = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)  # (3, 1)
    result = flatten_audio_for_write(x)
    assert result.shape == (3,)


def test_flatten_2d_mean_rows():
    # shape (2, 4) → rows <= cols → mean(axis=0) → (4,)
    x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    result = flatten_audio_for_write(x)
    assert result.shape == (4,)
    np.testing.assert_allclose(result, [3.0, 4.0, 5.0, 6.0])


def test_flatten_2d_mean_cols():
    # shape (4, 2) → rows > cols → mean(axis=1) → (4,)
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    result = flatten_audio_for_write(x)
    assert result.shape == (4,)
    np.testing.assert_allclose(result, [1.5, 3.5, 5.5, 7.5])


def test_flatten_3d_batch_1_mono():
    x = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)  # (1, 1, 3)
    result = flatten_audio_for_write(x)
    assert result.shape == (3,)


def test_flatten_3d_batch_1_multichannel():
    x = np.zeros((1, 2, 100), dtype=np.float32)
    x[0, 0, :] = 1.0
    x[0, 1, :] = -1.0
    result = flatten_audio_for_write(x)
    assert result.shape == (100,)
    np.testing.assert_allclose(result, 0.0)


def test_flatten_3d_batch_gt_1_raises():
    with pytest.raises(ValueError, match="batch size 1"):
        flatten_audio_for_write(np.zeros((2, 1, 10), dtype=np.float32))


def test_flatten_4d_raises():
    with pytest.raises(ValueError, match="Unsupported audio tensor shape"):
        flatten_audio_for_write(np.zeros((1, 1, 1, 10), dtype=np.float32))


# ── save_audio / load_audio roundtrip ──────────────────────────────────────


def test_save_and_load_roundtrip():
    rng = np.random.default_rng(42)
    original = rng.uniform(-0.5, 0.5, size=(44100,)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "subdir" / "test.wav"
        out_path = save_audio(path, original, sample_rate=44100)
        assert out_path.exists()

        loaded, sr = load_audio(out_path, target_sample_rate=44100)
        assert sr == 44100
        np.testing.assert_allclose(loaded, original, atol=1e-4)


def test_load_audio_resamples():
    rng = np.random.default_rng(123)
    original = rng.uniform(-0.3, 0.3, size=(22050,)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test_22k.wav"
        import soundfile as sf

        sf.write(str(path), original, 22050)

        loaded, sr = load_audio(path, target_sample_rate=44100)
        assert sr == 44100
        # Resampled should be roughly double the length
        assert abs(loaded.shape[0] - 44100) < 100


# ── peak_amplitude ─────────────────────────────────────────────────────────


def test_peak_amplitude_basic():
    x = np.array([0.1, -0.5, 0.3], dtype=np.float32)
    assert abs(peak_amplitude(x) - 0.5) < 1e-6


def test_peak_amplitude_empty():
    x = np.array([], dtype=np.float32)
    assert peak_amplitude(x) == 0.0


# ── duration_seconds ───────────────────────────────────────────────────────


def test_duration_seconds_basic():
    x = np.zeros(44100, dtype=np.float32)
    assert abs(duration_seconds(x, sample_rate=44100) - 1.0) < 1e-6


def test_duration_seconds_invalid_sr():
    with pytest.raises(ValueError, match="sample_rate must be > 0"):
        duration_seconds(np.zeros(100, dtype=np.float32), sample_rate=0)
