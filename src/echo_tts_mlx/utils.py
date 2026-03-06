"""Runtime utilities for audio I/O and lightweight reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _to_mono(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        # soundfile returns (samples, channels) for multichannel input.
        return x.mean(axis=1, dtype=np.float32)
    raise ValueError(f"Unsupported audio shape: {tuple(x.shape)}")


def load_audio(path: str | Path, *, target_sample_rate: int = 44100) -> tuple[np.ndarray, int]:
    """Load audio from disk, convert to mono float32, and resample if needed."""
    import soundfile as sf

    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    mono = _to_mono(np.asarray(samples, dtype=np.float32))

    if int(sample_rate) != int(target_sample_rate):
        import soxr

        mono = soxr.resample(mono, int(sample_rate), int(target_sample_rate))
        sample_rate = int(target_sample_rate)

    return np.asarray(mono, dtype=np.float32), int(sample_rate)


def flatten_audio_for_write(audio: Any) -> np.ndarray:
    """Normalize various tensor layouts into mono `(samples,)` float32."""
    x = _to_numpy(audio).astype(np.float32, copy=False)

    if x.ndim == 1:
        out = x
    elif x.ndim == 2:
        if x.shape[0] == 1:
            out = x[0]
        elif x.shape[1] == 1:
            out = x[:, 0]
        else:
            out = x.mean(axis=0) if x.shape[0] <= x.shape[1] else x.mean(axis=1)
    elif x.ndim == 3:
        if x.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for audio output, got shape {tuple(x.shape)}")
        if x.shape[1] == 1:
            out = x[0, 0]
        else:
            out = x[0].mean(axis=0)
    else:
        raise ValueError(f"Unsupported audio tensor shape: {tuple(x.shape)}")

    return np.asarray(out, dtype=np.float32)


def save_audio(path: str | Path, audio: Any, *, sample_rate: int = 44100) -> Path:
    """Write mono float32 waveform to disk."""
    import soundfile as sf

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples = flatten_audio_for_write(audio)
    sf.write(str(out_path), samples, int(sample_rate))
    return out_path


def peak_amplitude(audio: Any) -> float:
    x = flatten_audio_for_write(audio)
    return float(np.max(np.abs(x))) if x.size else 0.0


def duration_seconds(audio: Any, *, sample_rate: int) -> float:
    x = flatten_audio_for_write(audio)
    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be > 0")
    return float(x.shape[0] / int(sample_rate))

