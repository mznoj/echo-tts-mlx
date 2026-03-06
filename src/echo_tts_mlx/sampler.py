"""Diffusion sampling utilities for Echo-TTS."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib

import numpy as np


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for Euler sampling with independent dual CFG."""

    num_steps: int = 32
    cfg_scale_text: float = 3.0
    cfg_scale_speaker: float = 8.0
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    truncation_factor: float | None = 0.8
    init_scale: float = 0.999
    speaker_kv_scale: float | None = None
    speaker_kv_max_layers: int | None = None
    speaker_kv_min_t: float | None = None


@dataclass(frozen=True)
class BlockwiseSamplerConfig:
    """Configuration for blockwise Euler sampling."""

    block_sizes: list[int]
    num_steps: int = 32
    cfg_scale_text: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    truncation_factor: float | None = 0.8
    init_scale: float = 0.999
    speaker_kv_scale: float | None = None
    speaker_kv_max_layers: int | None = None
    speaker_kv_min_t: float | None = None


def build_timestep_schedule(num_steps: int, *, init_scale: float = 0.999) -> np.ndarray:
    """Return sampling schedule `(num_steps + 1,)` from 1.0 to 0.0, scaled by `init_scale`."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    t = np.linspace(1.0, 0.0, num_steps + 1, dtype=np.float32)
    return t * np.float32(init_scale)


def sample_euler_cfg_independent_guidances(
    *,
    x_t,
    config: SamplerConfig,
    predict_velocity: Callable[[object, float, bool], object],
    eval_step: Callable[[object], None] | None = None,
    on_speaker_kv_scale_reversal: Callable[[], None] | None = None,
    on_step: Callable[[int, int, float, bool], None] | None = None,
):
    """Run Euler integration over a diffusion trajectory.

    `predict_velocity(x_t, t, cfg_active)` must return velocity tensor `v_pred`
    with the same shape/dtype semantics as `x_t`.
    """
    if config.truncation_factor is not None:
        x_t = x_t * float(config.truncation_factor)

    t_schedule = build_timestep_schedule(config.num_steps, init_scale=config.init_scale)
    scale_reversed = False

    for i in range(config.num_steps):
        t = float(t_schedule[i])
        t_next = float(t_schedule[i + 1])

        cfg_active = config.cfg_min_t <= t <= config.cfg_max_t

        if (
            not scale_reversed
            and config.speaker_kv_scale is not None
            and config.speaker_kv_min_t is not None
            and t_next < config.speaker_kv_min_t <= t
        ):
            if on_speaker_kv_scale_reversal is not None:
                on_speaker_kv_scale_reversal()
            scale_reversed = True

        v_pred = predict_velocity(x_t, t, cfg_active)
        x_t = x_t + v_pred * (t_next - t)

        if eval_step is not None:
            eval_step(x_t)
        if on_step is not None:
            on_step(i + 1, config.num_steps, t, cfg_active)

    return x_t


def sample_blockwise_euler_cfg(
    *,
    prefix_latent,
    continuation_length: int,
    config: BlockwiseSamplerConfig,
    make_noise: Callable[[int], object],
    build_latent_kv: Callable[[object], tuple[object, object]],
    predict_velocity: Callable[[object, float, bool, int, object, object], object],
    eval_step: Callable[[object], None] | None = None,
    on_block_start: Callable[[int, int, int], None] | None = None,
    on_speaker_kv_scale_reversal: Callable[[], None] | None = None,
    on_step: Callable[[int, int, float, bool], None] | None = None,
    on_block_complete: Callable[[int, int, object], None] | None = None,
):
    """Run blockwise Euler integration over sequential latent blocks."""
    if not config.block_sizes:
        raise ValueError("block_sizes must not be empty.")
    if any(int(size) <= 0 for size in config.block_sizes):
        raise ValueError("All block sizes must be > 0.")

    total_blocks = len(config.block_sizes)
    start_pos = int(continuation_length)
    t_schedule = build_timestep_schedule(config.num_steps, init_scale=config.init_scale)

    for block_idx, block_size in enumerate(config.block_sizes):
        if on_block_start is not None:
            on_block_start(block_idx, total_blocks, start_pos)

        kv_latent_full, kv_latent_cond = build_latent_kv(prefix_latent)
        x_t = make_noise(int(block_size))
        if config.truncation_factor is not None:
            x_t = x_t * float(config.truncation_factor)

        scale_reversed = False
        for i in range(config.num_steps):
            t = float(t_schedule[i])
            t_next = float(t_schedule[i + 1])
            cfg_active = config.cfg_min_t <= t <= config.cfg_max_t

            if (
                not scale_reversed
                and config.speaker_kv_scale is not None
                and config.speaker_kv_min_t is not None
                and t_next < config.speaker_kv_min_t <= t
            ):
                if on_speaker_kv_scale_reversal is not None:
                    on_speaker_kv_scale_reversal()
                scale_reversed = True

            v_pred = predict_velocity(x_t, t, cfg_active, start_pos, kv_latent_full, kv_latent_cond)
            x_t = x_t + v_pred * (t_next - t)

            if eval_step is not None:
                eval_step(x_t)
            if on_step is not None:
                on_step(i + 1, config.num_steps, t, cfg_active)

        block_start = start_pos
        block_end = block_start + int(block_size)
        prefix_latent[:, block_start:block_end, :] = x_t
        if on_block_complete is not None:
            on_block_complete(block_idx, total_blocks, x_t)
        start_pos = block_end

    return prefix_latent


def find_flattening_point(
    latents: np.ndarray,
    *,
    window_size: int = 20,
    std_threshold: float = 0.05,
    mean_threshold: float = 0.1,
) -> int:
    """Find the first frame where latents flatten to near-zero statistics."""
    latents = np.asarray(latents, dtype=np.float32)
    if latents.ndim != 2:
        raise ValueError(f"Expected latents shape (T, 80), got {tuple(latents.shape)}")
    if latents.shape[0] == 0:
        return 0

    padded = np.concatenate(
        [latents, np.zeros((window_size, latents.shape[1]), dtype=np.float32)],
        axis=0,
    )
    for i in range(0, padded.shape[0] - window_size + 1):
        window = padded[i : i + window_size]
        if float(window.std()) < std_threshold and abs(float(window.mean())) < mean_threshold:
            return i
    return int(latents.shape[0])


def _energy_rms_windows(audio: np.ndarray, *, energy_hop_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if energy_hop_samples <= 0:
        raise ValueError(f"energy_hop_samples must be > 0, got {energy_hop_samples}")

    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    starts = np.arange(0, x.size, energy_hop_samples, dtype=np.int32)
    rms = np.zeros((starts.shape[0],), dtype=np.float32)
    for i, start in enumerate(starts):
        window = x[start : start + energy_hop_samples]
        if window.size == 0:
            continue
        rms[i] = float(np.sqrt(np.mean(window * window)))
    return rms, starts


def _find_energy_drop_point(
    *,
    audio: np.ndarray,
    ae_downsample_factor: int,
    energy_threshold_db: float,
    energy_hop_samples: int,
) -> int | None:
    rms, starts = _energy_rms_windows(audio, energy_hop_samples=energy_hop_samples)
    if rms.size == 0:
        return None

    peak_rms = float(np.max(rms))
    if peak_rms <= 0.0:
        return 0

    threshold_linear = peak_rms * (10.0 ** (float(energy_threshold_db) / 20.0))
    below = np.nonzero(rms <= threshold_linear)[0]
    if below.size == 0:
        return None
    start_sample = int(starts[int(below[0])])
    return max(0, int(start_sample // ae_downsample_factor))


def _analyze_f0_variance(
    *,
    audio: np.ndarray,
    sample_rate: int,
    ae_downsample_factor: int,
    f0_fmin: float,
    f0_fmax: float,
    f0_variance_window_s: float,
    f0_variance_hop_s: float,
    f0_variance_ratio_threshold: float,
    min_voiced_ratio: float,
) -> dict[str, float | int | None | np.ndarray]:
    librosa = importlib.import_module("librosa")

    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return {
            "f0_hz": np.zeros((0,), dtype=np.float32),
            "f0_times_s": np.zeros((0,), dtype=np.float32),
            "window_centers_s": np.zeros((0,), dtype=np.float32),
            "window_variances": np.zeros((0,), dtype=np.float32),
            "tail_to_body_ratio": float("nan"),
            "onset_window_time_s": None,
            "onset_latent_frame": None,
        }

    hop_length = 256
    f0_hz, _, _ = librosa.pyin(
        x,
        fmin=float(f0_fmin),
        fmax=float(f0_fmax),
        sr=int(sample_rate),
        hop_length=hop_length,
    )
    f0_hz = np.asarray(f0_hz, dtype=np.float32)
    f0_times = (np.arange(f0_hz.shape[0], dtype=np.float32) * float(hop_length)) / float(sample_rate)

    window_frames = max(1, int(round(float(f0_variance_window_s) * float(sample_rate) / float(hop_length))))
    hop_frames = max(1, int(round(float(f0_variance_hop_s) * float(sample_rate) / float(hop_length))))

    centers: list[float] = []
    variances: list[float] = []
    for start in range(0, max(f0_hz.shape[0] - window_frames + 1, 0), hop_frames):
        end = start + window_frames
        window = f0_hz[start:end]
        voiced = np.isfinite(window)
        voiced_ratio = float(np.mean(voiced)) if voiced.size > 0 else 0.0
        if voiced_ratio < float(min_voiced_ratio):
            continue
        voiced_f0 = window[voiced]
        if voiced_f0.size == 0:
            continue
        centers.append(float((start + window_frames // 2) * hop_length / sample_rate))
        variances.append(float(np.var(voiced_f0)))

    centers_arr = np.asarray(centers, dtype=np.float32)
    variances_arr = np.asarray(variances, dtype=np.float32)
    if variances_arr.size == 0:
        return {
            "f0_hz": f0_hz,
            "f0_times_s": f0_times,
            "window_centers_s": centers_arr,
            "window_variances": variances_arr,
            "tail_to_body_ratio": float("nan"),
            "onset_window_time_s": None,
            "onset_latent_frame": None,
        }

    duration_s = float(x.size) / float(sample_rate)
    tail_start_s = max(0.0, duration_s - float(f0_variance_window_s))
    tail_mask = centers_arr >= tail_start_s
    body_mask = ~tail_mask
    body_vals = variances_arr[body_mask]
    if body_vals.size == 0:
        body_vals = variances_arr
    body_median = float(np.median(body_vals)) if body_vals.size > 0 else float("nan")

    tail_vals = variances_arr[tail_mask]
    if tail_vals.size == 0:
        tail_vals = variances_arr[-1:]
    tail_var = float(np.median(tail_vals)) if tail_vals.size > 0 else float("nan")

    if body_median > 0.0 and np.isfinite(body_median) and np.isfinite(tail_var):
        tail_ratio = tail_var / body_median
    else:
        tail_ratio = float("nan")

    onset_time: float | None = None
    onset_latent_frame: int | None = None
    if body_median > 0.0 and np.isfinite(body_median):
        exceed = np.nonzero(variances_arr > (float(f0_variance_ratio_threshold) * body_median))[0]
        if exceed.size > 0:
            onset_time = float(centers_arr[int(exceed[0])])
            onset_latent_frame = int(max(0.0, onset_time * float(sample_rate) / float(ae_downsample_factor)))

    return {
        "f0_hz": f0_hz,
        "f0_times_s": f0_times,
        "window_centers_s": centers_arr,
        "window_variances": variances_arr,
        "tail_to_body_ratio": tail_ratio,
        "onset_window_time_s": onset_time,
        "onset_latent_frame": onset_latent_frame,
    }


def analyze_tail_pitch(
    *,
    audio: np.ndarray,
    sample_rate: int = 44100,
    ae_downsample_factor: int = 2048,
    f0_fmin: float = 50.0,
    f0_fmax: float = 500.0,
    f0_variance_window_s: float = 2.0,
    f0_variance_hop_s: float = 0.5,
    f0_variance_ratio_threshold: float = 2.0,
    min_voiced_ratio: float = 0.3,
) -> dict[str, float | int | None | np.ndarray]:
    """Compute F0 variance diagnostics for tail-pitch instability analysis."""
    try:
        return _analyze_f0_variance(
            audio=audio,
            sample_rate=sample_rate,
            ae_downsample_factor=ae_downsample_factor,
            f0_fmin=f0_fmin,
            f0_fmax=f0_fmax,
            f0_variance_window_s=f0_variance_window_s,
            f0_variance_hop_s=f0_variance_hop_s,
            f0_variance_ratio_threshold=f0_variance_ratio_threshold,
            min_voiced_ratio=min_voiced_ratio,
        )
    except ImportError as exc:
        raise ImportError("librosa is required for F0 analysis. Install with `pip install librosa`.") from exc


def find_content_boundary(
    latents: np.ndarray,
    audio: np.ndarray,
    sample_rate: int = 44100,
    ae_downsample_factor: int = 2048,
    *,
    latent_window: int = 20,
    latent_std_threshold: float = 0.05,
    latent_mean_threshold: float = 0.1,
    energy_enabled: bool = False,
    energy_threshold_db: float = -40.0,
    energy_hop_samples: int = 2048,
    f0_enabled: bool = False,
    f0_fmin: float = 50.0,
    f0_fmax: float = 500.0,
    f0_variance_window_s: float = 2.0,
    f0_variance_hop_s: float = 0.5,
    f0_variance_ratio_threshold: float = 2.0,
    min_voiced_ratio: float = 0.3,
    min_retained_ratio: float = 0.5,
) -> int:
    """Find a trim boundary from latent, audio-energy, and optional F0 instability signals."""
    lat = np.asarray(latents, dtype=np.float32)
    if lat.ndim != 2:
        raise ValueError(f"Expected latents shape (T, 80), got {tuple(lat.shape)}")

    latent_boundary = find_flattening_point(
        lat,
        window_size=latent_window,
        std_threshold=latent_std_threshold,
        mean_threshold=latent_mean_threshold,
    )
    candidates = [int(latent_boundary)]

    if energy_enabled:
        energy_boundary = _find_energy_drop_point(
            audio=np.asarray(audio, dtype=np.float32).reshape(-1),
            ae_downsample_factor=ae_downsample_factor,
            energy_threshold_db=energy_threshold_db,
            energy_hop_samples=energy_hop_samples,
        )
        if energy_boundary is not None:
            candidates.append(int(energy_boundary))

    if f0_enabled:
        try:
            f0_info = _analyze_f0_variance(
                audio=np.asarray(audio, dtype=np.float32).reshape(-1),
                sample_rate=sample_rate,
                ae_downsample_factor=ae_downsample_factor,
                f0_fmin=f0_fmin,
                f0_fmax=f0_fmax,
                f0_variance_window_s=f0_variance_window_s,
                f0_variance_hop_s=f0_variance_hop_s,
                f0_variance_ratio_threshold=f0_variance_ratio_threshold,
                min_voiced_ratio=min_voiced_ratio,
            )
        except ImportError as exc:
            raise ImportError("librosa is required for trim_mode='f0'. Install with `pip install librosa`.") from exc
        onset_latent_frame = f0_info.get("onset_latent_frame")
        if isinstance(onset_latent_frame, (int, np.integer)):
            candidates.append(int(onset_latent_frame))

    boundary = int(min(candidates))
    boundary = max(0, min(boundary, int(lat.shape[0])))

    if boundary < int(latent_boundary * float(min_retained_ratio)):
        return int(latent_boundary)
    return boundary
