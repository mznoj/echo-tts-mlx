"""Tests for echo_tts_mlx.sampler — target ≥92% coverage."""

from __future__ import annotations

import numpy as np
import pytest

from echo_tts_mlx.sampler import (
    BlockwiseSamplerConfig,
    SamplerConfig,
    build_timestep_schedule,
    find_flattening_point,
    sample_blockwise_euler_cfg,
    sample_euler_cfg_independent_guidances,
)


# ── build_timestep_schedule ────────────────────────────────────────────────


def test_schedule_endpoints():
    t = build_timestep_schedule(4, init_scale=1.0)
    assert t.shape == (5,)
    np.testing.assert_allclose(t[0], 1.0)
    np.testing.assert_allclose(t[-1], 0.0)


def test_schedule_init_scale():
    t = build_timestep_schedule(4, init_scale=0.999)
    np.testing.assert_allclose(t[0], 0.999, atol=1e-6)
    np.testing.assert_allclose(t[-1], 0.0)


def test_schedule_invalid_steps():
    with pytest.raises(ValueError, match="num_steps must be >= 1"):
        build_timestep_schedule(0)


def test_schedule_single_step():
    t = build_timestep_schedule(1)
    assert t.shape == (2,)


# ── sample_euler_cfg_independent_guidances ─────────────────────────────────


def test_euler_basic_integration():
    """Verify the sampler runs and applies truncation factor."""
    config = SamplerConfig(
        num_steps=4,
        cfg_scale_text=0.0,
        cfg_scale_speaker=0.0,
        truncation_factor=0.5,
        cfg_min_t=0.0,
        cfg_max_t=0.0,  # CFG never active
    )
    x_t = np.ones((1, 10, 80), dtype=np.float32)
    steps_seen = []

    def predict_velocity(x, t, cfg_active):
        assert not cfg_active
        return np.zeros_like(x)

    def on_step(step, total, t, cfg):
        steps_seen.append(step)

    result = sample_euler_cfg_independent_guidances(
        x_t=x_t,
        config=config,
        predict_velocity=predict_velocity,
        on_step=on_step,
    )
    assert result.shape == (1, 10, 80)
    # truncation_factor=0.5, zero velocity → output should be x_t * 0.5
    np.testing.assert_allclose(result, 0.5, atol=1e-6)
    assert steps_seen == [1, 2, 3, 4]


def test_euler_no_truncation():
    config = SamplerConfig(
        num_steps=2,
        truncation_factor=None,
        cfg_min_t=0.0,
        cfg_max_t=0.0,
    )
    x_t = np.ones((1, 5, 80), dtype=np.float32) * 3.0

    def predict_velocity(x, t, cfg_active):
        return np.zeros_like(x)

    result = sample_euler_cfg_independent_guidances(
        x_t=x_t,
        config=config,
        predict_velocity=predict_velocity,
    )
    # No truncation, zero velocity → output unchanged
    np.testing.assert_allclose(result, 3.0, atol=1e-6)


def test_euler_cfg_active_flag():
    """Verify CFG is active only within the specified time range."""
    config = SamplerConfig(
        num_steps=10,
        cfg_min_t=0.5,
        cfg_max_t=1.0,
        truncation_factor=None,
    )
    cfg_states = []

    def predict_velocity(x, t, cfg_active):
        cfg_states.append((t, cfg_active))
        return np.zeros_like(x)

    sample_euler_cfg_independent_guidances(
        x_t=np.zeros((1, 5, 80), dtype=np.float32),
        config=config,
        predict_velocity=predict_velocity,
    )
    # t starts at 0.999 and decreases
    for t, active in cfg_states:
        if 0.5 <= t <= 1.0:
            assert active, f"CFG should be active at t={t}"
        else:
            assert not active, f"CFG should be inactive at t={t}"


def test_euler_eval_step_called():
    config = SamplerConfig(num_steps=3, truncation_factor=None, cfg_min_t=2.0, cfg_max_t=2.0)
    eval_count = [0]

    def predict_velocity(x, t, cfg_active):
        return np.zeros_like(x)

    def eval_step(x):
        eval_count[0] += 1

    sample_euler_cfg_independent_guidances(
        x_t=np.zeros((1, 5, 80), dtype=np.float32),
        config=config,
        predict_velocity=predict_velocity,
        eval_step=eval_step,
    )
    assert eval_count[0] == 3


def test_euler_speaker_kv_scale_reversal():
    """Verify the on_speaker_kv_scale_reversal callback fires at the right time."""
    config = SamplerConfig(
        num_steps=10,
        truncation_factor=None,
        cfg_min_t=0.0,
        cfg_max_t=0.0,
        speaker_kv_scale=0.5,
        speaker_kv_min_t=0.3,
    )
    reversal_called = [False]
    reversal_t = [None]

    def predict_velocity(x, t, cfg_active):
        return np.zeros_like(x)

    def on_reversal():
        reversal_called[0] = True

    def on_step(step, total, t, cfg):
        if reversal_called[0] and reversal_t[0] is None:
            reversal_t[0] = t

    sample_euler_cfg_independent_guidances(
        x_t=np.zeros((1, 5, 80), dtype=np.float32),
        config=config,
        predict_velocity=predict_velocity,
        on_speaker_kv_scale_reversal=on_reversal,
        on_step=on_step,
    )
    assert reversal_called[0], "speaker_kv_scale reversal should have been called"


def test_euler_no_reversal_without_config():
    """No reversal callback when speaker_kv_scale is not set."""
    config = SamplerConfig(
        num_steps=4,
        truncation_factor=None,
        cfg_min_t=0.0,
        cfg_max_t=0.0,
        speaker_kv_scale=None,
    )
    reversal_called = [False]

    def predict_velocity(x, t, cfg_active):
        return np.zeros_like(x)

    def on_reversal():
        reversal_called[0] = True

    sample_euler_cfg_independent_guidances(
        x_t=np.zeros((1, 5, 80), dtype=np.float32),
        config=config,
        predict_velocity=predict_velocity,
        on_speaker_kv_scale_reversal=on_reversal,
    )
    assert not reversal_called[0]


def test_blockwise_sampler_basic_updates_prefix():
    cfg = BlockwiseSamplerConfig(
        block_sizes=[2, 3],
        num_steps=2,
        truncation_factor=0.5,
        cfg_min_t=2.0,
        cfg_max_t=2.0,  # CFG never active
    )
    prefix = np.zeros((1, 5, 80), dtype=np.float32)
    starts_seen: list[int] = []
    steps_seen: list[int] = []
    build_calls = [0]

    def make_noise(block_size: int):
        return np.ones((1, block_size, 80), dtype=np.float32)

    def build_latent_kv(_prefix):
        build_calls[0] += 1
        return "latent_full", "latent_cond"

    def predict_velocity(x, _t, cfg_active, start_pos, kv_latent_full, kv_latent_cond):
        assert not cfg_active
        assert kv_latent_full == "latent_full"
        assert kv_latent_cond == "latent_cond"
        starts_seen.append(int(start_pos))
        return np.zeros_like(x)

    def on_step(done, _total, _t, _cfg):
        steps_seen.append(int(done))

    out = sample_blockwise_euler_cfg(
        prefix_latent=prefix,
        continuation_length=0,
        config=cfg,
        make_noise=make_noise,
        build_latent_kv=build_latent_kv,
        predict_velocity=predict_velocity,
        on_step=on_step,
    )

    np.testing.assert_allclose(out[:, :2, :], 0.5, atol=1e-6)
    np.testing.assert_allclose(out[:, 2:, :], 0.5, atol=1e-6)
    assert build_calls[0] == 2
    assert starts_seen[:2] == [0, 0]
    assert starts_seen[2:] == [2, 2]
    assert steps_seen == [1, 2, 1, 2]


def test_blockwise_on_step_emits_per_block_steps():
    cfg = BlockwiseSamplerConfig(
        block_sizes=[2, 2],
        num_steps=4,
        truncation_factor=None,
        cfg_min_t=2.0,
        cfg_max_t=2.0,
    )
    seen: list[tuple[int, int]] = []

    def on_step(done, total, _t, _cfg):
        seen.append((int(done), int(total)))

    out = sample_blockwise_euler_cfg(
        prefix_latent=np.zeros((1, 4, 80), dtype=np.float32),
        continuation_length=0,
        config=cfg,
        make_noise=lambda block_size: np.zeros((1, block_size, 80), dtype=np.float32),
        build_latent_kv=lambda _prefix: ("latent_full", "latent_cond"),
        predict_velocity=lambda x, _t, _cfg, _start, _kvf, _kvc: np.zeros_like(x),
        on_step=on_step,
    )

    assert out.shape == (1, 4, 80)
    assert seen == [
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 4),
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 4),
    ]


def test_blockwise_sampler_reversal_called_once_per_block():
    cfg = BlockwiseSamplerConfig(
        block_sizes=[2, 2, 2],
        num_steps=2,
        truncation_factor=None,
        cfg_min_t=2.0,
        cfg_max_t=2.0,
        speaker_kv_scale=1.5,
        speaker_kv_min_t=0.75,
    )
    reversals = [0]

    def on_reversal():
        reversals[0] += 1

    out = sample_blockwise_euler_cfg(
        prefix_latent=np.zeros((1, 6, 80), dtype=np.float32),
        continuation_length=0,
        config=cfg,
        make_noise=lambda block_size: np.zeros((1, block_size, 80), dtype=np.float32),
        build_latent_kv=lambda _prefix: ("latent_full", "latent_cond"),
        predict_velocity=lambda x, _t, _cfg, _start, _kvf, _kvc: np.zeros_like(x),
        on_speaker_kv_scale_reversal=on_reversal,
    )

    assert out.shape == (1, 6, 80)
    assert reversals[0] == 3


# ── find_flattening_point ──────────────────────────────────────────────────


def test_flattening_detects_zero_tail():
    latents = np.zeros((100, 80), dtype=np.float32)
    latents[:50, :] = np.random.default_rng(0).standard_normal((50, 80)).astype(np.float32)
    idx = find_flattening_point(latents)
    # Should detect flattening around frame 50
    assert 40 <= idx <= 60


def test_flattening_no_flat_region():
    rng = np.random.default_rng(42)
    latents = rng.standard_normal((100, 80)).astype(np.float32)
    idx = find_flattening_point(latents)
    assert idx == 100  # No flattening → return full length


def test_flattening_all_zeros():
    latents = np.zeros((50, 80), dtype=np.float32)
    idx = find_flattening_point(latents)
    assert idx == 0


def test_flattening_empty():
    latents = np.zeros((0, 80), dtype=np.float32)
    idx = find_flattening_point(latents)
    assert idx == 0


def test_flattening_wrong_ndim():
    with pytest.raises(ValueError, match="Expected latents shape"):
        find_flattening_point(np.zeros((10,), dtype=np.float32))
