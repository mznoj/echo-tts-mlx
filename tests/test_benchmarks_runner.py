from __future__ import annotations

from pathlib import Path
import types

import numpy as np
import pytest

import benchmarks.run_benchmarks as rb
from benchmarks.run_benchmarks import (
    Tier2Config,
    effective_tier1_runs,
    evaluate_quality_gates,
    filter_benchmark_names,
    fit_power_law_exponent,
    make_synthetic_reference_audio,
    summarize_seconds,
)


def test_summarize_seconds_reports_median_and_std_ms() -> None:
    out = summarize_seconds([1.0, 2.0, 3.0])
    assert out["median_ms"] == pytest.approx(2000.0, abs=1e-9)
    assert out["std_ms"] == pytest.approx(816.49658, rel=1e-6)


def test_quality_gate_allows_small_numerical_jitter() -> None:
    sample_rate = 44_100
    t = np.arange(0, sample_rate, dtype=np.float32) / sample_rate
    reference = 0.4 * np.sin(2.0 * np.pi * 220.0 * t)
    candidate = reference + np.float32(1e-6)

    gates = evaluate_quality_gates(
        reference_audio=reference,
        candidate_audio=candidate,
        sample_rate=sample_rate,
        det_atol=1e-5,
        det_rtol=1e-4,
    )
    assert gates["determinism_ok"] is True
    assert gates["non_silence_ok"] is True
    assert gates["duration_ok"] is True
    assert gates["quality_ok"] is True


def test_quality_gate_rejects_short_or_silent_audio() -> None:
    sample_rate = 44_100
    short_silent = np.zeros(sample_rate // 4, dtype=np.float32)
    gates = evaluate_quality_gates(
        reference_audio=short_silent,
        candidate_audio=short_silent,
        sample_rate=sample_rate,
        det_atol=1e-5,
        det_rtol=1e-4,
    )
    assert gates["non_silence_ok"] is False
    assert gates["duration_ok"] is False
    assert gates["quality_ok"] is False


def test_fit_power_law_exponent_recovers_linear_and_quadratic() -> None:
    linear_n = fit_power_law_exponent(x=[8, 16, 32, 64], y=[1.0, 2.0, 4.0, 8.0])
    quad_n = fit_power_law_exponent(x=[8, 16, 32, 64], y=[1.0, 4.0, 16.0, 64.0])
    assert linear_n == pytest.approx(1.0, abs=1e-6)
    assert quad_n == pytest.approx(2.0, abs=1e-6)


def test_filter_benchmark_names_is_case_insensitive() -> None:
    names = ["bench_model_load", "bench_dit_forward_single", "bench_dac_decode"]
    assert filter_benchmark_names(names, "DIT") == ["bench_dit_forward_single"]
    assert filter_benchmark_names(names, "") == names


def test_effective_tier1_runs_caps_model_load_to_three() -> None:
    assert effective_tier1_runs("bench_model_load", 1) == 1
    assert effective_tier1_runs("bench_model_load", 3) == 3
    assert effective_tier1_runs("bench_model_load", 5) == 3
    assert effective_tier1_runs("bench_text_encode", 5) == 5


def test_synthetic_reference_audio_is_deterministic() -> None:
    a = make_synthetic_reference_audio(sample_rate=44_100, duration_s=10.0)
    b = make_synthetic_reference_audio(sample_rate=44_100, duration_s=10.0)
    assert a.dtype == np.float32
    assert np.array_equal(a, b)


class _DummySync:
    def sync(self, _obj) -> None:
        return None

    def reset_peak(self) -> None:
        return None

    def get_memory_metrics(self) -> dict[str, float]:
        return {}


class _DummyMx:
    float32 = np.float32
    bool_ = np.bool_

    @staticmethod
    def concatenate(items, axis=0):
        return np.concatenate(items, axis=axis)

    @staticmethod
    def zeros_like(x):
        return np.zeros_like(x)

    @staticmethod
    def eval(*_items):
        return None


class _DummyModel:
    def __init__(self, *, num_layers: int = 2) -> None:
        self.num_layers = num_layers
        self.has_blockwise_modules = False

    def _encode_text(self, text_ids, text_mask):
        hidden = np.zeros((int(text_ids.shape[0]), int(text_ids.shape[1]), 8), dtype=np.float32)
        return hidden, text_mask

    def _encode_speaker(self, speaker_latents, speaker_mask):
        hidden = np.zeros((int(speaker_latents.shape[0]), int(speaker_latents.shape[1]), 8), dtype=np.float32)
        return hidden, speaker_mask

    def get_kv_cache_text(self, text_ids, text_mask):
        kv = [(np.zeros((1, 1, 1, 1), dtype=np.float32), np.zeros((1, 1, 1, 1), dtype=np.float32))] * self.num_layers
        return kv, text_mask

    def get_kv_cache_speaker(self, speaker_latents, speaker_mask):
        kv = [(np.zeros((1, 1, 1, 1), dtype=np.float32), np.zeros((1, 1, 1, 1), dtype=np.float32))] * self.num_layers
        return kv, speaker_mask

    def _project_kv(self, _hidden, **_kwargs):
        return np.zeros((1, 1, 1, 1), dtype=np.float32), np.zeros((1, 1, 1, 1), dtype=np.float32)

    def t(self, _key: str):
        return np.ones((1,), dtype=np.float32)

    def forward(self, latents, *_args, **_kwargs):
        return np.asarray(latents, dtype=np.float32)


class _DummyAutoencoder:
    def encode_zq(self, _audio):
        return np.zeros((1, 1024, 8), dtype=np.float32)

    def decode_zq(self, _zq):
        return np.zeros((1, 1, 1024), dtype=np.float32)


class _DummyPipeline:
    def __init__(self) -> None:
        self.model = _DummyModel(num_layers=2)
        self.autoencoder = _DummyAutoencoder()

    @staticmethod
    def _repeat_kv_cache(kv, repeats: int):
        return kv * int(repeats)

    @staticmethod
    def pca_encode(z_q):
        return np.asarray(z_q, dtype=np.float32)

    @staticmethod
    def pca_decode(z):
        return np.asarray(z, dtype=np.float32)


def _make_dummy_runtime(*, blockwise_capable: bool):
    cfg = types.SimpleNamespace(
        text_vocab_size=256,
        sample_rate=44_100,
        ae_downsample_factor=512,
        max_speaker_latent_length=512,
        speaker_patch_size=4,
        latent_size=80,
        num_layers=2,
        pca_latent_dim=1024,
    )
    pipeline = _DummyPipeline()
    pipeline.model.has_blockwise_modules = blockwise_capable
    return rb.BenchmarkRuntime(
        mx=_DummyMx(),
        sync=_DummySync(),
        pipeline=pipeline,
        config=cfg,
        weights_dir=Path("."),
        dtype="float16",
        quantize="none",
        blockwise_capable=blockwise_capable,
    )


def test_tier1_blockwise_benches_emit_explicit_skip_without_blockwise_weights() -> None:
    runtime = _make_dummy_runtime(blockwise_capable=False)
    out = rb._benchmark_tier1(runtime, warmup=0, runs=1, name_filter="latent")
    assert out["bench_latent_encode"] == {"skipped": "no blockwise weights"}
    assert out["bench_kv_cache_latent"] == {"skipped": "no blockwise weights"}


def test_tier2_blockwise_registry_and_skip_when_unavailable() -> None:
    runtime = _make_dummy_runtime(blockwise_capable=False)
    cfg = Tier2Config(
        runs=1,
        warmup=0,
        cooldown_s=0.0,
        sequence_length=64,
        num_steps=2,
        seed=0,
        do_quality_checks=False,
        det_atol=1e-5,
        det_rtol=1e-4,
        cfg_scale_text=3.0,
        cfg_scale_speaker=8.0,
        truncation_factor=0.8,
    )
    out = rb._run_tier2(runtime, cfg, name_filter="bench_blockwise")
    assert set(out) == {
        "bench_blockwise_breakdown",
        "bench_blockwise_vs_standard",
        "bench_blockwise_scale_blocks",
        "bench_blockwise_scale_first_block",
        "bench_blockwise_continuation",
        "bench_blockwise_standard_regression",
    }
    assert all(v == {"skipped": "no blockwise weights"} for v in out.values())


def test_tier2_blockwise_registry_calls_impls_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _make_dummy_runtime(blockwise_capable=True)
    cfg = Tier2Config(
        runs=1,
        warmup=0,
        cooldown_s=0.0,
        sequence_length=64,
        num_steps=2,
        seed=0,
        do_quality_checks=False,
        det_atol=1e-5,
        det_rtol=1e-4,
        cfg_scale_text=3.0,
        cfg_scale_speaker=8.0,
        truncation_factor=0.8,
    )

    monkeypatch.setattr(rb, "_run_blockwise_breakdown", lambda *_args, **_kwargs: {"name": "breakdown"})
    monkeypatch.setattr(rb, "_run_blockwise_vs_standard", lambda *_args, **_kwargs: {"name": "vs_standard"})
    monkeypatch.setattr(rb, "_run_blockwise_scale_blocks", lambda *_args, **_kwargs: {"name": "scale_blocks"})
    monkeypatch.setattr(rb, "_run_blockwise_scale_first_block", lambda *_args, **_kwargs: {"name": "scale_first"})
    monkeypatch.setattr(rb, "_run_blockwise_continuation", lambda *_args, **_kwargs: {"name": "continuation"})
    monkeypatch.setattr(
        rb,
        "_run_blockwise_standard_regression",
        lambda *_args, **_kwargs: {"name": "standard_regression"},
    )

    out = rb._run_tier2(runtime, cfg, name_filter="bench_blockwise", weights_standard=Path("/tmp/converted"))
    assert out["bench_blockwise_breakdown"]["name"] == "breakdown"
    assert out["bench_blockwise_vs_standard"]["name"] == "vs_standard"
    assert out["bench_blockwise_scale_blocks"]["name"] == "scale_blocks"
    assert out["bench_blockwise_scale_first_block"]["name"] == "scale_first"
    assert out["bench_blockwise_continuation"]["name"] == "continuation"
    assert out["bench_blockwise_standard_regression"]["name"] == "standard_regression"


def test_resolve_standard_weights_dir_prefers_override(tmp_path: Path) -> None:
    override = tmp_path / "explicit-standard"
    override.mkdir()
    resolved, error = rb._resolve_standard_weights_dir(
        blockwise_weights_dir=tmp_path / "converted-blockwise",
        weights_standard=override,
    )
    assert resolved == override
    assert error is None
