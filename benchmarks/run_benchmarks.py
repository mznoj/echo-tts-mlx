from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path("benchmark_results.json")
DEFAULT_TIER1_RUNS = 5
DEFAULT_TIER2_RUNS = 3
DEFAULT_WARMUP = 2
DEFAULT_COOLDOWN_S = 3.0
DEFAULT_STEPS = 32
DEFAULT_SEQ_LENGTH = 200
DEFAULT_REFERENCE_CACHE = Path("benchmarks/cache")
DEFAULT_REFERENCE_TIMEOUT = 20.0
TIER2_DIFFUSION_TIMING_NOTE = (
    "Tier 2 diffusion timings use a direct model+sampler benchmark path "
    "(_diffusion_only) instead of end-to-end pipeline.generate()."
)
SKIP_NO_BLOCKWISE = {"skipped": "no blockwise weights"}


def summarize_seconds(samples_s: Sequence[float]) -> dict[str, float]:
    values = np.asarray(list(samples_s), dtype=np.float64)
    if values.size == 0:
        raise ValueError("samples_s must contain at least one value")
    return {
        "median_ms": float(np.median(values) * 1000.0),
        "std_ms": float(np.std(values) * 1000.0),
    }


def filter_benchmark_names(names: Sequence[str], pattern: str) -> list[str]:
    if pattern.strip() == "":
        return list(names)
    needle = pattern.lower()
    return [name for name in names if needle in name.lower()]


def effective_tier1_runs(bench_name: str, requested_runs: int) -> int:
    if requested_runs < 1:
        raise ValueError("requested_runs must be >= 1")
    if bench_name == "bench_model_load":
        return min(requested_runs, 3)
    return requested_runs


def fit_power_law_exponent(*, x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")

    mask = (x_arr > 0.0) & (y_arr > 0.0)
    if int(mask.sum()) < 2:
        return float("nan")

    coeffs = np.polyfit(np.log(x_arr[mask]), np.log(y_arr[mask]), deg=1)
    return float(coeffs[0])


def _flatten_audio(audio: Any) -> np.ndarray:
    x = audio
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 1:
        return x
    if x.ndim == 2:
        if x.shape[0] == 1:
            return x[0]
        if x.shape[1] == 1:
            return x[:, 0]
        if x.shape[0] <= x.shape[1]:
            return x.mean(axis=0, dtype=np.float32)
        return x.mean(axis=1, dtype=np.float32)
    if x.ndim == 3:
        if x.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 audio, got shape {tuple(x.shape)}")
        if x.shape[1] == 1:
            return x[0, 0]
        return x[0].mean(axis=0, dtype=np.float32)

    raise ValueError(f"Unsupported audio tensor shape: {tuple(x.shape)}")


def evaluate_quality_gates(
    *,
    reference_audio: Any,
    candidate_audio: Any,
    sample_rate: int,
    det_atol: float,
    det_rtol: float,
) -> dict[str, bool]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")

    reference = _flatten_audio(reference_audio)
    candidate = _flatten_audio(candidate_audio)

    same_shape = reference.shape == candidate.shape
    determinism_ok = bool(same_shape and np.allclose(reference, candidate, atol=det_atol, rtol=det_rtol))
    non_silence_ok = bool(np.max(np.abs(candidate)) > 0.01) if candidate.size > 0 else False
    duration_ok = bool((candidate.shape[0] / float(sample_rate)) > 0.5)

    return {
        "determinism_ok": determinism_ok,
        "non_silence_ok": non_silence_ok,
        "duration_ok": duration_ok,
        "quality_ok": bool(determinism_ok and non_silence_ok and duration_ok),
    }


def make_synthetic_reference_audio(*, sample_rate: int, duration_s: float) -> np.ndarray:
    n = int(round(sample_rate * duration_s))
    t = np.arange(n, dtype=np.float32) / np.float32(sample_rate)
    signal = 0.5 * np.sin(2.0 * np.pi * 220.0 * t) + 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
    return signal.astype(np.float32)


def _try_import_mlx() -> Any:
    try:
        import mlx.core as mx

        return mx
    except Exception as exc:  # pragma: no cover - runtime-only guard
        raise RuntimeError("MLX runtime is required for benchmark execution") from exc


def _mlx_runtime_available() -> bool:
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import mlx.core as mx; _ = mx.array([0], dtype=mx.float16)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return False
    return proc.returncode == 0


def _is_mlx_array(x: Any) -> bool:
    return x.__class__.__module__.startswith("mlx")


def _collect_mlx_arrays(obj: Any, out: list[Any]) -> None:
    if _is_mlx_array(obj):
        out.append(obj)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _collect_mlx_arrays(value, out)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            _collect_mlx_arrays(value, out)


@dataclass
class SyncAdapter:
    mx: Any

    def sync(self, obj: Any | None) -> None:
        arrays: list[Any] = []
        if obj is not None:
            _collect_mlx_arrays(obj, arrays)
        if arrays:
            self.mx.eval(*arrays)

    def get_memory_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        get_peak = getattr(self.mx, "get_peak_memory", None)
        get_active = getattr(self.mx, "get_active_memory", None)
        if not callable(get_peak) or not callable(get_active):
            metal = getattr(self.mx, "metal", None)
            if metal is not None:
                if not callable(get_peak):
                    get_peak = getattr(metal, "get_peak_memory", None)
                if not callable(get_active):
                    get_active = getattr(metal, "get_active_memory", None)
        if callable(get_peak):
            try:
                metrics["peak_memory_mb"] = float(get_peak() / 1e6)
            except Exception:
                pass
        if callable(get_active):
            try:
                metrics["active_memory_mb"] = float(get_active() / 1e6)
            except Exception:
                pass
        return metrics

    def reset_peak(self) -> None:
        fn = getattr(self.mx, "reset_peak_memory", None)
        if not callable(fn):
            metal = getattr(self.mx, "metal", None)
            if metal is not None:
                fn = getattr(metal, "reset_peak_memory", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


@dataclass
class BenchmarkRuntime:
    mx: Any
    sync: SyncAdapter
    pipeline: Any
    config: Any
    weights_dir: Path
    dtype: str
    quantize: str = "none"
    blockwise_capable: bool = False


@dataclass
class Tier2Config:
    runs: int
    warmup: int
    cooldown_s: float
    sequence_length: int
    num_steps: int
    seed: int
    do_quality_checks: bool
    det_atol: float
    det_rtol: float
    cfg_scale_text: float
    cfg_scale_speaker: float
    truncation_factor: float


def _make_token_ids(*, num_tokens: int, vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
    n = max(1, num_tokens)
    ids = (np.arange(n, dtype=np.int32) % max(1, vocab_size)).astype(np.int32)
    ids[0] = 0
    mask = np.ones((1, n), dtype=np.bool_)
    return ids[None, :], mask


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_latents(shape: tuple[int, ...], *, seed: int) -> np.ndarray:
    return _rng(seed).standard_normal(shape).astype(np.float32)


def _speaker_frames_for_seconds(*, seconds: float, sample_rate: int, downsample_factor: int, max_frames: int, patch_size: int) -> int:
    raw = int(seconds * sample_rate) // downsample_factor
    raw = min(raw, max_frames)
    raw = max(raw, patch_size)
    return raw


def _make_speaker_latents(config: Any, *, seconds: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    t = _speaker_frames_for_seconds(
        seconds=seconds,
        sample_rate=int(config.sample_rate),
        downsample_factor=int(config.ae_downsample_factor),
        max_frames=int(config.max_speaker_latent_length),
        patch_size=int(config.speaker_patch_size),
    )
    latents = _make_latents((1, t, int(config.latent_size)), seed=seed)
    mask = np.ones((1, t), dtype=np.bool_)
    return latents, mask


def _benchmark_measure(
    *,
    fn: Callable[[], Any],
    sync: SyncAdapter,
    warmup: int,
    runs: int,
) -> tuple[list[float], list[Any]]:
    if runs < 1:
        raise ValueError("runs must be >= 1")

    for _ in range(max(0, warmup)):
        out = fn()
        sync.sync(out)

    times: list[float] = []
    outs: list[Any] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn()
        sync.sync(out)
        times.append(time.perf_counter() - t0)
        outs.append(out)
    return times, outs


def _load_runtime(*, weights_dir: Path, dtype: str, quantize: str = "none") -> BenchmarkRuntime:
    from echo_tts_mlx.pipeline import EchoTTS

    mx = _try_import_mlx()
    pipeline = EchoTTS.from_pretrained(weights_dir, dtype=dtype, quantize=quantize)
    sync = SyncAdapter(mx=mx)
    return BenchmarkRuntime(
        mx=mx,
        sync=sync,
        pipeline=pipeline,
        config=pipeline.config,
        weights_dir=weights_dir,
        dtype=dtype,
        quantize=quantize,
        blockwise_capable=bool(getattr(pipeline.model, "has_blockwise_modules", False)),
    )


def _repeat_kv(pipeline: Any, kv: list[tuple[Any, Any]], repeats: int) -> list[tuple[Any, Any]]:
    return pipeline._repeat_kv_cache(kv, repeats=repeats)


def _diffusion_only(
    runtime: BenchmarkRuntime,
    *,
    x_t: Any,
    num_steps: int,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    truncation_factor: float | None,
    kv_text_cond: list[tuple[Any, Any]],
    text_k_mask: Any,
    kv_speaker_cond: list[tuple[Any, Any]],
    speaker_k_mask: Any,
) -> tuple[Any, float]:
    from echo_tts_mlx.sampler import SamplerConfig, sample_euler_cfg_independent_guidances

    mx = runtime.mx
    pipeline = runtime.pipeline
    model = pipeline.model

    sampler_cfg = SamplerConfig(
        num_steps=num_steps,
        cfg_scale_text=cfg_scale_text,
        cfg_scale_speaker=cfg_scale_speaker,
        cfg_min_t=0.5,
        cfg_max_t=1.0,
        truncation_factor=truncation_factor,
    )

    kv_text_full = _repeat_kv(pipeline, kv_text_cond, repeats=3)
    kv_speaker_full = _repeat_kv(pipeline, kv_speaker_cond, repeats=3)

    text_mask_batch = mx.concatenate([text_k_mask, mx.zeros_like(text_k_mask), text_k_mask], axis=0)
    speaker_mask_batch = mx.concatenate([speaker_k_mask, speaker_k_mask, mx.zeros_like(speaker_k_mask)], axis=0)

    def _predict_velocity(x_cur: Any, t: float, cfg_active: bool) -> Any:
        if cfg_active:
            x_batch = mx.concatenate([x_cur, x_cur, x_cur], axis=0)
            t_batch = mx.array([t, t, t], dtype=x_cur.dtype)
            out = model.forward(
                x_batch,
                t_batch,
                kv_text=kv_text_full,
                kv_speaker=kv_speaker_full,
                text_mask=text_mask_batch,
                speaker_mask=speaker_mask_batch,
            )
            v_cond, v_uncond_text, v_uncond_speaker = mx.split(out, 3, axis=0)
            return (
                v_cond
                + sampler_cfg.cfg_scale_text * (v_cond - v_uncond_text)
                + sampler_cfg.cfg_scale_speaker * (v_cond - v_uncond_speaker)
            )

        t_single = mx.array([t], dtype=x_cur.dtype)
        return model.forward(
            x_cur,
            t_single,
            kv_text=kv_text_cond,
            kv_speaker=kv_speaker_cond,
            text_mask=text_k_mask,
            speaker_mask=speaker_k_mask,
        )

    first_step_s = float("nan")
    t0 = time.perf_counter()

    def _on_step(done: int, _total: int, _t: float, _cfg_active: bool) -> None:
        nonlocal first_step_s
        if done == 1 and math.isnan(first_step_s):
            first_step_s = time.perf_counter() - t0

    latents = sample_euler_cfg_independent_guidances(
        x_t=x_t,
        config=sampler_cfg,
        predict_velocity=_predict_velocity,
        eval_step=lambda arr: mx.eval(arr),
        on_step=_on_step,
    )
    mx.eval(latents)
    if math.isnan(first_step_s):
        first_step_s = time.perf_counter() - t0
    return latents, float(first_step_s)


def _benchmark_tier1(runtime: BenchmarkRuntime, *, warmup: int, runs: int, name_filter: str) -> dict[str, Any]:
    mx = runtime.mx
    model = runtime.pipeline.model
    autoencoder = runtime.pipeline.autoencoder
    config = runtime.config
    pipeline = runtime.pipeline

    text_ids, text_mask = _make_token_ids(num_tokens=50, vocab_size=int(config.text_vocab_size))
    speaker_latents, speaker_mask = _make_speaker_latents(config, seconds=10.0, seed=17)

    text_hidden, _ = model._encode_text(text_ids, text_mask)
    speaker_hidden, _ = model._encode_speaker(speaker_latents, speaker_mask)
    kv_text_cond, text_k_mask = model.get_kv_cache_text(text_ids, text_mask)
    kv_speaker_cond, speaker_k_mask = model.get_kv_cache_speaker(speaker_latents, speaker_mask)
    mx.eval(text_hidden, speaker_hidden, text_k_mask, speaker_k_mask)

    latents_single = _make_latents((1, 200, int(config.latent_size)), seed=10)
    latents_cfg = _make_latents((3, 200, int(config.latent_size)), seed=11)
    timestep_single = np.array([0.5], dtype=np.float32)
    timestep_cfg = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    timestep_cfg_blockwise = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    kv_text_full = _repeat_kv(pipeline, kv_text_cond, repeats=3)
    kv_speaker_full = _repeat_kv(pipeline, kv_speaker_cond, repeats=3)
    text_mask_batch = mx.concatenate([text_k_mask, mx.zeros_like(text_k_mask), text_k_mask], axis=0)
    speaker_mask_batch = mx.concatenate([speaker_k_mask, speaker_k_mask, mx.zeros_like(speaker_k_mask)], axis=0)

    z_q_input = _make_latents((1, int(config.pca_latent_dim), 200), seed=12)
    z_input = _make_latents((1, 200, int(config.latent_size)), seed=13)
    audio_10s = make_synthetic_reference_audio(sample_rate=int(config.sample_rate), duration_s=10.0).reshape(1, 1, -1)
    z_q_decode = _make_latents((1, int(config.pca_latent_dim), 200), seed=14)

    prefix_latent: np.ndarray | None = None
    latents_cfg_blockwise: np.ndarray | None = None
    kv_latent_cond: list[tuple[Any, Any]] | None = None
    kv_latent_full: list[tuple[Any, Any]] | None = None
    if runtime.blockwise_capable:
        prefix_latent = _make_latents((1, 128, int(config.latent_size)), seed=20)
        latents_cfg_blockwise = _make_latents((3, 128, int(config.latent_size)), seed=21)
        kv_latent_cond = model.get_kv_cache_latent(prefix_latent)
        kv_latent_full = _repeat_kv(pipeline, kv_latent_cond, repeats=3)
        runtime.sync.sync((kv_latent_cond, kv_latent_full))

    def _bench_model_load() -> Any:
        from echo_tts_mlx.pipeline import EchoTTS

        fresh = EchoTTS.from_pretrained(runtime.weights_dir, dtype=runtime.dtype, quantize=runtime.quantize)
        tensor = fresh.model.t("in_proj.weight")
        mx.eval(tensor)
        return tensor

    def _bench_text_encode() -> Any:
        return model._encode_text(text_ids, text_mask)

    def _bench_speaker_encode() -> Any:
        return model._encode_speaker(speaker_latents, speaker_mask)

    def _bench_kv_cache_text() -> Any:
        out: list[tuple[Any, Any]] = []
        for i in range(config.num_layers):
            out.append(
                model._project_kv(
                    text_hidden,
                    k_weight=f"blocks.{i}.attention.wk_text",
                    v_weight=f"blocks.{i}.attention.wv_text",
                    k_norm_weight=model.t(f"blocks.{i}.attention.k_norm.weight"),
                )
            )
        return out

    def _bench_kv_cache_speaker() -> Any:
        out: list[tuple[Any, Any]] = []
        for i in range(config.num_layers):
            out.append(
                model._project_kv(
                    speaker_hidden,
                    k_weight=f"blocks.{i}.attention.wk_speaker",
                    v_weight=f"blocks.{i}.attention.wv_speaker",
                    k_norm_weight=model.t(f"blocks.{i}.attention.k_norm.weight"),
                )
            )
        return out

    def _bench_latent_encode() -> Any:
        if prefix_latent is None:
            raise RuntimeError("bench_latent_encode requires blockwise-capable weights")
        return model._encode_latent(prefix_latent)

    def _bench_kv_cache_latent() -> Any:
        if prefix_latent is None:
            raise RuntimeError("bench_kv_cache_latent requires blockwise-capable weights")
        return model.get_kv_cache_latent(prefix_latent)

    def _bench_dit_forward_single() -> Any:
        return model.forward(
            latents_single,
            timestep_single,
            kv_text=kv_text_cond,
            kv_speaker=kv_speaker_cond,
            text_mask=text_k_mask,
            speaker_mask=speaker_k_mask,
        )

    def _bench_dit_forward_cfg() -> Any:
        return model.forward(
            latents_cfg,
            timestep_cfg,
            kv_text=kv_text_full,
            kv_speaker=kv_speaker_full,
            text_mask=text_mask_batch,
            speaker_mask=speaker_mask_batch,
        )

    def _bench_dit_forward_cfg_blockwise() -> Any:
        if latents_cfg_blockwise is None or kv_latent_full is None:
            raise RuntimeError("bench_dit_forward_cfg_blockwise requires blockwise-capable weights")
        return model.forward(
            latents_cfg_blockwise,
            timestep_cfg_blockwise,
            kv_text=kv_text_full,
            kv_speaker=kv_speaker_full,
            text_mask=text_mask_batch,
            speaker_mask=speaker_mask_batch,
            start_pos=128,
            kv_latent=kv_latent_full,
        )

    def _bench_pca_encode() -> Any:
        return runtime.pipeline.pca_encode(z_q_input)

    def _bench_pca_decode() -> Any:
        return runtime.pipeline.pca_decode(z_input)

    def _bench_dac_encode() -> Any:
        return autoencoder.encode_zq(audio_10s)

    def _bench_dac_decode() -> Any:
        return autoencoder.decode_zq(z_q_decode)

    benches: OrderedDict[str, Callable[[], Any]] = OrderedDict(
        [
            ("bench_model_load", _bench_model_load),
            ("bench_text_encode", _bench_text_encode),
            ("bench_speaker_encode", _bench_speaker_encode),
            ("bench_kv_cache_text", _bench_kv_cache_text),
            ("bench_kv_cache_speaker", _bench_kv_cache_speaker),
            ("bench_latent_encode", _bench_latent_encode),
            ("bench_kv_cache_latent", _bench_kv_cache_latent),
            ("bench_dit_forward_single", _bench_dit_forward_single),
            ("bench_dit_forward_cfg", _bench_dit_forward_cfg),
            ("bench_dit_forward_cfg_blockwise", _bench_dit_forward_cfg_blockwise),
            ("bench_pca_encode", _bench_pca_encode),
            ("bench_pca_decode", _bench_pca_decode),
            ("bench_dac_encode", _bench_dac_encode),
            ("bench_dac_decode", _bench_dac_decode),
        ]
    )

    blockwise_tier1 = {
        "bench_latent_encode",
        "bench_kv_cache_latent",
        "bench_dit_forward_cfg_blockwise",
    }
    selected = filter_benchmark_names(list(benches.keys()), name_filter)
    out: dict[str, Any] = {}
    for name in selected:
        if name in blockwise_tier1 and not runtime.blockwise_capable:
            out[name] = dict(SKIP_NO_BLOCKWISE)
            continue
        fn = benches[name]
        bench_warmup = 0 if name == "bench_model_load" else warmup
        bench_runs = effective_tier1_runs(name, runs)
        times_s, _ = _benchmark_measure(fn=fn, sync=runtime.sync, warmup=bench_warmup, runs=bench_runs)
        stats = summarize_seconds(times_s)
        out[name] = {
            "median_ms": stats["median_ms"],
            "std_ms": stats["std_ms"],
            "runs": bench_runs,
            "warmup": bench_warmup,
        }
    return out


def _prepare_conditioning(
    runtime: BenchmarkRuntime,
    *,
    text_ids: np.ndarray,
    text_mask: np.ndarray,
    speaker_audio: np.ndarray | None,
) -> tuple[list[tuple[Any, Any]], Any, list[tuple[Any, Any]], Any, float, float]:
    mx = runtime.mx
    pipeline = runtime.pipeline

    t0 = time.perf_counter()
    kv_text_cond, text_k_mask = pipeline.model.get_kv_cache_text(text_ids, text_mask)
    runtime.sync.sync((kv_text_cond, text_k_mask))
    text_encode_s = time.perf_counter() - t0

    if speaker_audio is None:
        speaker_latents = mx.zeros((1, 4, int(runtime.config.latent_size)), dtype=mx.float32)
        speaker_mask = mx.zeros((1, 4), dtype=mx.bool_)
        t0 = time.perf_counter()
        kv_speaker_cond, speaker_k_mask = pipeline.model.get_kv_cache_speaker(speaker_latents, speaker_mask)
        runtime.sync.sync((kv_speaker_cond, speaker_k_mask))
        speaker_encode_s = 0.0
        return kv_text_cond, text_k_mask, kv_speaker_cond, speaker_k_mask, text_encode_s, speaker_encode_s

    t0 = time.perf_counter()
    speaker_latents, speaker_mask = pipeline.prepare_speaker_latents(speaker_audio=speaker_audio)
    kv_speaker_cond, speaker_k_mask = pipeline.model.get_kv_cache_speaker(speaker_latents, speaker_mask)
    runtime.sync.sync((speaker_latents, speaker_mask, kv_speaker_cond, speaker_k_mask))
    speaker_encode_s = time.perf_counter() - t0
    return kv_text_cond, text_k_mask, kv_speaker_cond, speaker_k_mask, text_encode_s, speaker_encode_s


def _breakdown_once(runtime: BenchmarkRuntime, cfg: Tier2Config, *, use_speaker: bool) -> tuple[dict[str, float], np.ndarray]:
    mx = runtime.mx
    config = runtime.config
    text_ids, text_mask = _make_token_ids(num_tokens=50, vocab_size=int(config.text_vocab_size))

    speaker_audio = None
    if use_speaker:
        speaker_audio = make_synthetic_reference_audio(sample_rate=int(config.sample_rate), duration_s=10.0)

    kv_text_cond, text_k_mask, kv_speaker_cond, speaker_k_mask, text_encode_s, speaker_encode_s = _prepare_conditioning(
        runtime,
        text_ids=text_ids,
        text_mask=text_mask,
        speaker_audio=speaker_audio,
    )

    mx.random.seed(cfg.seed)
    x_t = mx.random.normal((1, cfg.sequence_length, int(config.latent_size)), dtype=mx.float32)

    t0 = time.perf_counter()
    latents, first_step_s = _diffusion_only(
        runtime,
        x_t=x_t,
        num_steps=cfg.num_steps,
        cfg_scale_text=cfg.cfg_scale_text,
        cfg_scale_speaker=cfg.cfg_scale_speaker,
        truncation_factor=cfg.truncation_factor,
        kv_text_cond=kv_text_cond,
        text_k_mask=text_k_mask,
        kv_speaker_cond=kv_speaker_cond,
        speaker_k_mask=speaker_k_mask,
    )
    diffusion_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    audio = runtime.pipeline.decode_latents(latents)
    runtime.sync.sync(audio)
    decode_s = time.perf_counter() - t0

    audio_np = _flatten_audio(audio)
    audio_duration_s = float(audio_np.shape[0] / int(config.sample_rate))
    wall_time_s = float(text_encode_s + speaker_encode_s + diffusion_s + decode_s)

    return (
        {
            "wall_time_s": wall_time_s,
            "model_load_s": 0.0,
            "speaker_encode_s": float(speaker_encode_s),
            "text_encode_s": float(text_encode_s),
            "ttfb_s": float(text_encode_s + speaker_encode_s + first_step_s),
            "diffusion_s": float(diffusion_s),
            "decode_s": float(decode_s),
            "audio_duration_s": audio_duration_s,
        },
        audio_np,
    )


def _median_metric(metrics: list[dict[str, float]], key: str) -> float:
    return float(np.median([row[key] for row in metrics]))


def _quality_from_runs(
    *,
    audios: Sequence[np.ndarray],
    sample_rate: int,
    det_atol: float,
    det_rtol: float,
) -> tuple[bool, dict[str, bool]]:
    if not audios:
        return False, {"determinism_ok": False, "non_silence_ok": False, "duration_ok": False}

    ref = audios[0]
    determinism_ok = True
    non_silence_ok = True
    duration_ok = True
    for cand in audios:
        gates = evaluate_quality_gates(
            reference_audio=ref,
            candidate_audio=cand,
            sample_rate=sample_rate,
            det_atol=det_atol,
            det_rtol=det_rtol,
        )
        determinism_ok = determinism_ok and gates["determinism_ok"]
        non_silence_ok = non_silence_ok and gates["non_silence_ok"]
        duration_ok = duration_ok and gates["duration_ok"]
    return bool(determinism_ok and non_silence_ok and duration_ok), {
        "determinism_ok": determinism_ok,
        "non_silence_ok": non_silence_ok,
        "duration_ok": duration_ok,
    }


def _run_breakdown_benchmark(
    runtime: BenchmarkRuntime,
    cfg: Tier2Config,
    *,
    use_speaker: bool,
) -> dict[str, Any]:
    for _ in range(cfg.warmup):
        _breakdown_once(runtime, cfg, use_speaker=use_speaker)

    metrics: list[dict[str, float]] = []
    audios: list[np.ndarray] = []
    for _ in range(cfg.runs):
        row, audio = _breakdown_once(runtime, cfg, use_speaker=use_speaker)
        metrics.append(row)
        audios.append(audio)

    result = {
        "wall_time_s": _median_metric(metrics, "wall_time_s"),
        "model_load_s": 0.0,
        "speaker_encode_s": _median_metric(metrics, "speaker_encode_s"),
        "text_encode_s": _median_metric(metrics, "text_encode_s"),
        "ttfb_s": _median_metric(metrics, "ttfb_s"),
        "diffusion_s": _median_metric(metrics, "diffusion_s"),
        "decode_s": _median_metric(metrics, "decode_s"),
        "audio_duration_s": _median_metric(metrics, "audio_duration_s"),
    }
    result["latent_steps_per_sec"] = float(cfg.num_steps / max(result["diffusion_s"], 1e-12))
    result["realtime_factor"] = float(result["audio_duration_s"] / max(result["wall_time_s"], 1e-12))

    if cfg.do_quality_checks:
        quality_ok, checks = _quality_from_runs(
            audios=audios,
            sample_rate=int(runtime.config.sample_rate),
            det_atol=cfg.det_atol,
            det_rtol=cfg.det_rtol,
        )
        result["quality_ok"] = bool(quality_ok)
        result["status"] = "PASS" if quality_ok else "FAIL"
        result.update(checks)
    else:
        result["quality_ok"] = True
        result["status"] = "PASS"

    result.update(runtime.sync.get_memory_metrics())
    return result


def _bench_ttfb(runtime: BenchmarkRuntime, cfg: Tier2Config) -> dict[str, float]:
    # The helper above intentionally computes all setup each run.
    # Capture TTFB directly from setup + first diffusion step.
    values: list[float] = []

    for _ in range(cfg.warmup):
        _ = _ttfb_measurement(runtime, cfg)

    for _ in range(cfg.runs):
        values.append(_ttfb_measurement(runtime, cfg))

    return {"ttfb_s": float(np.median(values))}


def _ttfb_measurement(runtime: BenchmarkRuntime, cfg: Tier2Config) -> float:
    config = runtime.config
    text_ids, text_mask = _make_token_ids(num_tokens=50, vocab_size=int(config.text_vocab_size))
    speaker_audio = make_synthetic_reference_audio(sample_rate=int(config.sample_rate), duration_s=10.0)

    t_setup = time.perf_counter()
    kv_text_cond, text_k_mask, kv_speaker_cond, speaker_k_mask, _, _ = _prepare_conditioning(
        runtime,
        text_ids=text_ids,
        text_mask=text_mask,
        speaker_audio=speaker_audio,
    )
    setup_s = time.perf_counter() - t_setup

    runtime.mx.random.seed(cfg.seed)
    x_t = runtime.mx.random.normal((1, cfg.sequence_length, int(config.latent_size)), dtype=runtime.mx.float32)
    _, first_step_s = _diffusion_only(
        runtime,
        x_t=x_t,
        num_steps=cfg.num_steps,
        cfg_scale_text=cfg.cfg_scale_text,
        cfg_scale_speaker=cfg.cfg_scale_speaker,
        truncation_factor=cfg.truncation_factor,
        kv_text_cond=kv_text_cond,
        text_k_mask=text_k_mask,
        kv_speaker_cond=kv_speaker_cond,
        speaker_k_mask=speaker_k_mask,
    )
    return float(setup_s + first_step_s)


def _median_seconds(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(list(values), dtype=np.float64)))


def _run_scale_steps(runtime: BenchmarkRuntime, cfg: Tier2Config) -> dict[str, Any]:
    steps_values = [8, 16, 32, 64]
    config = runtime.config

    text_ids, text_mask = _make_token_ids(num_tokens=50, vocab_size=int(config.text_vocab_size))
    speaker_audio = make_synthetic_reference_audio(sample_rate=int(config.sample_rate), duration_s=10.0)
    kv_text_cond, text_k_mask, kv_speaker_cond, speaker_k_mask, _, _ = _prepare_conditioning(
        runtime,
        text_ids=text_ids,
        text_mask=text_mask,
        speaker_audio=speaker_audio,
    )

    points: list[dict[str, float]] = []
    for steps in steps_values:
        for _ in range(cfg.warmup):
            runtime.mx.random.seed(cfg.seed)
            x_t = runtime.mx.random.normal((1, cfg.sequence_length, int(config.latent_size)), dtype=runtime.mx.float32)
            _diffusion_only(
                runtime,
                x_t=x_t,
                num_steps=steps,
                cfg_scale_text=cfg.cfg_scale_text,
                cfg_scale_speaker=cfg.cfg_scale_speaker,
                truncation_factor=cfg.truncation_factor,
                kv_text_cond=kv_text_cond,
                text_k_mask=text_k_mask,
                kv_speaker_cond=kv_speaker_cond,
                speaker_k_mask=speaker_k_mask,
            )

        values: list[float] = []
        for _ in range(cfg.runs):
            runtime.mx.random.seed(cfg.seed)
            x_t = runtime.mx.random.normal((1, cfg.sequence_length, int(config.latent_size)), dtype=runtime.mx.float32)
            t0 = time.perf_counter()
            _diffusion_only(
                runtime,
                x_t=x_t,
                num_steps=steps,
                cfg_scale_text=cfg.cfg_scale_text,
                cfg_scale_speaker=cfg.cfg_scale_speaker,
                truncation_factor=cfg.truncation_factor,
                kv_text_cond=kv_text_cond,
                text_k_mask=text_k_mask,
                kv_speaker_cond=kv_speaker_cond,
                speaker_k_mask=speaker_k_mask,
            )
            values.append(time.perf_counter() - t0)

        points.append({"steps": float(steps), "diffusion_s": _median_seconds(values)})

    exponent = fit_power_law_exponent(x=[p["steps"] for p in points], y=[p["diffusion_s"] for p in points])
    return {
        "points": [{"steps": int(p["steps"]), "diffusion_s": p["diffusion_s"]} for p in points],
        "scaling_exponent": exponent,
    }


def _run_scale_seq_length(runtime: BenchmarkRuntime, cfg: Tier2Config) -> dict[str, Any]:
    seq_values = [100, 200, 400, 640]
    config = runtime.config

    text_ids, text_mask = _make_token_ids(num_tokens=50, vocab_size=int(config.text_vocab_size))
    speaker_audio = make_synthetic_reference_audio(sample_rate=int(config.sample_rate), duration_s=10.0)
    kv_text_cond, text_k_mask, kv_speaker_cond, speaker_k_mask, _, _ = _prepare_conditioning(
        runtime,
        text_ids=text_ids,
        text_mask=text_mask,
        speaker_audio=speaker_audio,
    )

    points: list[dict[str, float]] = []
    for seq_length in seq_values:
        for _ in range(cfg.warmup):
            runtime.mx.random.seed(cfg.seed)
            x_t = runtime.mx.random.normal((1, seq_length, int(config.latent_size)), dtype=runtime.mx.float32)
            _diffusion_only(
                runtime,
                x_t=x_t,
                num_steps=cfg.num_steps,
                cfg_scale_text=cfg.cfg_scale_text,
                cfg_scale_speaker=cfg.cfg_scale_speaker,
                truncation_factor=cfg.truncation_factor,
                kv_text_cond=kv_text_cond,
                text_k_mask=text_k_mask,
                kv_speaker_cond=kv_speaker_cond,
                speaker_k_mask=speaker_k_mask,
            )

        values: list[float] = []
        for _ in range(cfg.runs):
            runtime.mx.random.seed(cfg.seed)
            x_t = runtime.mx.random.normal((1, seq_length, int(config.latent_size)), dtype=runtime.mx.float32)
            t0 = time.perf_counter()
            _diffusion_only(
                runtime,
                x_t=x_t,
                num_steps=cfg.num_steps,
                cfg_scale_text=cfg.cfg_scale_text,
                cfg_scale_speaker=cfg.cfg_scale_speaker,
                truncation_factor=cfg.truncation_factor,
                kv_text_cond=kv_text_cond,
                text_k_mask=text_k_mask,
                kv_speaker_cond=kv_speaker_cond,
                speaker_k_mask=speaker_k_mask,
            )
            values.append(time.perf_counter() - t0)

        points.append({"seq_length": float(seq_length), "diffusion_s": _median_seconds(values)})

    exponent = fit_power_law_exponent(x=[p["seq_length"] for p in points], y=[p["diffusion_s"] for p in points])
    return {
        "points": [{"seq_length": int(p["seq_length"]), "diffusion_s": p["diffusion_s"]} for p in points],
        "scaling_exponent": exponent,
    }


def _run_scale_speaker_length(runtime: BenchmarkRuntime, cfg: Tier2Config) -> dict[str, Any]:
    speaker_seconds = [2.0, 5.0, 10.0, 30.0]
    config = runtime.config

    text_ids, text_mask = _make_token_ids(num_tokens=50, vocab_size=int(config.text_vocab_size))
    kv_text_cond, text_k_mask, _, _, _, _ = _prepare_conditioning(
        runtime,
        text_ids=text_ids,
        text_mask=text_mask,
        speaker_audio=None,
    )

    points: list[dict[str, float]] = []
    for seconds in speaker_seconds:
        warm_audio = make_synthetic_reference_audio(sample_rate=int(config.sample_rate), duration_s=seconds)
        for _ in range(cfg.warmup):
            t0 = time.perf_counter()
            speaker_latents, speaker_mask = runtime.pipeline.prepare_speaker_latents(speaker_audio=warm_audio)
            kv_speaker_cond, speaker_k_mask = runtime.pipeline.model.get_kv_cache_speaker(speaker_latents, speaker_mask)
            runtime.sync.sync((kv_speaker_cond, speaker_k_mask))
            _ = time.perf_counter() - t0

            runtime.mx.random.seed(cfg.seed)
            x_t = runtime.mx.random.normal((1, cfg.sequence_length, int(config.latent_size)), dtype=runtime.mx.float32)
            _diffusion_only(
                runtime,
                x_t=x_t,
                num_steps=cfg.num_steps,
                cfg_scale_text=cfg.cfg_scale_text,
                cfg_scale_speaker=cfg.cfg_scale_speaker,
                truncation_factor=cfg.truncation_factor,
                kv_text_cond=kv_text_cond,
                text_k_mask=text_k_mask,
                kv_speaker_cond=kv_speaker_cond,
                speaker_k_mask=speaker_k_mask,
            )

        speaker_times: list[float] = []
        diffusion_times: list[float] = []
        for _ in range(cfg.runs):
            audio = make_synthetic_reference_audio(sample_rate=int(config.sample_rate), duration_s=seconds)
            t0 = time.perf_counter()
            speaker_latents, speaker_mask = runtime.pipeline.prepare_speaker_latents(speaker_audio=audio)
            kv_speaker_cond, speaker_k_mask = runtime.pipeline.model.get_kv_cache_speaker(speaker_latents, speaker_mask)
            runtime.sync.sync((speaker_latents, speaker_mask, kv_speaker_cond, speaker_k_mask))
            speaker_times.append(time.perf_counter() - t0)

            runtime.mx.random.seed(cfg.seed)
            x_t = runtime.mx.random.normal((1, cfg.sequence_length, int(config.latent_size)), dtype=runtime.mx.float32)
            t0 = time.perf_counter()
            _diffusion_only(
                runtime,
                x_t=x_t,
                num_steps=cfg.num_steps,
                cfg_scale_text=cfg.cfg_scale_text,
                cfg_scale_speaker=cfg.cfg_scale_speaker,
                truncation_factor=cfg.truncation_factor,
                kv_text_cond=kv_text_cond,
                text_k_mask=text_k_mask,
                kv_speaker_cond=kv_speaker_cond,
                speaker_k_mask=speaker_k_mask,
            )
            diffusion_times.append(time.perf_counter() - t0)

        points.append(
            {
                "speaker_seconds": float(seconds),
                "speaker_encode_s": _median_seconds(speaker_times),
                "diffusion_s": _median_seconds(diffusion_times),
            }
        )

    return {
        "points": [
            {
                "speaker_seconds": int(p["speaker_seconds"]) if float(p["speaker_seconds"]).is_integer() else p["speaker_seconds"],
                "speaker_encode_s": p["speaker_encode_s"],
                "diffusion_s": p["diffusion_s"],
            }
            for p in points
        ]
    }


def _run_scale_text_length(runtime: BenchmarkRuntime, cfg: Tier2Config) -> dict[str, Any]:
    token_lengths = [10, 50, 200, 500]
    config = runtime.config

    speaker_audio = make_synthetic_reference_audio(sample_rate=int(config.sample_rate), duration_s=10.0)
    _, _, kv_speaker_cond, speaker_k_mask, _, _ = _prepare_conditioning(
        runtime,
        text_ids=_make_token_ids(num_tokens=50, vocab_size=int(config.text_vocab_size))[0],
        text_mask=_make_token_ids(num_tokens=50, vocab_size=int(config.text_vocab_size))[1],
        speaker_audio=speaker_audio,
    )

    points: list[dict[str, float]] = []
    for token_len in token_lengths:
        text_ids, text_mask = _make_token_ids(
            num_tokens=min(token_len, int(config.max_text_length)),
            vocab_size=int(config.text_vocab_size),
        )

        for _ in range(cfg.warmup):
            kv_text_cond, text_k_mask = runtime.pipeline.model.get_kv_cache_text(text_ids, text_mask)
            runtime.sync.sync((kv_text_cond, text_k_mask))
            runtime.mx.random.seed(cfg.seed)
            x_t = runtime.mx.random.normal((1, cfg.sequence_length, int(config.latent_size)), dtype=runtime.mx.float32)
            _diffusion_only(
                runtime,
                x_t=x_t,
                num_steps=cfg.num_steps,
                cfg_scale_text=cfg.cfg_scale_text,
                cfg_scale_speaker=cfg.cfg_scale_speaker,
                truncation_factor=cfg.truncation_factor,
                kv_text_cond=kv_text_cond,
                text_k_mask=text_k_mask,
                kv_speaker_cond=kv_speaker_cond,
                speaker_k_mask=speaker_k_mask,
            )

        text_times: list[float] = []
        diffusion_times: list[float] = []
        for _ in range(cfg.runs):
            t0 = time.perf_counter()
            kv_text_cond, text_k_mask = runtime.pipeline.model.get_kv_cache_text(text_ids, text_mask)
            runtime.sync.sync((kv_text_cond, text_k_mask))
            text_times.append(time.perf_counter() - t0)

            runtime.mx.random.seed(cfg.seed)
            x_t = runtime.mx.random.normal((1, cfg.sequence_length, int(config.latent_size)), dtype=runtime.mx.float32)
            t0 = time.perf_counter()
            _diffusion_only(
                runtime,
                x_t=x_t,
                num_steps=cfg.num_steps,
                cfg_scale_text=cfg.cfg_scale_text,
                cfg_scale_speaker=cfg.cfg_scale_speaker,
                truncation_factor=cfg.truncation_factor,
                kv_text_cond=kv_text_cond,
                text_k_mask=text_k_mask,
                kv_speaker_cond=kv_speaker_cond,
                speaker_k_mask=speaker_k_mask,
            )
            diffusion_times.append(time.perf_counter() - t0)

        points.append(
            {
                "text_tokens": int(token_len),
                "text_encode_s": _median_seconds(text_times),
                "diffusion_s": _median_seconds(diffusion_times),
            }
        )

    return {"points": points}


def _median_list(rows: Sequence[Sequence[float]]) -> list[float]:
    if not rows:
        return []
    max_len = max(len(row) for row in rows)
    out: list[float] = []
    for i in range(max_len):
        values = [float(row[i]) for row in rows if i < len(row) and not math.isnan(float(row[i]))]
        out.append(_median_seconds(values) if values else float("nan"))
    return out


def _make_continuation_audio_for_frames(config: Any, *, frames: int) -> np.ndarray:
    """Generate synthetic audio (sine-wave reference) for continuation benchmarks.

    This keeps continuation timing stable across runs. Real speech can trigger
    different DAC encode behavior (content-dependent codebook activation), so
    production continuation timings may vary slightly from these synthetic runs.
    """
    if frames <= 0:
        return np.zeros((0,), dtype=np.float32)
    total_samples = int(frames) * int(config.ae_downsample_factor)
    duration_s = max(float(total_samples) / float(config.sample_rate), 0.01)
    audio = make_synthetic_reference_audio(sample_rate=int(config.sample_rate), duration_s=duration_s)
    if audio.shape[0] >= total_samples:
        return audio[:total_samples]
    padded = np.zeros((total_samples,), dtype=np.float32)
    padded[: audio.shape[0]] = audio
    return padded


def _measure_standard_generate_once(
    runtime: BenchmarkRuntime,
    cfg: Tier2Config,
    *,
    sequence_length: int,
    speaker_audio: np.ndarray | None,
    text: str = "[S1] Blockwise benchmark standard baseline.",
) -> tuple[dict[str, float], np.ndarray]:
    t0 = time.perf_counter()
    audio = runtime.pipeline.generate(
        text=text,
        speaker_audio=speaker_audio,
        sequence_length=int(sequence_length),
        seed=cfg.seed,
        num_steps=cfg.num_steps,
        cfg_scale_text=cfg.cfg_scale_text,
        cfg_scale_speaker=cfg.cfg_scale_speaker,
        truncation_factor=cfg.truncation_factor,
        trim_latents=True,
    )
    runtime.sync.sync(audio)
    wall_time_s = time.perf_counter() - t0
    audio_np = _flatten_audio(audio)
    audio_duration_s = float(audio_np.shape[0] / float(runtime.config.sample_rate))
    return (
        {
            "wall_time_s": float(wall_time_s),
            "ttfb_s": float(wall_time_s),
            "audio_duration_s": float(audio_duration_s),
            "realtime_factor": float(audio_duration_s / max(wall_time_s, 1e-12)),
        },
        audio_np,
    )


def _run_standard_generate_median(
    runtime: BenchmarkRuntime,
    cfg: Tier2Config,
    *,
    sequence_length: int,
    speaker_audio: np.ndarray | None,
) -> dict[str, Any]:
    for _ in range(cfg.warmup):
        _measure_standard_generate_once(runtime, cfg, sequence_length=sequence_length, speaker_audio=speaker_audio)

    rows: list[dict[str, float]] = []
    for _ in range(cfg.runs):
        runtime.sync.reset_peak()
        row, _ = _measure_standard_generate_once(runtime, cfg, sequence_length=sequence_length, speaker_audio=speaker_audio)
        row.update(runtime.sync.get_memory_metrics())
        rows.append(row)

    out: dict[str, Any] = {
        "wall_time_s": _median_seconds([row["wall_time_s"] for row in rows]),
        "ttfb_s": _median_seconds([row["ttfb_s"] for row in rows]),
        "audio_duration_s": _median_seconds([row["audio_duration_s"] for row in rows]),
        "realtime_factor": _median_seconds([row["realtime_factor"] for row in rows]),
    }
    for key in ("peak_memory_mb", "active_memory_mb"):
        values = [float(row[key]) for row in rows if key in row]
        if values:
            out[key] = _median_seconds(values)
    return out


def _run_standard_component_breakdown(
    runtime: BenchmarkRuntime,
    cfg: Tier2Config,
    *,
    sequence_length: int,
) -> dict[str, float]:
    cfg_local = Tier2Config(
        runs=cfg.runs,
        warmup=cfg.warmup,
        cooldown_s=cfg.cooldown_s,
        sequence_length=int(sequence_length),
        num_steps=cfg.num_steps,
        seed=cfg.seed,
        do_quality_checks=False,
        det_atol=cfg.det_atol,
        det_rtol=cfg.det_rtol,
        cfg_scale_text=cfg.cfg_scale_text,
        cfg_scale_speaker=cfg.cfg_scale_speaker,
        truncation_factor=cfg.truncation_factor,
    )

    for _ in range(cfg_local.warmup):
        _breakdown_once(runtime, cfg_local, use_speaker=True)

    rows: list[dict[str, float]] = []
    for _ in range(cfg_local.runs):
        row, _ = _breakdown_once(runtime, cfg_local, use_speaker=True)
        rows.append(row)

    return {
        "conditioning_s": _median_seconds([row["text_encode_s"] + row["speaker_encode_s"] for row in rows]),
        "diffusion_s": _median_seconds([row["diffusion_s"] for row in rows]),
        "decode_s": _median_seconds([row["decode_s"] for row in rows]),
        "wall_time_s": _median_seconds([row["wall_time_s"] for row in rows]),
    }


def _measure_blockwise_once(
    runtime: BenchmarkRuntime,
    cfg: Tier2Config,
    *,
    block_sizes: list[int],
    speaker_audio: np.ndarray | None,
    continuation_audio: np.ndarray | None = None,
    text: str = "[S1] Blockwise benchmark run.",
) -> tuple[dict[str, Any], np.ndarray]:
    pipeline = runtime.pipeline
    model = pipeline.model

    ttfb_diffusion_s = float("nan")
    ttfb_audio_s = float("nan")
    block_wall_starts: list[float] = []
    block_diffusion_starts: list[float] = []
    block_kv_s: list[float] = []
    block_diffusion_s: list[float] = []
    block_wall_s: list[float] = []
    continuation_encode_s = 0.0
    last_block_done = float("nan")

    original_get_kv = model.get_kv_cache_latent
    original_encode_continuation = pipeline.encode_continuation

    def _timed_get_kv(prefix_latent: Any) -> Any:
        start = time.perf_counter()
        block_wall_starts.append(start)
        out = original_get_kv(prefix_latent)
        runtime.sync.sync(out)
        block_kv_s.append(time.perf_counter() - start)
        block_diffusion_starts.append(time.perf_counter())
        return out

    def _timed_encode_continuation(*args: Any, **kwargs: Any) -> Any:
        nonlocal continuation_encode_s
        t0 = time.perf_counter()
        out = original_encode_continuation(*args, **kwargs)
        runtime.sync.sync(out)
        continuation_encode_s += time.perf_counter() - t0
        return out

    current_block_idx = 0

    def _on_progress(step: int, num_steps: int, _t: float, _cfg_active: bool) -> None:
        nonlocal ttfb_diffusion_s, current_block_idx
        now = time.perf_counter()
        if step == num_steps:
            if math.isnan(ttfb_diffusion_s):
                ttfb_diffusion_s = now - t_origin
            if current_block_idx < len(block_diffusion_starts):
                block_diffusion_s.append(max(0.0, now - block_diffusion_starts[current_block_idx]))
            current_block_idx += 1

    def _on_block_complete(block_idx: int, _total_blocks: int, _data: Any) -> None:
        nonlocal ttfb_audio_s, last_block_done
        now = time.perf_counter()
        if 0 <= block_idx < len(block_wall_starts):
            block_wall_s.append(max(0.0, now - block_wall_starts[block_idx]))
        if block_idx == 0 and math.isnan(ttfb_audio_s):
            ttfb_audio_s = now - t_origin
        last_block_done = now

    t_origin = time.perf_counter()
    # IMPLEMENTATION NOTE: Monkey-patching model.get_kv_cache_latent and
    # pipeline.encode_continuation to measure sub-operation timing.
    #
    # This works because generate_blockwise calls these via attribute lookup
    # on each invocation (self.model.get_kv_cache_latent / self.encode_continuation),
    # so the replacement is picked up at runtime.
    #
    # Limitations:
    # - Not thread-safe with shared model/pipeline instances.
    # - Would break if generate_blockwise cached method references ahead of time.
    # - A cleaner design would pass explicit timing hooks into the pipeline API.
    #
    # try/finally below always restores the original bound methods.
    model.get_kv_cache_latent = _timed_get_kv
    if continuation_audio is not None:
        pipeline.encode_continuation = _timed_encode_continuation
    try:
        kwargs: dict[str, Any] = {
            "text": text,
            "block_sizes": list(block_sizes),
            "speaker_audio": speaker_audio,
            "seed": cfg.seed,
            "num_steps": cfg.num_steps,
            "cfg_scale_text": cfg.cfg_scale_text,
            "cfg_scale_speaker": cfg.cfg_scale_speaker,
            "truncation_factor": cfg.truncation_factor,
            "trim_latents": True,
            "decode_intermediate_blocks": False,
            "progress_callback": _on_progress,
            "on_block_complete": _on_block_complete,
        }
        if continuation_audio is not None:
            kwargs["continuation_audio"] = continuation_audio
        audio = pipeline.generate_blockwise(**kwargs)
        runtime.sync.sync(audio)
    finally:
        model.get_kv_cache_latent = original_get_kv
        pipeline.encode_continuation = original_encode_continuation

    end = time.perf_counter()
    audio_np = _flatten_audio(audio)
    audio_duration_s = float(audio_np.shape[0] / float(runtime.config.sample_rate))

    if math.isnan(ttfb_audio_s):
        ttfb_audio_s = end - t_origin
    if math.isnan(ttfb_diffusion_s):
        ttfb_diffusion_s = ttfb_audio_s

    conditioning_s = float(block_wall_starts[0] - t_origin) if block_wall_starts else 0.0
    decode_s = float(max(0.0, end - last_block_done)) if not math.isnan(last_block_done) else 0.0

    block_wall_out: list[float] = []
    block_diffusion_out: list[float] = []
    for idx in range(len(block_sizes)):
        wall_i = block_wall_s[idx] if idx < len(block_wall_s) else float("nan")
        diff_i = block_diffusion_s[idx] if idx < len(block_diffusion_s) else float("nan")
        block_wall_out.append(wall_i)
        block_diffusion_out.append(diff_i)

    result = {
        "wall_time_s": float(end - t_origin),
        "conditioning_s": float(conditioning_s),
        "ttfb_diffusion_s": float(ttfb_diffusion_s),
        "ttfb_audio_s": float(ttfb_audio_s),
        "diffusion_total_s": float(sum(x for x in block_diffusion_out if not math.isnan(x))),
        "latent_kv_total_s": float(sum(block_kv_s)),
        "decode_s": float(decode_s),
        "continuation_encode_s": float(continuation_encode_s),
        "per_block_s": block_wall_out,
        "per_block_diffusion_s": block_diffusion_out,
        "per_block_latent_kv_s": [float(x) for x in block_kv_s],
        "audio_duration_s": float(audio_duration_s),
        "realtime_factor": float(audio_duration_s / max(end - t_origin, 1e-12)),
    }
    return result, audio_np


def _run_blockwise_config(
    runtime: BenchmarkRuntime,
    cfg: Tier2Config,
    *,
    block_sizes: list[int],
    speaker_audio: np.ndarray | None,
    continuation_audio: np.ndarray | None = None,
) -> dict[str, Any]:
    for _ in range(cfg.warmup):
        _measure_blockwise_once(
            runtime,
            cfg,
            block_sizes=block_sizes,
            speaker_audio=speaker_audio,
            continuation_audio=continuation_audio,
        )

    rows: list[dict[str, Any]] = []
    audios: list[np.ndarray] = []
    for _ in range(cfg.runs):
        runtime.sync.reset_peak()
        row, audio = _measure_blockwise_once(
            runtime,
            cfg,
            block_sizes=block_sizes,
            speaker_audio=speaker_audio,
            continuation_audio=continuation_audio,
        )
        row.update(runtime.sync.get_memory_metrics())
        rows.append(row)
        audios.append(audio)

    out: dict[str, Any] = {
        "wall_time_s": _median_seconds([float(row["wall_time_s"]) for row in rows]),
        "conditioning_s": _median_seconds([float(row["conditioning_s"]) for row in rows]),
        "ttfb_diffusion_s": _median_seconds([float(row["ttfb_diffusion_s"]) for row in rows]),
        "ttfb_audio_s": _median_seconds([float(row["ttfb_audio_s"]) for row in rows]),
        "diffusion_total_s": _median_seconds([float(row["diffusion_total_s"]) for row in rows]),
        "latent_kv_total_s": _median_seconds([float(row["latent_kv_total_s"]) for row in rows]),
        "decode_s": _median_seconds([float(row["decode_s"]) for row in rows]),
        "continuation_encode_s": _median_seconds([float(row["continuation_encode_s"]) for row in rows]),
        "audio_duration_s": _median_seconds([float(row["audio_duration_s"]) for row in rows]),
    }
    out["realtime_factor"] = float(out["audio_duration_s"] / max(out["wall_time_s"], 1e-12))
    out["per_block_s"] = _median_list([row["per_block_s"] for row in rows])
    out["per_block_diffusion_s"] = _median_list([row["per_block_diffusion_s"] for row in rows])
    out["per_block_latent_kv_s"] = _median_list([row["per_block_latent_kv_s"] for row in rows])
    for key in ("peak_memory_mb", "active_memory_mb"):
        values = [float(row[key]) for row in rows if key in row]
        if values:
            out[key] = _median_seconds(values)

    if cfg.do_quality_checks:
        quality_ok, checks = _quality_from_runs(
            audios=audios,
            sample_rate=int(runtime.config.sample_rate),
            det_atol=cfg.det_atol,
            det_rtol=cfg.det_rtol,
        )
        out["quality_ok"] = bool(quality_ok)
        out["status"] = "PASS" if quality_ok else "FAIL"
        out.update(checks)
    else:
        out["quality_ok"] = True
        out["status"] = "PASS"
    return out


def _run_blockwise_breakdown(runtime: BenchmarkRuntime, cfg: Tier2Config) -> dict[str, Any]:
    speaker_audio = make_synthetic_reference_audio(sample_rate=int(runtime.config.sample_rate), duration_s=10.0)
    configs: list[tuple[str, list[int], int]] = [
        ("bw_single", [320], 320),
        ("bw_2block", [160, 160], 320),
        ("bw_3block", [128, 128, 64], 320),
        ("bw_streaming", [64, 128, 128], 320),
        ("bw_max_2block", [320, 320], 640),
    ]

    standard_wall_cache: dict[int, float] = {}
    out: dict[str, Any] = {}
    for idx, (config_id, blocks, total_frames) in enumerate(configs):
        if total_frames not in standard_wall_cache:
            baseline = _run_standard_generate_median(
                runtime,
                cfg,
                sequence_length=total_frames,
                speaker_audio=speaker_audio,
            )
            standard_wall_cache[total_frames] = float(baseline["wall_time_s"])

        row = _run_blockwise_config(runtime, cfg, block_sizes=blocks, speaker_audio=speaker_audio)
        row["block_sizes"] = list(blocks)
        row["overhead_vs_standard"] = float(row["wall_time_s"] / max(standard_wall_cache[total_frames], 1e-12))
        out[config_id] = row
        if cfg.cooldown_s > 0 and idx < len(configs) - 1:
            time.sleep(cfg.cooldown_s)

    return out


def _run_blockwise_vs_standard(runtime: BenchmarkRuntime, cfg: Tier2Config) -> dict[str, Any]:
    speaker_audio = make_synthetic_reference_audio(sample_rate=int(runtime.config.sample_rate), duration_s=10.0)
    sweeps: list[tuple[int, list[tuple[str, list[int]]]]] = [
        (
            320,
            [
                ("1_block", [320]),
                ("2_blocks", [160, 160]),
                ("3_blocks", [128, 96, 96]),
                ("4_blocks", [80, 80, 80, 80]),
            ],
        ),
        (
            640,
            [
                ("1_block", [640]),
                ("2_blocks", [320, 320]),
                ("3_blocks", [256, 192, 192]),
                ("4_blocks", [160, 160, 160, 160]),
            ],
        ),
    ]

    out: dict[str, Any] = {}
    for sweep_idx, (total_frames, configs) in enumerate(sweeps):
        standard = _run_standard_generate_median(
            runtime,
            cfg,
            sequence_length=total_frames,
            speaker_audio=speaker_audio,
        )
        components = _run_standard_component_breakdown(runtime, cfg, sequence_length=total_frames)
        standard_wall = float(standard["wall_time_s"])

        result_entry: dict[str, Any] = {
            "standard_wall_s": standard_wall,
            "standard_ttfb_s": float(standard["ttfb_s"]),
            "standard_peak_memory_mb": float(standard.get("peak_memory_mb", float("nan"))),
            "configs": {},
        }
        theoretical = {
            "conditioning_s": float(components["conditioning_s"]),
            "decode_s": float(components["decode_s"]),
            "diffusion_per_block_s": float(components["diffusion_s"]),
        }
        warnings_out: list[str] = []

        for config_idx, (label, blocks) in enumerate(configs):
            bw = _run_blockwise_config(runtime, cfg, block_sizes=blocks, speaker_audio=speaker_audio)
            wall_s = float(bw["wall_time_s"])
            overhead_ratio = float(wall_s / max(standard_wall, 1e-12))
            ttfb_audio_s = float(bw["ttfb_audio_s"])
            ttfb_speedup = float(standard_wall / max(ttfb_audio_s, 1e-12))
            num_blocks = len(blocks)
            expected_wall = (
                num_blocks * float(components["diffusion_s"])
                + float(components["conditioning_s"])
                + float(components["decode_s"])
            )
            expected_ratio = float(expected_wall / max(standard_wall, 1e-12))
            deviation_pct = float((overhead_ratio - expected_ratio) / max(expected_ratio, 1e-12) * 100.0)
            if abs(deviation_pct) > 20.0:
                warnings_out.append(
                    f"{total_frames}_frames/{label}: measured overhead deviates {deviation_pct:+.1f}% from adjusted theoretical ratio"
                )

            result_entry["configs"][label] = {
                "block_sizes": list(blocks),
                "wall_s": wall_s,
                "overhead_ratio": overhead_ratio,
                "ttfb_diffusion_s": float(bw["ttfb_diffusion_s"]),
                "ttfb_audio_s": ttfb_audio_s,
                "ttfb_speedup": ttfb_speedup,
                "peak_memory_mb": float(bw.get("peak_memory_mb", float("nan"))),
            }
            theoretical[label] = {
                "expected_ratio": expected_ratio,
                "actual_ratio": overhead_ratio,
                "deviation_pct": deviation_pct,
            }
            if cfg.cooldown_s > 0 and config_idx < len(configs) - 1:
                time.sleep(cfg.cooldown_s)

        result_entry["theoretical_check"] = theoretical
        if warnings_out:
            result_entry["warnings"] = warnings_out
        out[f"{total_frames}_frames"] = result_entry
        if cfg.cooldown_s > 0 and sweep_idx < len(sweeps) - 1:
            time.sleep(cfg.cooldown_s)

    return out


def _run_blockwise_scale_blocks(runtime: BenchmarkRuntime, cfg: Tier2Config) -> dict[str, Any]:
    speaker_audio = make_synthetic_reference_audio(sample_rate=int(runtime.config.sample_rate), duration_s=10.0)
    configs = [[320], [160, 160], [108, 108, 104], [80, 80, 80, 80], [64, 64, 64, 64, 64]]

    points: list[dict[str, Any]] = []
    for idx, blocks in enumerate(configs):
        row = _run_blockwise_config(runtime, cfg, block_sizes=blocks, speaker_audio=speaker_audio)
        points.append(
            {
                "num_blocks": len(blocks),
                "block_sizes": list(blocks),
                "wall_time_s": float(row["wall_time_s"]),
                "ttfb_audio_s": float(row["ttfb_audio_s"]),
            }
        )
        if cfg.cooldown_s > 0 and idx < len(configs) - 1:
            time.sleep(cfg.cooldown_s)

    exponent = fit_power_law_exponent(
        x=[float(point["num_blocks"]) for point in points],
        y=[float(point["wall_time_s"]) for point in points],
    )
    return {"points": points, "wall_scaling_exponent": float(exponent)}


def _run_blockwise_scale_first_block(runtime: BenchmarkRuntime, cfg: Tier2Config) -> dict[str, Any]:
    speaker_audio = make_synthetic_reference_audio(sample_rate=int(runtime.config.sample_rate), duration_s=10.0)
    first_block_sizes = [32, 64, 128, 160, 256]

    points: list[dict[str, Any]] = []
    for idx, first_size in enumerate(first_block_sizes):
        blocks = [int(first_size), int(320 - first_size)]
        row = _run_blockwise_config(runtime, cfg, block_sizes=blocks, speaker_audio=speaker_audio)
        points.append(
            {
                "first_block_size": int(first_size),
                "second_block_size": int(320 - first_size),
                "ttfb_audio_s": float(row["ttfb_audio_s"]),
                "wall_time_s": float(row["wall_time_s"]),
            }
        )
        if cfg.cooldown_s > 0 and idx < len(first_block_sizes) - 1:
            time.sleep(cfg.cooldown_s)
    return {"points": points}


def _run_blockwise_continuation(runtime: BenchmarkRuntime, cfg: Tier2Config) -> dict[str, Any]:
    speaker_audio = make_synthetic_reference_audio(sample_rate=int(runtime.config.sample_rate), duration_s=10.0)
    configs: list[tuple[str, int, list[int]]] = [
        ("cont_none", 0, [160, 160]),
        ("cont_short", 64, [128, 128]),
        ("cont_medium", 256, [192, 192]),
    ]

    out: dict[str, Any] = {}
    for idx, (config_id, continuation_frames, blocks) in enumerate(configs):
        continuation_audio = (
            None
            if continuation_frames <= 0
            else _make_continuation_audio_for_frames(runtime.config, frames=continuation_frames)
        )
        row = _run_blockwise_config(
            runtime,
            cfg,
            block_sizes=blocks,
            speaker_audio=speaker_audio,
            continuation_audio=continuation_audio,
        )
        row["continuation_frames"] = int(continuation_frames)
        row["block_sizes"] = list(blocks)
        out[config_id] = row
        if cfg.cooldown_s > 0 and idx < len(configs) - 1:
            time.sleep(cfg.cooldown_s)
    return out


def _resolve_standard_weights_dir(
    *,
    blockwise_weights_dir: Path,
    weights_standard: Path | None,
) -> tuple[Path | None, str | None]:
    if weights_standard is not None:
        candidate = Path(weights_standard)
        if candidate.exists():
            return candidate, None
        return None, f"pruned weights not found at {candidate}"

    derived = blockwise_weights_dir.parent / "converted"
    if derived.exists():
        return derived, None
    return None, f"pruned weights not found at {derived}"


def _run_blockwise_standard_regression(
    runtime: BenchmarkRuntime,
    cfg: Tier2Config,
    *,
    weights_standard: Path | None,
) -> dict[str, Any]:
    standard_weights_dir, error = _resolve_standard_weights_dir(
        blockwise_weights_dir=runtime.weights_dir,
        weights_standard=weights_standard,
    )
    if standard_weights_dir is None:
        return {"skipped": error or "pruned weights not found"}

    pruned_runtime = _load_runtime(
        weights_dir=standard_weights_dir,
        dtype=runtime.dtype,
        quantize=runtime.quantize,
    )
    speaker_audio = make_synthetic_reference_audio(sample_rate=int(runtime.config.sample_rate), duration_s=10.0)
    pruned = _run_standard_generate_median(pruned_runtime, cfg, sequence_length=320, speaker_audio=speaker_audio)
    blockwise_std = _run_standard_generate_median(runtime, cfg, sequence_length=320, speaker_audio=speaker_audio)

    pruned_wall = float(pruned["wall_time_s"])
    blockwise_wall = float(blockwise_std["wall_time_s"])
    regression_pct = float((blockwise_wall - pruned_wall) / max(pruned_wall, 1e-12) * 100.0)
    return {
        "weights_standard": str(standard_weights_dir),
        "std_pruned": pruned,
        "std_blockwise_weights": blockwise_std,
        "std_pruned_wall_s": pruned_wall,
        "std_blockwise_weights_wall_s": blockwise_wall,
        "regression_pct": regression_pct,
        "gate_passed": bool(regression_pct <= 5.0),
    }


def _run_tier2(
    runtime: BenchmarkRuntime,
    cfg: Tier2Config,
    *,
    name_filter: str,
    weights_standard: Path | None = None,
) -> dict[str, Any]:
    benchmarks: OrderedDict[str, Callable[[], dict[str, Any]]] = OrderedDict(
        [
            ("bench_breakdown_unconditional", lambda: _run_breakdown_benchmark(runtime, cfg, use_speaker=False)),
            ("bench_breakdown_cloned", lambda: _run_breakdown_benchmark(runtime, cfg, use_speaker=True)),
            ("bench_ttfb", lambda: _bench_ttfb(runtime, cfg)),
            ("bench_scale_steps", lambda: _run_scale_steps(runtime, cfg)),
            ("bench_scale_seq_length", lambda: _run_scale_seq_length(runtime, cfg)),
            ("bench_scale_speaker_length", lambda: _run_scale_speaker_length(runtime, cfg)),
            ("bench_scale_text_length", lambda: _run_scale_text_length(runtime, cfg)),
            ("bench_blockwise_breakdown", lambda: _run_blockwise_breakdown(runtime, cfg)),
            ("bench_blockwise_vs_standard", lambda: _run_blockwise_vs_standard(runtime, cfg)),
            ("bench_blockwise_scale_blocks", lambda: _run_blockwise_scale_blocks(runtime, cfg)),
            ("bench_blockwise_scale_first_block", lambda: _run_blockwise_scale_first_block(runtime, cfg)),
            ("bench_blockwise_continuation", lambda: _run_blockwise_continuation(runtime, cfg)),
            (
                "bench_blockwise_standard_regression",
                lambda: _run_blockwise_standard_regression(runtime, cfg, weights_standard=weights_standard),
            ),
        ]
    )

    blockwise_benchmarks = {
        "bench_blockwise_breakdown",
        "bench_blockwise_vs_standard",
        "bench_blockwise_scale_blocks",
        "bench_blockwise_scale_first_block",
        "bench_blockwise_continuation",
        "bench_blockwise_standard_regression",
    }

    selected = filter_benchmark_names(list(benchmarks.keys()), name_filter)
    out: dict[str, Any] = {}
    for idx, name in enumerate(selected):
        if name in blockwise_benchmarks and not runtime.blockwise_capable:
            out[name] = dict(SKIP_NO_BLOCKWISE)
            continue
        runtime.sync.reset_peak()
        out[name] = benchmarks[name]()
        if cfg.cooldown_s > 0 and idx < len(selected) - 1:
            time.sleep(cfg.cooldown_s)

    return out


def _detect_device_name() -> str:
    machine = platform.machine()
    processor = platform.processor()
    if processor:
        return f"{processor} ({machine})"
    return machine


def _safe_package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_sha() -> str | None:
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha if sha else None


def _build_metadata(*, runtime: BenchmarkRuntime, warmup: int, cooldown_s: float) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "implementation": "echo-tts-mlx",
        "version": _safe_package_version("echo-tts-mlx") or "0.1.0",
        "backend": "mlx",
        "device": _detect_device_name(),
        "os": platform.platform(),
        "dtype": runtime.dtype,
        "quantize": runtime.quantize,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "mlx_version": _safe_package_version("mlx"),
        "warmup_runs": int(warmup),
        "cooldown_s": float(cooldown_s),
    }

    if runtime.config is not None:
        metadata["sample_rate"] = int(runtime.config.sample_rate)

    return metadata


def _run_tier3(
    runtime: BenchmarkRuntime,
    *,
    det_atol: float,
    det_rtol: float,
    quality_checks: bool,
    cache_dir: Path,
    timeout_s: float,
    force_synthetic_reference: bool,
) -> dict[str, Any]:
    from benchmarks.cross_impl_protocol import AbstractBenchmarkRunner, run_cross_impl_suite

    class _MlxTier3Runner(AbstractBenchmarkRunner):
        def __init__(self, *, runtime_obj: BenchmarkRuntime) -> None:
            super().__init__(
                implementation="echo-tts-mlx",
                version=_safe_package_version("echo-tts-mlx") or "0.1.0",
                backend="mlx",
                device=_detect_device_name(),
                dtype=runtime_obj.dtype,
                seed=0,
                cfg_scale_text=3.0,
                cfg_scale_speaker=8.0,
                truncation_factor=0.8,
                sample_rate=int(runtime_obj.config.sample_rate),
            )
            self.runtime = runtime_obj

        def run_case(
            self,
            *,
            text: str,
            speaker_audio: np.ndarray | None,
            seq_length: int,
            num_steps: int,
            seed: int,
            cfg_scale_text: float,
            cfg_scale_speaker: float,
            truncation_factor: float,
        ) -> np.ndarray:
            audio = self.runtime.pipeline.generate(
                text=text,
                speaker_audio=speaker_audio,
                sequence_length=seq_length,
                seed=seed,
                num_steps=num_steps,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_speaker=cfg_scale_speaker,
                truncation_factor=truncation_factor,
                trim_latents=True,
            )
            self.runtime.sync.sync(audio)
            return _flatten_audio(audio)

        def run_case_blockwise(
            self,
            *,
            text: str,
            speaker_audio: np.ndarray | None,
            block_sizes: list[int],
            num_steps: int,
            seed: int,
            cfg_scale_text: float,
            cfg_scale_speaker: float,
            truncation_factor: float,
        ) -> tuple[np.ndarray, dict[str, float]]:
            if not self.runtime.blockwise_capable:
                raise NotImplementedError("no blockwise weights")

            t0 = time.perf_counter()
            ttfb_diffusion_s = float("nan")
            ttfb_audio_s = float("nan")

            def _on_progress(step: int, _total_steps: int, _t: float, _cfg_active: bool) -> None:
                nonlocal ttfb_diffusion_s
                # NOTE: Tier 3 diffusion progress timestamps are approximate because
                # we cannot reliably reach inside the sampler from this callback to
                # evaluate the current diffusion tensor before reading the clock.
                if step == num_steps and math.isnan(ttfb_diffusion_s):
                    ttfb_diffusion_s = time.perf_counter() - t0

            def _on_block_complete(block_idx: int, _total_blocks: int, _block_audio: Any) -> None:
                nonlocal ttfb_audio_s
                # mx.eval() barrier: force lazy MLX work to complete before timing.
                # Without this, callback timestamps can under-report GPU compute time.
                self.runtime.mx.eval(_block_audio)
                if block_idx == 0 and math.isnan(ttfb_audio_s):
                    ttfb_audio_s = time.perf_counter() - t0

            audio = self.runtime.pipeline.generate_blockwise(
                text=text,
                block_sizes=list(block_sizes),
                speaker_audio=speaker_audio,
                seed=seed,
                num_steps=num_steps,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_speaker=cfg_scale_speaker,
                truncation_factor=truncation_factor,
                trim_latents=True,
                progress_callback=_on_progress,
                on_block_complete=_on_block_complete,
            )
            self.runtime.sync.sync(audio)
            if math.isnan(ttfb_audio_s):
                ttfb_audio_s = time.perf_counter() - t0
            if math.isnan(ttfb_diffusion_s):
                ttfb_diffusion_s = float(ttfb_audio_s)
            return _flatten_audio(audio), {
                "ttfb_diffusion_s": float(ttfb_diffusion_s),
                "ttfb_audio_s": float(ttfb_audio_s),
            }

    runner = _MlxTier3Runner(runtime_obj=runtime)
    out = run_cross_impl_suite(
        runner=runner,
        cache_dir=cache_dir,
        quality_checks=quality_checks,
        det_atol=det_atol,
        det_rtol=det_rtol,
        timeout_s=timeout_s,
        force_synthetic_reference=force_synthetic_reference,
    )
    if not runtime.blockwise_capable:
        out["tier3_blockwise"] = dict(SKIP_NO_BLOCKWISE)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Echo-TTS benchmark runner (Tier 1/2/3)")
    parser.add_argument("--tier", choices=["all", "1", "2", "3"], default="all", help="Benchmark tier to run")
    parser.add_argument(
        "--filter",
        default="",
        help="Run only benchmarks whose name contains this substring (e.g., 'blockwise', 'tier1', 'continuation')",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument("--weights", type=Path, default=None, help="Converted weights directory (auto-detects blockwise if available)")
    parser.add_argument(
        "--weights-standard",
        type=Path,
        default=None,
        help="Optional pruned/standard weights directory for bench_blockwise_standard_regression",
    )
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16", help="Inference dtype")
    parser.add_argument("--quantize", choices=["none", "8bit", "4bit", "mxfp4", "mixed"], default="none", help="Quantization mode")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup runs for non-cold benchmarks")
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Measured runs override. Defaults: Tier1=5, Tier2=3",
    )
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_S, help="Cooldown seconds between Tier2 benchmarks")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic fixtures/noise")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Diffusion steps for Tier2 breakdown benchmarks")
    parser.add_argument("--seq-length", type=int, default=DEFAULT_SEQ_LENGTH, help="Sequence length for Tier2 breakdown benchmarks")
    parser.add_argument("--cfg-text", type=float, default=3.0, help="Text guidance scale")
    parser.add_argument("--cfg-speaker", type=float, default=8.0, help="Speaker guidance scale")
    parser.add_argument("--truncation-factor", type=float, default=0.8, help="Initial noise truncation factor")
    parser.add_argument("--det-atol", type=float, default=1e-5, help="Determinism allclose absolute tolerance")
    parser.add_argument("--det-rtol", type=float, default=1e-4, help="Determinism allclose relative tolerance")
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=DEFAULT_REFERENCE_CACHE,
        help="Tier 3 reference cache directory",
    )
    parser.add_argument(
        "--reference-timeout",
        type=float,
        default=DEFAULT_REFERENCE_TIMEOUT,
        help="Tier 3 reference download timeout seconds",
    )
    parser.add_argument(
        "--force-synthetic-reference",
        action="store_true",
        help="Tier 3: skip LJ Speech download and use deterministic synthetic reference",
    )
    parser.add_argument("--no-quality-check", action="store_true", help="Skip quality gate checks")
    return parser


def _normalize_report(report: dict[str, Any]) -> dict[str, Any]:
    # Drop None metadata values for cleaner JSON.
    metadata = report.get("metadata", {})
    report["metadata"] = {k: v for k, v in metadata.items() if v is not None}
    return report


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)

    if not _mlx_runtime_available():
        raise RuntimeError("MLX runtime unavailable (no compatible Metal device/runtime in this environment).")

    tier1_runs = int(args.runs) if args.runs is not None else DEFAULT_TIER1_RUNS
    tier2_runs = int(args.runs) if args.runs is not None else DEFAULT_TIER2_RUNS
    if tier1_runs < 1 or tier2_runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    # Auto-detect weights directory: prefer blockwise (full capability), fall back to standard
    weights_dir = args.weights
    if weights_dir is None:
        blockwise_candidate = Path("weights/converted-blockwise")
        standard_candidate = Path("weights/converted")
        if blockwise_candidate.exists():
            weights_dir = blockwise_candidate
            print(f"Auto-detected blockwise weights: {blockwise_candidate}")
        elif standard_candidate.exists():
            weights_dir = standard_candidate
            print(f"Auto-detected standard weights: {standard_candidate}")
        else:
            raise FileNotFoundError(
                "No weights directory found. Expected weights/converted-blockwise or weights/converted. "
                "Use --weights to specify a custom path."
            )

    runtime = _load_runtime(weights_dir=weights_dir, dtype=args.dtype, quantize=args.quantize)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metadata": _build_metadata(runtime=runtime, warmup=args.warmup, cooldown_s=args.cooldown),
    }

    if args.tier in ("all", "1"):
        report["tier1"] = _benchmark_tier1(runtime, warmup=args.warmup, runs=tier1_runs, name_filter=args.filter)

    if args.tier in ("all", "2"):
        tier2_cfg = Tier2Config(
            runs=tier2_runs,
            warmup=args.warmup,
            cooldown_s=args.cooldown,
            sequence_length=args.seq_length,
            num_steps=args.steps,
            seed=args.seed,
            do_quality_checks=not args.no_quality_check,
            det_atol=args.det_atol,
            det_rtol=args.det_rtol,
            cfg_scale_text=args.cfg_text,
            cfg_scale_speaker=args.cfg_speaker,
            truncation_factor=args.truncation_factor,
        )
        report["tier2"] = _run_tier2(
            runtime,
            tier2_cfg,
            name_filter=args.filter,
            weights_standard=args.weights_standard,
        )
        report["metadata"]["tier2_diffusion_timing_note"] = TIER2_DIFFUSION_TIMING_NOTE

    if args.tier in ("all", "3"):
        tier3_report = _run_tier3(
            runtime,
            det_atol=args.det_atol,
            det_rtol=args.det_rtol,
            quality_checks=not args.no_quality_check,
            cache_dir=args.reference_cache,
            timeout_s=args.reference_timeout,
            force_synthetic_reference=bool(args.force_synthetic_reference),
        )
        report["tier3"] = tier3_report["tier3"]
        if "tier3_blockwise" in tier3_report:
            report["tier3_blockwise"] = tier3_report["tier3_blockwise"]
        report["metadata"].update(tier3_report["metadata"])

    return _normalize_report(report)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(argv)
    except (ValueError, FileNotFoundError, RuntimeError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args = build_parser().parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=False))
    print(f"Wrote benchmark report: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
