"""Echo-TTS inference pipeline."""

from __future__ import annotations

from collections.abc import Callable
import importlib
from pathlib import Path
import shutil
from typing import Any
import warnings

import numpy as np

from .autoencoder import MlxFishS1DAC
from .config import ModelConfig, load_model_config
from .model import MlxEchoDiT, save_quantize_config
from .pca import PCAState, load_pca_state
from .sampler import (
    BlockwiseSamplerConfig,
    SamplerConfig,
    find_content_boundary,
    find_flattening_point,
    sample_blockwise_euler_cfg,
    sample_euler_cfg_independent_guidances,
)
from .tokenizer import tokenize
from .utils import save_audio as save_audio_file


DEFAULT_CONVERTED_WEIGHTS_DIR = Path("weights/converted")
VALID_QUANTIZE_MODES = {"none", "8bit", "4bit", "mxfp4", "mixed"}
VALID_TRIM_MODES = {"latent", "energy", "f0"}
# Experimental defaults pending calibration sweep with listening tests.
ADAPTIVE_TRUNCATION: dict[int, float] = {
    100: 0.8,
    150: 0.8,
    200: 0.8,
    300: 0.8,
    400: 0.8,
    640: 0.8,
}
QUALITY_PRESETS: dict[str, dict[str, float | int | None | str]] = {
    "quality": {"num_steps": 32, "truncation_factor": "auto"},
    "ultra": {"num_steps": 40, "truncation_factor": "auto"},
    "balanced": {"num_steps": 16, "truncation_factor": "auto"},
    "fast": {"num_steps": 8, "truncation_factor": "auto"},
    "draft": {"num_steps": 4, "truncation_factor": "auto"},
}


def _require_mlx():
    try:
        return importlib.import_module("mlx.core")
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MLX is required for EchoTTS pipeline.") from exc


def _is_mlx_array(x: Any) -> bool:
    return x.__class__.__module__.startswith("mlx")


def _validate_quantize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_QUANTIZE_MODES:
        raise ValueError(f"Invalid quantize mode '{mode}'. Expected one of: none, 8bit, 4bit, mxfp4, mixed.")
    return normalized


def _validate_trim_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_TRIM_MODES:
        raise ValueError(f"Invalid trim mode '{mode}'. Expected one of: latent, energy, f0.")
    return normalized


def resolve_adaptive_truncation(sequence_length: int | None) -> float:
    if not ADAPTIVE_TRUNCATION:
        return 0.8

    xs = np.asarray(sorted(ADAPTIVE_TRUNCATION.keys()), dtype=np.float32)
    ys = np.asarray([float(ADAPTIVE_TRUNCATION[int(x)]) for x in xs], dtype=np.float32)
    if sequence_length is None:
        return float(ys[-1])

    seq = float(sequence_length)
    if seq <= float(xs[0]):
        return float(ys[0])
    if seq >= float(xs[-1]):
        return float(ys[-1])
    return float(np.interp(seq, xs, ys))


def resolve_quality_preset(name: str, *, sequence_length: int | None = None) -> tuple[int, float | None]:
    key = str(name).strip().lower()
    if key not in QUALITY_PRESETS:
        valid = ", ".join(sorted(QUALITY_PRESETS))
        raise ValueError(f"Invalid preset '{name}'. Expected one of: {valid}.")
    cfg = QUALITY_PRESETS[key]
    truncation_raw = cfg["truncation_factor"]
    if truncation_raw is None:
        truncation = None
    elif str(truncation_raw).strip().lower() == "auto":
        truncation = resolve_adaptive_truncation(sequence_length)
    else:
        truncation = float(truncation_raw)
    return int(cfg["num_steps"]), truncation


def _normalize_block_sizes(block_sizes: list[int] | tuple[int, ...], *, patch_size: int) -> list[int]:
    sizes = [int(x) for x in block_sizes]
    if not sizes:
        raise ValueError("block_sizes must not be empty.")
    if any(size <= 0 for size in sizes):
        raise ValueError("All block sizes must be > 0.")
    if any(size < patch_size for size in sizes):
        raise ValueError(f"All block sizes must be >= {patch_size} latent frames.")
    if any(size < 32 for size in sizes):
        warnings.warn("Block sizes < 32 may degrade quality.", stacklevel=2)
    if any(size % patch_size != 0 for size in sizes):
        warnings.warn(
            f"Block sizes not divisible by patch size ({patch_size}) will be zero-padded in latent encoder.",
            stacklevel=2,
        )
    return sizes


class EchoTTS:
    """End-to-end tensor-first inference pipeline."""

    def __init__(
        self,
        *,
        model: MlxEchoDiT,
        autoencoder: MlxFishS1DAC,
        config: ModelConfig,
        pca_state: PCAState,
        quantize: str = "none",
        weights_dir: Path | None = None,
    ) -> None:
        mx = _require_mlx()
        self.mx = mx
        self.model = model
        self.autoencoder = autoencoder
        self.config = config
        self.pca_state = pca_state
        self.quantize = _validate_quantize_mode(quantize)
        self.weights_dir = Path(weights_dir) if weights_dir is not None else DEFAULT_CONVERTED_WEIGHTS_DIR

        self._pca_components = mx.array(pca_state.pca_components, dtype=mx.float32)
        self._pca_mean = mx.array(pca_state.pca_mean, dtype=mx.float32)
        self._latent_scale = float(pca_state.latent_scale)

    @classmethod
    def from_pretrained(
        cls,
        weights_dir: str | Path = DEFAULT_CONVERTED_WEIGHTS_DIR,
        *,
        dtype: str = "float16",
        quantize: str = "none",
    ) -> "EchoTTS":
        quantize_mode = _validate_quantize_mode(quantize)
        config = load_model_config(weights_dir)
        model = MlxEchoDiT.from_pretrained(weights_dir, dtype=dtype, quantize=quantize_mode)
        autoencoder = MlxFishS1DAC.from_pretrained(weights_dir, dtype="float32")
        pca_state = load_pca_state(weights_dir)
        return cls(
            model=model,
            autoencoder=autoencoder,
            config=config,
            pca_state=pca_state,
            quantize=quantize_mode,
            weights_dir=Path(weights_dir),
        )

    def save_quantized(self, output_dir: str | Path) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        mode = _validate_quantize_mode(self.model.quantize_mode)
        if mode == "none":
            raise ValueError("Model is not quantized; load with --quantize 8bit, --quantize 4bit, --quantize mxfp4, or --quantize mixed first.")

        # Save DiT weights in-place after quantization.
        self.model.save_weights(out_dir / "dit_weights.safetensors")

        # Preserve non-DiT artifacts from source checkpoint.
        copy_names = ("config.json", "dac_weights.safetensors", "pca_state.safetensors", "weight_map.json")
        for name in copy_names:
            src = self.weights_dir / name
            if src.exists():
                shutil.copy2(src, out_dir / name)

        qcfg = self.model.quantize_config(group_size=64)
        save_quantize_config(out_dir, qcfg)
        return out_dir

    def _to_mx_array(self, x: Any, *, dtype: Any | None = None):
        mx = self.mx
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        if _is_mlx_array(x):
            return x.astype(dtype) if dtype is not None else x
        out = mx.array(np.asarray(x))
        return out.astype(dtype) if dtype is not None else out

    def pca_encode(self, z_q: Any):
        """Encode `(B, 1024, T)` DAC latents into `(B, T, 80)` diffusion latents."""
        mx = self.mx
        z_q = self._to_mx_array(z_q, dtype=mx.float32)
        if z_q.ndim != 3 or int(z_q.shape[1]) != self.config.pca_latent_dim:
            raise ValueError(
                f"z_q must have shape (B, {self.config.pca_latent_dim}, T), got {tuple(z_q.shape)}"
            )
        z = mx.transpose(z_q, (0, 2, 1))
        z = mx.matmul(z - mx.reshape(self._pca_mean, (1, 1, self.config.pca_latent_dim)), mx.transpose(self._pca_components))
        return z * self._latent_scale

    def pca_decode(self, z: Any):
        """Decode `(B, T, 80)` diffusion latents into `(B, 1024, T)` DAC latents."""
        mx = self.mx
        z = self._to_mx_array(z, dtype=mx.float32)
        if z.ndim != 3 or int(z.shape[2]) != self.config.latent_size:
            raise ValueError(f"z must have shape (B, T, {self.config.latent_size}), got {tuple(z.shape)}")
        z_q = (z / self._latent_scale) @ self._pca_components
        z_q = z_q + mx.reshape(self._pca_mean, (1, 1, self.config.pca_latent_dim))
        return mx.transpose(z_q, (0, 2, 1))

    def tokenize_text(self, text: str) -> list[int]:
        return tokenize(text, max_length=self.config.max_text_length)

    def prepare_text(self, text: str):
        mx = self.mx
        token_ids = self.tokenize_text(text)
        text_ids = mx.array(np.asarray(token_ids, dtype=np.int32)[None, :], dtype=mx.int32)
        text_mask = mx.ones((1, len(token_ids)), dtype=mx.bool_)
        return text_ids, text_mask

    def _normalize_audio(self, audio: Any) -> np.ndarray:
        x = audio
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 1:
            mono = x
        elif x.ndim == 2:
            if x.shape[0] == 1:
                mono = x[0]
            elif x.shape[1] == 1:
                mono = x[:, 0]
            elif x.shape[0] <= x.shape[1]:
                mono = x.mean(axis=0)
            else:
                mono = x.mean(axis=1)
        elif x.ndim == 3:
            if x.shape[0] != 1:
                raise ValueError(f"Expected a single reference clip (batch=1), got shape {tuple(x.shape)}")
            if x.shape[1] == 1:
                mono = x[0, 0]
            else:
                mono = x[0].mean(axis=0)
        else:
            raise ValueError(f"Unsupported speaker_audio shape: {tuple(x.shape)}")

        mono = mono.astype(np.float32, copy=False)
        peak = float(np.max(np.abs(mono))) if mono.size > 0 else 0.0
        mono = mono / max(peak, 1.0)
        return mono.reshape(1, 1, -1)

    def prepare_speaker_latents(
        self,
        *,
        speaker_latents: Any | None = None,
        speaker_audio: Any | None = None,
        speaker_mask: Any | None = None,
    ):
        """Return `(speaker_latents, speaker_mask)` with shape `(1, T, 80)` and `(1, T)`."""
        if speaker_latents is not None and speaker_audio is not None:
            raise ValueError("Pass either speaker_latents or speaker_audio, not both.")
        if speaker_latents is None and speaker_audio is None:
            raise ValueError("One of speaker_latents or speaker_audio must be provided.")

        if speaker_latents is not None:
            return self._prepare_speaker_latents_direct(speaker_latents, speaker_mask=speaker_mask)
        return self._prepare_speaker_latents_from_audio(speaker_audio)

    def _prepare_speaker_latents_direct(self, speaker_latents: Any, *, speaker_mask: Any | None):
        mx = self.mx
        latents = self._to_mx_array(speaker_latents, dtype=mx.float32)
        if latents.ndim == 2:
            latents = mx.reshape(latents, (1, latents.shape[0], latents.shape[1]))
        if latents.ndim != 3 or int(latents.shape[0]) != 1 or int(latents.shape[2]) != self.config.latent_size:
            raise ValueError(
                f"speaker_latents must have shape (1, T, {self.config.latent_size}) or (T, {self.config.latent_size}), "
                f"got {tuple(latents.shape)}"
            )

        t = min(int(latents.shape[1]), self.config.max_speaker_latent_length)
        latents = latents[:, :t, :]

        if speaker_mask is None:
            mask = mx.ones((1, t), dtype=mx.bool_)
            actual_length = t
        else:
            mask = self._to_mx_array(speaker_mask, dtype=mx.bool_)
            if mask.ndim == 1:
                mask = mx.reshape(mask, (1, mask.shape[0]))
            if mask.ndim != 2 or int(mask.shape[0]) != 1:
                raise ValueError(f"speaker_mask must have shape (1, T) or (T,), got {tuple(mask.shape)}")
            mask = mask[:, :t].astype(mx.bool_)
            actual_length = int(np.asarray(mask, dtype=np.bool_).sum())

        trim_to = actual_length - (actual_length % self.config.speaker_patch_size)
        if trim_to <= 0:
            raise ValueError("Speaker latents are too short after patch-size trimming.")

        latents = latents[:, :trim_to, :]
        mask = mx.ones((1, trim_to), dtype=mx.bool_)
        return latents, mask

    def _prepare_speaker_latents_from_audio(self, speaker_audio: Any):
        mx = self.mx
        audio = self._normalize_audio(speaker_audio)

        max_samples = self.config.max_speaker_latent_length * self.config.ae_downsample_factor
        audio = audio[..., :max_samples]

        total_samples = int(audio.shape[-1])
        if total_samples < self.config.ae_downsample_factor:
            raise ValueError(
                f"speaker_audio too short: need at least {self.config.ae_downsample_factor} samples, got {total_samples}"
            )

        chunk_samples = self.config.max_latent_length * self.config.ae_downsample_factor
        actual_length = total_samples // self.config.ae_downsample_factor
        actual_length = min(actual_length, self.config.max_speaker_latent_length)

        z_chunks: list[Any] = []
        for start in range(0, total_samples, chunk_samples):
            chunk = audio[..., start : start + chunk_samples]
            if chunk.shape[-1] < chunk_samples:
                pad = chunk_samples - int(chunk.shape[-1])
                chunk = np.pad(chunk, ((0, 0), (0, 0), (0, pad)))
            z_q, _, _ = self.autoencoder.encode_zq(chunk)
            mx.eval(z_q)
            z_chunks.append(z_q)

        if not z_chunks:
            raise ValueError("Failed to produce speaker latents from empty audio input.")

        z_q = z_chunks[0] if len(z_chunks) == 1 else mx.concatenate(z_chunks, axis=-1)
        z_q = z_q[:, :, :actual_length]

        latents = self.pca_encode(z_q)

        trim_to = actual_length - (actual_length % self.config.speaker_patch_size)
        if trim_to <= 0:
            raise ValueError("Speaker reference produced fewer than one patch after trimming.")
        latents = latents[:, :trim_to, :]
        mask = mx.ones((1, trim_to), dtype=mx.bool_)
        return latents, mask

    def _repeat_kv_cache(self, kv: list[tuple[Any, Any]], repeats: int) -> list[tuple[Any, Any]]:
        mx = self.mx
        out: list[tuple[Any, Any]] = []
        for k, v in kv:
            out.append((mx.concatenate([k] * repeats, axis=0), mx.concatenate([v] * repeats, axis=0)))
        return out

    def _scale_kv_cache(self, kv: list[tuple[Any, Any]], scale: float, max_layers: int | None = None) -> None:
        limit = len(kv) if max_layers is None else min(max_layers, len(kv))
        for i in range(limit):
            k, v = kv[i]
            kv[i] = (k * scale, v * scale)

    def generate_latents(
        self,
        *,
        text: str,
        speaker_latents: Any | None = None,
        speaker_audio: Any | None = None,
        speaker_mask: Any | None = None,
        noise: Any | None = None,
        sequence_length: int | None = None,
        seed: int | None = 0,
        num_steps: int = 32,
        cfg_scale_text: float = 3.0,
        cfg_scale_speaker: float = 8.0,
        cfg_min_t: float = 0.5,
        cfg_max_t: float = 1.0,
        truncation_factor: float | None = 0.8,
        speaker_kv_scale: float | None = None,
        speaker_kv_max_layers: int | None = None,
        speaker_kv_min_t: float | None = None,
        progress_callback: Callable[[int, int, float, bool], None] | None = None,
    ):
        """Run text+speaker-conditioned diffusion and return final latents `(1, T, 80)`."""
        mx = self.mx

        text_ids, text_mask = self.prepare_text(text)
        use_speaker = speaker_latents is not None or speaker_audio is not None
        if not use_speaker and speaker_mask is not None:
            raise ValueError("speaker_mask was provided without speaker latents/audio.")
        if not use_speaker and speaker_kv_scale is not None:
            raise ValueError("speaker_kv_scale requires speaker conditioning.")

        kv_text_cond, text_k_mask = self.model.get_kv_cache_text(text_ids, text_mask)

        if use_speaker:
            speaker_latents, speaker_mask = self.prepare_speaker_latents(
                speaker_latents=speaker_latents,
                speaker_audio=speaker_audio,
                speaker_mask=speaker_mask,
            )
        else:
            # Upstream always provides speaker KV — even unconditionally.
            # Zero latents + zero mask preserves the attention pattern the model
            # was trained with (different KV length = different softmax normalization).
            speaker_latents = mx.zeros((1, 4, self.config.latent_size), dtype=mx.float32)
            speaker_mask = mx.zeros((1, 4), dtype=mx.bool_)

        kv_speaker_cond, speaker_k_mask = self.model.get_kv_cache_speaker(speaker_latents, speaker_mask)
        if speaker_kv_scale is not None:
            self._scale_kv_cache(kv_speaker_cond, float(speaker_kv_scale), max_layers=speaker_kv_max_layers)

        kv_text_full = self._repeat_kv_cache(kv_text_cond, repeats=3)
        kv_speaker_full = self._repeat_kv_cache(kv_speaker_cond, repeats=3)

        text_mask_batch = mx.concatenate(
            [text_k_mask, mx.zeros_like(text_k_mask), text_k_mask],
            axis=0,
        )
        speaker_mask_batch = mx.concatenate(
            [speaker_k_mask, speaker_k_mask, mx.zeros_like(speaker_k_mask)],
            axis=0,
        )

        if noise is None:
            seq_len = sequence_length if sequence_length is not None else self.config.max_latent_length
            if seed is not None:
                mx.random.seed(seed)
            x_t = mx.random.normal((1, seq_len, self.config.latent_size), dtype=mx.float32)
        else:
            x_t = self._to_mx_array(noise, dtype=mx.float32)

        if x_t.ndim != 3 or int(x_t.shape[0]) != 1 or int(x_t.shape[2]) != self.config.latent_size:
            raise ValueError(
                f"noise must have shape (1, T, {self.config.latent_size}), got {tuple(x_t.shape)}"
            )

        sampler_cfg = SamplerConfig(
            num_steps=num_steps,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_min_t=cfg_min_t,
            cfg_max_t=cfg_max_t,
            truncation_factor=truncation_factor,
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_max_layers=speaker_kv_max_layers,
            speaker_kv_min_t=speaker_kv_min_t,
        )

        def _predict_velocity(x_cur, t: float, cfg_active: bool):
            if cfg_active:
                # Always use 3-way CFG (upstream always passes speaker KV)
                x_batch = mx.concatenate([x_cur, x_cur, x_cur], axis=0)
                t_batch = mx.array([t, t, t], dtype=x_cur.dtype)
                out = self.model.forward(
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
            return self.model.forward(
                x_cur,
                t_single,
                kv_text=kv_text_cond,
                kv_speaker=kv_speaker_cond,
                text_mask=text_k_mask,
                speaker_mask=speaker_k_mask,
            )

        def _reverse_speaker_scale() -> None:
            nonlocal kv_speaker_full
            if speaker_kv_scale is None or kv_speaker_cond is None:
                return
            self._scale_kv_cache(
                kv_speaker_cond,
                1.0 / float(speaker_kv_scale),
                max_layers=speaker_kv_max_layers,
            )
            kv_speaker_full = self._repeat_kv_cache(kv_speaker_cond, repeats=3)

        latents = sample_euler_cfg_independent_guidances(
            x_t=x_t,
            config=sampler_cfg,
            predict_velocity=_predict_velocity,
            eval_step=lambda arr: mx.eval(arr),
            on_speaker_kv_scale_reversal=_reverse_speaker_scale,
            on_step=progress_callback,
        )
        mx.eval(latents)
        return latents

    def encode_continuation(
        self,
        *,
        audio: Any | None = None,
        latents: Any | None = None,
    ) -> tuple[Any, int]:
        """Encode continuation input to `(1, T, 80)` latents and return `(latents, length)`."""
        mx = self.mx
        if audio is not None and latents is not None:
            raise ValueError("Pass either audio or latents for continuation, not both.")
        if audio is None and latents is None:
            raise ValueError("One of audio or latents must be provided for continuation.")

        patch = int(self.config.speaker_patch_size)

        if latents is not None:
            cont = self._to_mx_array(latents, dtype=mx.float32)
            if cont.ndim == 2:
                cont = mx.reshape(cont, (1, cont.shape[0], cont.shape[1]))
            if cont.ndim != 3 or int(cont.shape[0]) != 1 or int(cont.shape[2]) != self.config.latent_size:
                raise ValueError(
                    f"continuation_latents must have shape (1, T, {self.config.latent_size}) or (T, {self.config.latent_size}), "
                    f"got {tuple(cont.shape)}"
                )
            t = min(int(cont.shape[1]), int(self.config.max_latent_length))
            trim_to = t - (t % patch)
            if trim_to <= 0:
                raise ValueError("Continuation latents are too short after patch-size trimming.")
            if trim_to != t:
                warnings.warn(
                    f"Continuation length trimmed from {t} to {trim_to} frames for patch alignment.",
                    stacklevel=2,
                )
            cont = cont[:, :trim_to, :]
            mx.eval(cont)
            return cont, trim_to

        audio_norm = self._normalize_audio(audio)
        max_samples = self.config.max_latent_length * self.config.ae_downsample_factor
        audio_norm = audio_norm[..., :max_samples]
        total_samples = int(audio_norm.shape[-1])
        if total_samples < self.config.ae_downsample_factor:
            raise ValueError(
                f"continuation_audio too short: need at least {self.config.ae_downsample_factor} samples, got {total_samples}"
            )

        chunk_samples = self.config.max_latent_length * self.config.ae_downsample_factor
        actual_length = min(total_samples // self.config.ae_downsample_factor, self.config.max_latent_length)

        z_chunks: list[Any] = []
        for start in range(0, total_samples, chunk_samples):
            chunk = audio_norm[..., start : start + chunk_samples]
            if chunk.shape[-1] < chunk_samples:
                pad = chunk_samples - int(chunk.shape[-1])
                chunk = np.pad(chunk, ((0, 0), (0, 0), (0, pad)))
            z_q, _, _ = self.autoencoder.encode_zq(chunk)
            mx.eval(z_q)
            z_chunks.append(z_q)

        if not z_chunks:
            raise ValueError("Failed to produce continuation latents from empty audio input.")

        z_q = z_chunks[0] if len(z_chunks) == 1 else mx.concatenate(z_chunks, axis=-1)
        z_q = z_q[:, :, :actual_length]
        cont = self.pca_encode(z_q)

        trim_to = actual_length - (actual_length % patch)
        if trim_to <= 0:
            raise ValueError("Continuation audio produced fewer than one patch after trimming.")
        if trim_to != actual_length:
            warnings.warn(
                f"Continuation length trimmed from {actual_length} to {trim_to} frames for patch alignment.",
                stacklevel=2,
            )
        cont = cont[:, :trim_to, :]
        mx.eval(cont)
        return cont, trim_to

    def generate_blockwise(
        self,
        *,
        text: str,
        block_sizes: list[int],
        speaker_latents: Any | None = None,
        speaker_audio: Any | None = None,
        speaker_mask: Any | None = None,
        continuation_audio: Any | None = None,
        continuation_latents: Any | None = None,
        seed: int | None = 0,
        num_steps: int = 32,
        cfg_scale_text: float = 3.0,
        cfg_scale_speaker: float = 5.0,
        cfg_min_t: float = 0.5,
        cfg_max_t: float = 1.0,
        truncation_factor: float | None = 0.8,
        speaker_kv_scale: float | None = None,
        speaker_kv_max_layers: int | None = None,
        speaker_kv_min_t: float | None = None,
        trim_latents: bool = True,
        trim_mode: str = "latent",
        return_latents: bool = False,
        on_block_complete: Callable[[int, int, Any], None] | None = None,
        decode_intermediate_blocks: bool = True,
        progress_callback: Callable[[int, int, float, bool], None] | None = None,
    ):
        """Generate waveform from text using sequential blockwise latent sampling."""
        mx = self.mx
        if not self.model.has_blockwise_modules:
            raise RuntimeError(
                "Blockwise generation requires weights converted with --include-blockwise. "
                "Re-run: echo-tts-mlx convert --include-blockwise ..."
            )
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1.")

        patch = int(self.config.speaker_patch_size)
        blocks = _normalize_block_sizes(block_sizes, patch_size=patch)
        total_new_frames = int(sum(blocks))

        use_continuation = continuation_audio is not None or continuation_latents is not None
        if continuation_audio is not None and continuation_latents is not None:
            raise ValueError("Pass either continuation_audio or continuation_latents, not both.")

        continuation = None
        continuation_len = 0
        if use_continuation:
            continuation, continuation_len = self.encode_continuation(
                audio=continuation_audio,
                latents=continuation_latents,
            )

        total_frames = continuation_len + total_new_frames
        if total_frames > int(self.config.max_latent_length):
            raise ValueError(
                f"continuation_length + sum(block_sizes) must be <= {self.config.max_latent_length}, got {total_frames}"
            )

        text_ids, text_mask = self.prepare_text(text)
        use_speaker = speaker_latents is not None or speaker_audio is not None
        if not use_speaker and speaker_mask is not None:
            raise ValueError("speaker_mask was provided without speaker latents/audio.")
        if not use_speaker and speaker_kv_scale is not None:
            raise ValueError("speaker_kv_scale requires speaker conditioning.")

        kv_text_cond, text_k_mask = self.model.get_kv_cache_text(text_ids, text_mask)

        if use_speaker:
            speaker_latents, speaker_mask = self.prepare_speaker_latents(
                speaker_latents=speaker_latents,
                speaker_audio=speaker_audio,
                speaker_mask=speaker_mask,
            )
        else:
            speaker_latents = mx.zeros((1, 4, self.config.latent_size), dtype=mx.float32)
            speaker_mask = mx.zeros((1, 4), dtype=mx.bool_)

        kv_speaker_cond, speaker_k_mask = self.model.get_kv_cache_speaker(speaker_latents, speaker_mask)
        kv_text_full = self._repeat_kv_cache(kv_text_cond, repeats=3)

        def _clone_tensor(x: Any) -> Any:
            copy_fn = getattr(x, "copy", None)
            if callable(copy_fn):
                return copy_fn()
            return x * 1.0

        kv_speaker_cond_unscaled = [(_clone_tensor(k), _clone_tensor(v)) for k, v in kv_speaker_cond]

        text_mask_batch = mx.concatenate(
            [text_k_mask, mx.zeros_like(text_k_mask), text_k_mask],
            axis=0,
        )
        speaker_mask_batch = mx.concatenate(
            [speaker_k_mask, speaker_k_mask, mx.zeros_like(speaker_k_mask)],
            axis=0,
        )

        prefix_latent = mx.zeros((1, total_frames, self.config.latent_size), dtype=mx.float32)
        if continuation is not None:
            prefix_latent[:, :continuation_len, :] = continuation

        if seed is not None:
            mx.random.seed(seed)

        sampler_cfg = BlockwiseSamplerConfig(
            block_sizes=blocks,
            num_steps=num_steps,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_min_t=cfg_min_t,
            cfg_max_t=cfg_max_t,
            truncation_factor=truncation_factor,
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_max_layers=speaker_kv_max_layers,
            speaker_kv_min_t=speaker_kv_min_t,
        )

        kv_speaker_full = self._repeat_kv_cache(kv_speaker_cond, repeats=3)

        def _apply_speaker_scale_for_block() -> None:
            nonlocal kv_speaker_full
            if speaker_kv_scale is None:
                return
            max_layers = len(kv_speaker_cond_unscaled) if speaker_kv_max_layers is None else int(speaker_kv_max_layers)
            scale = float(speaker_kv_scale)
            for i, (k_orig, v_orig) in enumerate(kv_speaker_cond_unscaled):
                if i < max_layers:
                    kv_speaker_cond[i] = (k_orig * scale, v_orig * scale)
                else:
                    kv_speaker_cond[i] = (_clone_tensor(k_orig), _clone_tensor(v_orig))
            kv_speaker_full = self._repeat_kv_cache(kv_speaker_cond, repeats=3)

        def _reverse_speaker_scale() -> None:
            nonlocal kv_speaker_full
            if speaker_kv_scale is None:
                return
            for i, (k_orig, v_orig) in enumerate(kv_speaker_cond_unscaled):
                kv_speaker_cond[i] = (_clone_tensor(k_orig), _clone_tensor(v_orig))
            kv_speaker_full = self._repeat_kv_cache(kv_speaker_cond, repeats=3)

        def _build_latent_kv(full_prefix):
            kv_latent_cond = self.model.get_kv_cache_latent(full_prefix)
            kv_latent_full = self._repeat_kv_cache(kv_latent_cond, repeats=3)
            return kv_latent_full, kv_latent_cond

        def _make_noise(block_size: int):
            return mx.random.normal((1, block_size, self.config.latent_size), dtype=mx.float32)

        def _predict_velocity(x_cur, t: float, cfg_active: bool, start_pos: int, kv_latent_full, kv_latent_cond):
            if cfg_active:
                x_batch = mx.concatenate([x_cur, x_cur, x_cur], axis=0)
                t_batch = mx.array([t, t, t], dtype=x_cur.dtype)
                out = self.model.forward(
                    x_batch,
                    t_batch,
                    kv_text=kv_text_full,
                    kv_speaker=kv_speaker_full,
                    text_mask=text_mask_batch,
                    speaker_mask=speaker_mask_batch,
                    start_pos=start_pos,
                    kv_latent=kv_latent_full,
                )
                v_cond, v_uncond_text, v_uncond_speaker = mx.split(out, 3, axis=0)
                return (
                    v_cond
                    + sampler_cfg.cfg_scale_text * (v_cond - v_uncond_text)
                    + sampler_cfg.cfg_scale_speaker * (v_cond - v_uncond_speaker)
                )

            t_single = mx.array([t], dtype=x_cur.dtype)
            return self.model.forward(
                x_cur,
                t_single,
                kv_text=kv_text_cond,
                kv_speaker=kv_speaker_cond,
                text_mask=text_k_mask,
                speaker_mask=speaker_k_mask,
                start_pos=start_pos,
                kv_latent=kv_latent_cond,
            )

        def _on_block_done(block_idx: int, total_blocks: int, block_latents: Any) -> None:
            if on_block_complete is None:
                return
            if decode_intermediate_blocks:
                block_audio = self.decode_latents(block_latents, trim_latents=False)
                on_block_complete(block_idx, total_blocks, block_audio)
            else:
                on_block_complete(block_idx, total_blocks, block_latents)

        latents = sample_blockwise_euler_cfg(
            prefix_latent=prefix_latent,
            continuation_length=continuation_len,
            config=sampler_cfg,
            make_noise=_make_noise,
            build_latent_kv=_build_latent_kv,
            predict_velocity=_predict_velocity,
            eval_step=lambda arr: mx.eval(arr),
            on_block_start=lambda _idx, _total, _start: _apply_speaker_scale_for_block(),
            on_speaker_kv_scale_reversal=_reverse_speaker_scale,
            on_step=progress_callback,
            on_block_complete=_on_block_done,
        )

        mx.eval(latents)
        audio = self.decode_latents(
            latents,
            trim_latents=trim_latents,
            trim_mode=trim_mode,
        )
        if return_latents:
            return audio, latents
        return audio

    def decode_latents(
        self,
        latents: Any,
        *,
        trim_latents: bool = True,
        trim_mode: str = "latent",
        trim_window_size: int = 20,
        trim_std_threshold: float = 0.05,
        trim_mean_threshold: float = 0.1,
        energy_threshold_db: float = -40.0,
        energy_hop_samples: int = 2048,
        f0_variance_window_s: float = 2.0,
        f0_variance_ratio_threshold: float = 2.0,
        f0_min_voiced_ratio: float = 0.3,
        min_retained_ratio: float = 0.5,
    ):
        """Decode latent tensor `(1, T, 80)` into waveform `(1, 1, samples)`."""
        mx = self.mx
        latents = self._to_mx_array(latents, dtype=mx.float32)
        z_q = self.pca_decode(latents)
        audio = self.autoencoder.decode_zq(z_q)

        if trim_latents:
            trim_mode = _validate_trim_mode(trim_mode)
            mx.eval(latents)
            lat_np = np.asarray(latents, dtype=np.float32)
            if trim_mode == "latent":
                trim_idx = find_flattening_point(
                    lat_np[0],
                    window_size=trim_window_size,
                    std_threshold=trim_std_threshold,
                    mean_threshold=trim_mean_threshold,
                )
            else:
                mx.eval(audio)
                audio_np = np.asarray(audio, dtype=np.float32).reshape(-1)
                trim_idx = find_content_boundary(
                    lat_np[0],
                    audio_np,
                    sample_rate=self.config.sample_rate,
                    ae_downsample_factor=self.config.ae_downsample_factor,
                    latent_window=trim_window_size,
                    latent_std_threshold=trim_std_threshold,
                    latent_mean_threshold=trim_mean_threshold,
                    energy_enabled=True,
                    energy_threshold_db=energy_threshold_db,
                    energy_hop_samples=energy_hop_samples,
                    f0_enabled=(trim_mode == "f0"),
                    f0_variance_window_s=f0_variance_window_s,
                    f0_variance_ratio_threshold=f0_variance_ratio_threshold,
                    min_voiced_ratio=f0_min_voiced_ratio,
                    min_retained_ratio=min_retained_ratio,
                )
            audio = audio[..., : trim_idx * self.config.ae_downsample_factor]

        mx.eval(audio)
        return audio

    def generate(
        self,
        *,
        text: str,
        speaker_latents: Any | None = None,
        speaker_audio: Any | None = None,
        speaker_mask: Any | None = None,
        noise: Any | None = None,
        sequence_length: int | None = None,
        seed: int | None = 0,
        num_steps: int = 32,
        cfg_scale_text: float = 3.0,
        cfg_scale_speaker: float = 8.0,
        cfg_min_t: float = 0.5,
        cfg_max_t: float = 1.0,
        truncation_factor: float | None = 0.8,
        speaker_kv_scale: float | None = None,
        speaker_kv_max_layers: int | None = None,
        speaker_kv_min_t: float | None = None,
        trim_latents: bool = True,
        trim_mode: str = "latent",
        return_latents: bool = False,
        progress_callback: Callable[[int, int, float, bool], None] | None = None,
    ):
        """Generate waveform tensor from text and a single speaker reference."""
        latents = self.generate_latents(
            text=text,
            speaker_latents=speaker_latents,
            speaker_audio=speaker_audio,
            speaker_mask=speaker_mask,
            noise=noise,
            sequence_length=sequence_length,
            seed=seed,
            num_steps=num_steps,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_min_t=cfg_min_t,
            cfg_max_t=cfg_max_t,
            truncation_factor=truncation_factor,
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_max_layers=speaker_kv_max_layers,
            speaker_kv_min_t=speaker_kv_min_t,
            progress_callback=progress_callback,
        )
        audio = self.decode_latents(
            latents,
            trim_latents=trim_latents,
            trim_mode=trim_mode,
        )
        if return_latents:
            return audio, latents
        return audio

    def save_audio(self, audio: Any, path: str | Path):
        """Save waveform to disk at the model sample rate."""
        return save_audio_file(path, audio, sample_rate=self.config.sample_rate)
