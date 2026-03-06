"""Core EchoDiT model (MLX inference path).

This module intentionally supports only converted checkpoints from
`weights/converted/` and does not include upstream/HF loading fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file as save_file_np

from .config import ModelConfig, load_model_config, resolve_converted_paths


NEG_INF = -1e4
VALID_QUANTIZE_MODES = {"none", "8bit", "4bit", "mxfp4", "mixed"}
QUANTIZE_CONFIG_FILENAME = "quantize_config.json"
_LEGACY_QUANTIZE_CONFIG_FILENAMES = ("quantizeconfig.json",)

_QUANTIZE_SKIP = {
    "in_proj",
    "out_proj",
    "out_norm",
    "text_norm",
    "speaker_norm",
    "latent_norm",
    "text_embedding",
    "k_norm",
    "q_norm",
    "attention_norm",
    "mlp_norm",
}
_QUANTIZE_SKIP_SUFFIXES = {"_down"}
# Initial conservative estimate — refine with layer_sensitivity_sweep.py
SENSITIVE_MODULES = {
    f"blocks.{block}.attention.{name}"
    for block in range(18, 24)
    for name in ("wq", "wk", "wv", "wo", "gate")
}


@dataclass(frozen=True)
class QuantizeConfig:
    mode: str
    bits: int
    group_size: int
    quantized_modules: list[str]
    per_module: bool = False
    modules: dict[str, dict[str, int | str]] | None = None


def _normalize_quantize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_QUANTIZE_MODES:
        raise ValueError(f"Invalid quantize mode '{mode}'. Expected one of: none, 8bit, 4bit, mxfp4, mixed.")
    return normalized


def _quantize_config_candidates(weights_dir: Path) -> list[Path]:
    return [weights_dir / QUANTIZE_CONFIG_FILENAME] + [weights_dir / name for name in _LEGACY_QUANTIZE_CONFIG_FILENAMES]


def _quantize_config_path(weights_dir: str | Path) -> Path:
    root = Path(weights_dir)
    for candidate in _quantize_config_candidates(root):
        if candidate.exists():
            return candidate
    return root / QUANTIZE_CONFIG_FILENAME


def detect_quantize_config(weights_dir: str | Path) -> bool:
    return _quantize_config_path(weights_dir).exists()


def load_quantize_config(weights_dir: str | Path) -> QuantizeConfig:
    cfg_path = _quantize_config_path(weights_dir)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing quantize config: {cfg_path}")

    payload = json.loads(cfg_path.read_text())
    required = {"mode", "bits", "group_size", "quantized_modules"}
    if not required.issubset(payload):
        missing = sorted(required - set(payload))
        raise ValueError(f"Invalid quantize config: missing keys {', '.join(missing)}")

    mode = _normalize_quantize_mode(str(payload["mode"]))
    bits = int(payload["bits"])
    group_size = int(payload["group_size"])
    quantized_modules_raw = payload["quantized_modules"]
    if not isinstance(quantized_modules_raw, list) or not all(isinstance(x, str) for x in quantized_modules_raw):
        raise ValueError("Invalid quantize config: quantized_modules must be a list[str].")

    per_module = bool(payload.get("per_module", False))
    modules_raw = payload.get("modules")
    modules: dict[str, dict[str, int | str]] | None = None
    if per_module:
        if not isinstance(modules_raw, dict):
            raise ValueError("Invalid quantize config: modules must be a dict when per_module=true.")
        modules = {}
        for name, spec in modules_raw.items():
            if not isinstance(name, str) or not isinstance(spec, dict):
                raise ValueError("Invalid quantize config: modules must map str -> dict.")
            if not {"bits", "group_size", "mode"}.issubset(spec):
                raise ValueError("Invalid quantize config: module specs require bits/group_size/mode.")
            modules[name] = {
                "bits": int(spec["bits"]),
                "group_size": int(spec["group_size"]),
                "mode": str(spec["mode"]).strip().lower(),
            }

    return QuantizeConfig(
        mode=mode,
        bits=bits,
        group_size=group_size,
        quantized_modules=list(quantized_modules_raw),
        per_module=per_module,
        modules=modules,
    )


def save_quantize_config(weights_dir: str | Path, config: QuantizeConfig) -> Path:
    mode = _normalize_quantize_mode(config.mode)
    root = Path(weights_dir)
    root.mkdir(parents=True, exist_ok=True)
    cfg_path = root / QUANTIZE_CONFIG_FILENAME
    payload: dict[str, Any] = {
        "mode": mode,
        "bits": int(config.bits),
        "group_size": int(config.group_size),
        "quantized_modules": list(dict.fromkeys(config.quantized_modules)),
    }
    if bool(config.per_module):
        payload["per_module"] = True
        payload["modules"] = dict(config.modules or {})
    cfg_path.write_text(json.dumps(payload, indent=2) + "\n")
    return cfg_path


def _require_mlx():
    """Lazily import MLX to keep this module import-safe on non-MLX hosts."""
    try:
        return importlib.import_module("mlx.core")
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MLX is required for EchoDiT inference.") from exc


def _require_mlx_nn():
    try:
        return importlib.import_module("mlx.nn")
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MLX is required for EchoDiT inference.") from exc


def _is_mlx_array(x: Any) -> bool:
    return x.__class__.__module__.startswith("mlx")


def _to_mx_array(x: Any, mx: Any) -> Any:
    return x if _is_mlx_array(x) else mx.array(x)


def _load_dit_state(weights_path: Path, dtype_name: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    float_dtype = np.dtype(dtype_name)
    with safe_open(str(weights_path), framework="np") as f:
        for key in f.keys():
            arr = np.asarray(f.get_tensor(key))
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(float_dtype)
            out[key] = arr
    return out


def _silu(x: Any, mx: Any) -> Any:
    return x * (1.0 / (1.0 + mx.exp(-x)))


def _sigmoid(x: Any, mx: Any) -> Any:
    return 1.0 / (1.0 + mx.exp(-x))


def _linear(x: Any, weight: Any, mx: Any, bias: Any | None = None) -> Any:
    # Ensure weight matches x dtype to avoid mixed-precision matmul surprises
    if weight.dtype != x.dtype:
        weight = weight.astype(x.dtype)
    y = x @ mx.transpose(weight)
    if bias is not None:
        if bias.dtype != y.dtype:
            bias = bias.astype(y.dtype)
        y = y + bias
    return y


def _rms_norm(x: Any, weight: Any, mx: Any, eps: float) -> Any:
    if weight.dtype != x.dtype:
        weight = weight.astype(x.dtype)
    return mx.fast.rms_norm(x, weight, eps)


def _rms_norm_head(x: Any, weight: Any, mx: Any, eps: float) -> Any:
    # x: (B, H, T, D), weight: (H, D)
    orig_dtype = x.dtype
    x = x.astype(mx.float32)
    norm = 1.0 / mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x * norm).astype(orig_dtype) * mx.reshape(weight, (1, weight.shape[0], 1, weight.shape[1]))


def _swiglu(x: Any, w1: Any, w2: Any, w3: Any, mx: Any) -> Any:
    # Compute in float32 to avoid fp16 overflow in the intermediate 5888-dim space
    orig_dtype = x.dtype
    x = x.astype(mx.float32)
    w1 = w1.astype(mx.float32)
    w2 = w2.astype(mx.float32)
    w3 = w3.astype(mx.float32)
    out = _linear(_silu(_linear(x, w1, mx), mx) * _linear(x, w3, mx), w2, mx)
    return out.astype(orig_dtype)


def _timestep_embedding(t: Any, dim: int, mx: Any, dtype: Any, theta: float = 10000.0) -> Any:
    # t: (B,)
    half = dim // 2
    freqs = 1000.0 * np.exp(-math.log(theta) * np.arange(half, dtype=np.float32) / half)
    freqs = mx.array(freqs, dtype=dtype)
    args = mx.reshape(t.astype(dtype), (-1, 1)) * mx.reshape(freqs, (1, -1))
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2 == 1:
        emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=dtype)], axis=-1)
    return emb


def _apply_rotary(x: Any, mx: Any, offset: int = 0) -> Any:
    # x: (B, H, T, D)
    return mx.fast.rope(
        x,
        dims=int(x.shape[-1]),
        traditional=True,
        base=10000.0,
        scale=1.0,
        offset=int(offset),
    )


def _apply_half_rotary(x: Any, mx: Any, offset: int = 0) -> Any:
    # x: (B, H, T, D), rotate only first half of heads.
    h = int(x.shape[1])
    h_half = h // 2
    if h_half == 0:
        return x
    x_rot = _apply_rotary(x[:, :h_half, :, :], mx, offset=offset)
    return mx.concatenate([x_rot, x[:, h_half:, :, :]], axis=1)


def _apply_rotary_at_positions(x: Any, positions: Any, mx: Any, theta: float = 10000.0) -> Any:
    """Apply RoPE at arbitrary (including non-contiguous) positions.

    This is used by blockwise latent KV cache construction, where position
    indices follow latent patch boundaries (e.g., [0, 4, 8, ...] when
    `speaker_patch_size=4`). `mx.fast.rope` only supports contiguous indices
    via scalar offset, so this helper computes rotation directly.

    The implementation uses split-half layout (equivalent to
    `mx.fast.rope(..., traditional=False)` for contiguous positions).
    """
    # x: (B, H, T, D), positions: (T,)
    d = int(x.shape[-1])
    half_d = d // 2
    if half_d == 0:
        return x

    freqs = 1.0 / (theta ** (mx.arange(0, half_d, dtype=mx.float32) / float(half_d)))
    angles = mx.reshape(positions.astype(mx.float32), (-1, 1)) * mx.reshape(freqs, (1, -1))
    cos_vals = mx.reshape(mx.cos(angles), (1, 1, int(angles.shape[0]), half_d))
    sin_vals = mx.reshape(mx.sin(angles), (1, 1, int(angles.shape[0]), half_d))

    x1 = x[..., :half_d]
    x2 = x[..., half_d : half_d * 2]
    rot = mx.concatenate(
        [x1 * cos_vals - x2 * sin_vals, x1 * sin_vals + x2 * cos_vals],
        axis=-1,
    )
    if d == half_d * 2:
        return rot
    return mx.concatenate([rot, x[..., half_d * 2 :]], axis=-1)


def _apply_half_rotary_at_positions(x: Any, positions: Any, mx: Any) -> Any:
    """Half-head rotary wrapper for `_apply_rotary_at_positions`."""
    # x: (B, H, T, D), rotate only first half of heads.
    h = int(x.shape[1])
    h_half = h // 2
    if h_half == 0:
        return x
    x_rot = _apply_rotary_at_positions(x[:, :h_half, :, :], positions, mx)
    return mx.concatenate([x_rot, x[:, h_half:, :, :]], axis=1)


def _ensure_mask(mask: Any | None, *, batch: int, length: int, mx: Any) -> Any:
    if mask is None:
        return mx.ones((batch, length), dtype=mx.bool_)
    m = _to_mx_array(mask, mx)
    if m.ndim != 2:
        raise ValueError(f"Mask must be rank-2 (B, T), got shape {tuple(m.shape)}")
    if int(m.shape[0]) != batch or int(m.shape[1]) != length:
        raise ValueError(f"Mask shape mismatch: expected {(batch, length)}, got {tuple(m.shape)}")
    return m.astype(mx.bool_)


def _make_causal_mask(length: int, dtype: Any, mx: Any) -> Any:
    mask = np.triu(np.ones((length, length), dtype=np.float32), k=1)
    return mx.array(mask, dtype=dtype)


def _reshape_heads(x: Any, num_heads: int, mx: Any) -> Any:
    # (B, T, C) -> (B, H, T, D)
    b, t, c = x.shape
    d = c // num_heads
    x = mx.reshape(x, (b, t, num_heads, d))
    return mx.transpose(x, (0, 2, 1, 3))


def _merge_heads(x: Any, mx: Any) -> Any:
    # (B, H, T, D) -> (B, T, C)
    b, h, t, d = x.shape
    x = mx.transpose(x, (0, 2, 1, 3))
    return mx.reshape(x, (b, t, h * d))


class MlxEchoDiT:
    """Inference-only EchoDiT core."""

    def __init__(self, config: ModelConfig, *, dtype: str = "float16") -> None:
        mx = _require_mlx()
        nn = _require_mlx_nn()
        self.mx = mx
        self.nn = nn
        self.config = config
        self.dtype = getattr(mx, dtype)
        self.quantize_mode = "none"
        self._quantized_modules: dict[str, int] = {}
        self._quantized_module_params: dict[str, dict[str, int | str]] = {}
        self._all_checkpoint_keys: list[str] = []
        self._has_blockwise_modules = False

        self.tree = self._build_module_tree()

        self.head_dim = self.config.model_size // self.config.num_heads
        self.text_head_dim = self.config.text_model_size // self.config.text_num_heads
        self.speaker_head_dim = self.config.speaker_model_size // self.config.speaker_num_heads
        self._compiled_swiglu = mx.compile(lambda x, w1, w2, w3: _swiglu(x, w1, w2, w3, mx)) if hasattr(mx, "compile") else None

    @property
    def has_blockwise_modules(self) -> bool:
        return bool(self._has_blockwise_modules)

    @classmethod
    def from_pretrained(
        cls,
        weights_dir: str | Path = "weights/converted",
        *,
        dtype: str = "float16",
        quantize: str = "none",
    ) -> "MlxEchoDiT":
        config_path, dit_path = resolve_converted_paths(weights_dir)
        config = load_model_config(config_path)
        if config.model_type != "echo-dit":
            raise ValueError(f"Unsupported model_type in config: {config.model_type}")

        requested_mode = _normalize_quantize_mode(quantize)
        prequantized_cfg: QuantizeConfig | None = None
        if detect_quantize_config(weights_dir):
            prequantized_cfg = load_quantize_config(weights_dir)
            if requested_mode != "none" and requested_mode != prequantized_cfg.mode:
                raise ValueError(
                    f"Requested quantize='{requested_mode}' but checkpoint was saved as '{prequantized_cfg.mode}'."
                )
            requested_mode = prequantized_cfg.mode

        model = cls(config=config, dtype=dtype)
        if prequantized_cfg is not None:
            model._load_tree_weights(dit_path)
            model.quantize_mode = requested_mode
            model._quantized_module_params = {}
            if prequantized_cfg.per_module and prequantized_cfg.modules:
                for name, spec in prequantized_cfg.modules.items():
                    model._quantized_module_params[name] = {
                        "bits": int(spec["bits"]),
                        "group_size": int(spec["group_size"]),
                        "mode": str(spec["mode"]),
                    }
                model._quantized_modules = {
                    name: int(spec["bits"])
                    for name, spec in model._quantized_module_params.items()
                }
            elif requested_mode == "4bit":
                model._quantized_modules = {
                    name: (4 if name.startswith("blocks.") or name.startswith("cond_module.") else 8)
                    for name in prequantized_cfg.quantized_modules
                }
            elif requested_mode in {"8bit", "mixed"}:
                model._quantized_modules = {name: 8 for name in prequantized_cfg.quantized_modules}
            else:
                model._quantized_modules = {name: 4 for name in prequantized_cfg.quantized_modules}
            return model

        state = _load_dit_state(dit_path, dtype_name=dtype)
        model._load_state_dict(state)
        if requested_mode != "none":
            model.apply_quantization(mode=requested_mode, group_size=64)
        return model

    def t(self, key: str) -> Any:
        module_path, field = key.rsplit(".", 1)
        obj = self._resolve_path(module_path)
        if not hasattr(obj, field):
            raise KeyError(f"Missing checkpoint key: {key}")
        return getattr(obj, field)

    def _resolve_path(self, path: str) -> Any:
        obj: Any = self.tree
        if not path:
            return obj
        for part in path.split("."):
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    def _build_module_tree(self) -> Any:
        mx = self.mx
        nn = self.nn
        cfg = self.config
        dtype = self.dtype

        class Node(nn.Module):
            def __init__(self) -> None:
                super().__init__()

        class WeightNode(nn.Module):
            def __init__(self, shape: tuple[int, ...]) -> None:
                super().__init__()
                self.weight = mx.zeros(shape, dtype=dtype)

        class IdentityNode(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def __call__(self, x: Any) -> Any:
                return x

        root = Node()

        root.in_proj = nn.Linear(cfg.latent_size, cfg.model_size, bias=True)
        root.out_proj = nn.Linear(cfg.model_size, cfg.latent_size, bias=True)
        root.out_norm = WeightNode((cfg.model_size,))

        root.cond_module = [
            nn.Linear(cfg.timestep_embed_size, cfg.model_size, bias=False),
            IdentityNode(),
            nn.Linear(cfg.model_size, cfg.model_size, bias=False),
            IdentityNode(),
            nn.Linear(cfg.model_size, cfg.model_size * 3, bias=False),
        ]

        root.blocks = []
        for _ in range(cfg.num_layers):
            blk = Node()
            blk.attention = Node()
            blk.attention.wq = nn.Linear(cfg.model_size, cfg.model_size, bias=False)
            blk.attention.wk = nn.Linear(cfg.model_size, cfg.model_size, bias=False)
            blk.attention.wv = nn.Linear(cfg.model_size, cfg.model_size, bias=False)
            blk.attention.wo = nn.Linear(cfg.model_size, cfg.model_size, bias=False)
            blk.attention.gate = nn.Linear(cfg.model_size, cfg.model_size, bias=False)
            blk.attention.wk_text = nn.Linear(cfg.text_model_size, cfg.model_size, bias=False)
            blk.attention.wv_text = nn.Linear(cfg.text_model_size, cfg.model_size, bias=False)
            blk.attention.wk_latent = nn.Linear(cfg.speaker_model_size, cfg.model_size, bias=False)
            blk.attention.wv_latent = nn.Linear(cfg.speaker_model_size, cfg.model_size, bias=False)
            blk.attention.wk_speaker = nn.Linear(cfg.speaker_model_size, cfg.model_size, bias=False)
            blk.attention.wv_speaker = nn.Linear(cfg.speaker_model_size, cfg.model_size, bias=False)
            blk.attention.q_norm = WeightNode((cfg.num_heads, cfg.model_size // cfg.num_heads))
            blk.attention.k_norm = WeightNode((cfg.num_heads, cfg.model_size // cfg.num_heads))

            blk.mlp = Node()
            blk.mlp.w1 = nn.Linear(cfg.model_size, cfg.intermediate_size, bias=False)
            blk.mlp.w2 = nn.Linear(cfg.intermediate_size, cfg.model_size, bias=False)
            blk.mlp.w3 = nn.Linear(cfg.model_size, cfg.intermediate_size, bias=False)

            blk.attention_adaln = Node()
            blk.attention_adaln.shift_down = nn.Linear(cfg.model_size, cfg.adaln_rank, bias=False)
            blk.attention_adaln.shift_up = nn.Linear(cfg.adaln_rank, cfg.model_size, bias=True)
            blk.attention_adaln.scale_down = nn.Linear(cfg.model_size, cfg.adaln_rank, bias=False)
            blk.attention_adaln.scale_up = nn.Linear(cfg.adaln_rank, cfg.model_size, bias=True)
            blk.attention_adaln.gate_down = nn.Linear(cfg.model_size, cfg.adaln_rank, bias=False)
            blk.attention_adaln.gate_up = nn.Linear(cfg.adaln_rank, cfg.model_size, bias=True)

            blk.mlp_adaln = Node()
            blk.mlp_adaln.shift_down = nn.Linear(cfg.model_size, cfg.adaln_rank, bias=False)
            blk.mlp_adaln.shift_up = nn.Linear(cfg.adaln_rank, cfg.model_size, bias=True)
            blk.mlp_adaln.scale_down = nn.Linear(cfg.model_size, cfg.adaln_rank, bias=False)
            blk.mlp_adaln.scale_up = nn.Linear(cfg.adaln_rank, cfg.model_size, bias=True)
            blk.mlp_adaln.gate_down = nn.Linear(cfg.model_size, cfg.adaln_rank, bias=False)
            blk.mlp_adaln.gate_up = nn.Linear(cfg.adaln_rank, cfg.model_size, bias=True)
            root.blocks.append(blk)

        root.text_encoder = Node()
        root.text_encoder.text_embedding = nn.Embedding(cfg.text_vocab_size, cfg.text_model_size)
        root.text_encoder.blocks = []
        for _ in range(cfg.text_num_layers):
            blk = Node()
            blk.attention = Node()
            blk.attention.wq = nn.Linear(cfg.text_model_size, cfg.text_model_size, bias=False)
            blk.attention.wk = nn.Linear(cfg.text_model_size, cfg.text_model_size, bias=False)
            blk.attention.wv = nn.Linear(cfg.text_model_size, cfg.text_model_size, bias=False)
            blk.attention.wo = nn.Linear(cfg.text_model_size, cfg.text_model_size, bias=False)
            blk.attention.gate = nn.Linear(cfg.text_model_size, cfg.text_model_size, bias=False)
            blk.attention.q_norm = WeightNode((cfg.text_num_heads, cfg.text_model_size // cfg.text_num_heads))
            blk.attention.k_norm = WeightNode((cfg.text_num_heads, cfg.text_model_size // cfg.text_num_heads))
            blk.attention_norm = WeightNode((cfg.text_model_size,))
            blk.mlp_norm = WeightNode((cfg.text_model_size,))
            blk.mlp = Node()
            blk.mlp.w1 = nn.Linear(cfg.text_model_size, cfg.text_intermediate_size, bias=False)
            blk.mlp.w2 = nn.Linear(cfg.text_intermediate_size, cfg.text_model_size, bias=False)
            blk.mlp.w3 = nn.Linear(cfg.text_model_size, cfg.text_intermediate_size, bias=False)
            root.text_encoder.blocks.append(blk)
        root.text_norm = WeightNode((cfg.text_model_size,))

        root.speaker_encoder = Node()
        root.speaker_encoder.in_proj = nn.Linear(cfg.speaker_patch_size * cfg.latent_size, cfg.speaker_model_size, bias=True)
        root.speaker_encoder.blocks = []
        for _ in range(cfg.speaker_num_layers):
            blk = Node()
            blk.attention = Node()
            blk.attention.wq = nn.Linear(cfg.speaker_model_size, cfg.speaker_model_size, bias=False)
            blk.attention.wk = nn.Linear(cfg.speaker_model_size, cfg.speaker_model_size, bias=False)
            blk.attention.wv = nn.Linear(cfg.speaker_model_size, cfg.speaker_model_size, bias=False)
            blk.attention.wo = nn.Linear(cfg.speaker_model_size, cfg.speaker_model_size, bias=False)
            blk.attention.gate = nn.Linear(cfg.speaker_model_size, cfg.speaker_model_size, bias=False)
            blk.attention.q_norm = WeightNode((cfg.speaker_num_heads, cfg.speaker_model_size // cfg.speaker_num_heads))
            blk.attention.k_norm = WeightNode((cfg.speaker_num_heads, cfg.speaker_model_size // cfg.speaker_num_heads))
            blk.attention_norm = WeightNode((cfg.speaker_model_size,))
            blk.mlp_norm = WeightNode((cfg.speaker_model_size,))
            blk.mlp = Node()
            blk.mlp.w1 = nn.Linear(cfg.speaker_model_size, cfg.speaker_intermediate_size, bias=False)
            blk.mlp.w2 = nn.Linear(cfg.speaker_intermediate_size, cfg.speaker_model_size, bias=False)
            blk.mlp.w3 = nn.Linear(cfg.speaker_model_size, cfg.speaker_intermediate_size, bias=False)
            root.speaker_encoder.blocks.append(blk)
        root.speaker_norm = WeightNode((cfg.speaker_model_size,))

        root.latent_encoder = Node()
        root.latent_encoder.in_proj = nn.Linear(cfg.speaker_patch_size * cfg.latent_size, cfg.speaker_model_size, bias=True)
        root.latent_encoder.blocks = []
        for _ in range(cfg.speaker_num_layers):
            blk = Node()
            blk.attention = Node()
            blk.attention.wq = nn.Linear(cfg.speaker_model_size, cfg.speaker_model_size, bias=False)
            blk.attention.wk = nn.Linear(cfg.speaker_model_size, cfg.speaker_model_size, bias=False)
            blk.attention.wv = nn.Linear(cfg.speaker_model_size, cfg.speaker_model_size, bias=False)
            blk.attention.wo = nn.Linear(cfg.speaker_model_size, cfg.speaker_model_size, bias=False)
            blk.attention.gate = nn.Linear(cfg.speaker_model_size, cfg.speaker_model_size, bias=False)
            blk.attention.q_norm = WeightNode((cfg.speaker_num_heads, cfg.speaker_model_size // cfg.speaker_num_heads))
            blk.attention.k_norm = WeightNode((cfg.speaker_num_heads, cfg.speaker_model_size // cfg.speaker_num_heads))
            blk.attention_norm = WeightNode((cfg.speaker_model_size,))
            blk.mlp_norm = WeightNode((cfg.speaker_model_size,))
            blk.mlp = Node()
            blk.mlp.w1 = nn.Linear(cfg.speaker_model_size, cfg.speaker_intermediate_size, bias=False)
            blk.mlp.w2 = nn.Linear(cfg.speaker_intermediate_size, cfg.speaker_model_size, bias=False)
            blk.mlp.w3 = nn.Linear(cfg.speaker_model_size, cfg.speaker_intermediate_size, bias=False)
            root.latent_encoder.blocks.append(blk)
        root.latent_norm = WeightNode((cfg.speaker_model_size,))

        return root

    def _load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        mx = self.mx
        self._all_checkpoint_keys = sorted(state.keys())
        self._has_blockwise_modules = self._detect_blockwise_modules(self._all_checkpoint_keys)
        for key, value in state.items():
            module_path, field = key.rsplit(".", 1)
            obj = self._resolve_path(module_path)
            setattr(obj, field, mx.array(value))

    def _load_tree_weights(self, weights_path: Path) -> None:
        with safe_open(str(weights_path), framework="np") as f:
            self._all_checkpoint_keys = sorted(f.keys())
        self._has_blockwise_modules = self._detect_blockwise_modules(self._all_checkpoint_keys)
        if hasattr(self.tree, "load_weights"):
            self.tree.load_weights(str(weights_path))
            return
        state = _load_dit_state(weights_path, dtype_name="float16")
        self._load_state_dict(state)

    def _detect_blockwise_modules(self, keys: list[str]) -> bool:
        required = (
            "latent_encoder.in_proj.weight",
            "latent_norm.weight",
            "blocks.0.attention.wk_latent.weight",
            "blocks.0.attention.wv_latent.weight",
        )
        key_set = set(keys)
        return all(k in key_set for k in required)

    def save_weights(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.tree, "save_weights"):
            self.tree.save_weights(str(out))
            return out

        if not self._all_checkpoint_keys:
            raise RuntimeError("No checkpoint keys are tracked for fallback save.")
        payload = {k: np.asarray(self.t(k)) for k in self._all_checkpoint_keys}
        save_file_np(payload, str(out))
        return out

    def _apply_linear_path(self, x: Any, path: str, *, bias: bool = False) -> Any:
        mod = self._resolve_path(path)
        if self.quantize_mode != "none":
            return mod(x)
        b = getattr(mod, "bias", None) if bias else None
        return _linear(x, mod.weight, self.mx, b)

    def _swiglu_paths(self, x: Any, *, w1_path: str, w2_path: str, w3_path: str) -> Any:
        if self.quantize_mode == "none":
            w1 = self._resolve_path(w1_path).weight
            w2 = self._resolve_path(w2_path).weight
            w3 = self._resolve_path(w3_path).weight
            if self._compiled_swiglu is not None:
                return self._compiled_swiglu(x, w1, w2, w3)
            return _swiglu(x, w1, w2, w3, self.mx)

        mx = self.mx
        orig_dtype = x.dtype
        x32 = x.astype(mx.float32)
        h1 = self._apply_linear_path(x32, w1_path)
        h3 = self._apply_linear_path(x32, w3_path)
        out = self._apply_linear_path(_silu(h1, mx) * h3, w2_path)
        return out.astype(orig_dtype)

    def _build_self_attention_mask(self, *, t: int, key_mask: Any | None, causal: bool, dtype: Any) -> Any | str | None:
        mx = self.mx
        if not causal and key_mask is None:
            return None
        if causal and key_mask is None:
            return "causal"

        mask = mx.zeros((1, 1, t, t), dtype=dtype)
        if causal:
            causal_mask = _make_causal_mask(t, dtype, mx)
            mask = mask + mx.reshape(causal_mask * NEG_INF, (1, 1, t, t))

        if key_mask is not None:
            b = int(key_mask.shape[0])
            inv = 1.0 - key_mask.astype(dtype)
            key_additive = mx.reshape(inv * NEG_INF, (b, 1, 1, t))
            mask = mask + mx.broadcast_to(key_additive, (b, 1, t, t))
        return mask

    def _build_joint_attention_mask(
        self,
        *,
        batch: int,
        t_query: int,
        t_latent: int,
        t_text: int,
        t_speaker: int,
        latent_mask: Any | None,
        text_mask: Any | None,
        speaker_mask: Any | None,
        dtype: Any,
    ) -> Any | None:
        mx = self.mx

        if latent_mask is None and text_mask is None and speaker_mask is None:
            return None

        parts: list[Any] = [mx.zeros((batch, 1, t_query, t_query), dtype=dtype)]

        if t_latent > 0:
            if latent_mask is None:
                parts.append(mx.zeros((batch, 1, t_query, t_latent), dtype=dtype))
            else:
                if int(latent_mask.shape[1]) != t_latent:
                    raise ValueError(
                        f"latent_mask shape mismatch: expected (_, {t_latent}), got {tuple(latent_mask.shape)}"
                    )
                inv_latent = (1.0 - latent_mask.astype(dtype)) * NEG_INF
                latent_additive = mx.reshape(inv_latent, (batch, 1, 1, t_latent))
                parts.append(mx.broadcast_to(latent_additive, (batch, 1, t_query, t_latent)))

        if t_text > 0:
            if text_mask is None:
                parts.append(mx.zeros((batch, 1, t_query, t_text), dtype=dtype))
            else:
                if int(text_mask.shape[1]) != t_text:
                    raise ValueError(f"text_mask shape mismatch: expected (_, {t_text}), got {tuple(text_mask.shape)}")
                inv_text = (1.0 - text_mask.astype(dtype)) * NEG_INF
                text_additive = mx.reshape(inv_text, (batch, 1, 1, t_text))
                parts.append(mx.broadcast_to(text_additive, (batch, 1, t_query, t_text)))

        if t_speaker > 0:
            if speaker_mask is None:
                parts.append(mx.zeros((batch, 1, t_query, t_speaker), dtype=dtype))
            else:
                if int(speaker_mask.shape[1]) != t_speaker:
                    raise ValueError(
                        f"speaker_mask shape mismatch: expected (_, {t_speaker}), got {tuple(speaker_mask.shape)}"
                    )
                inv_speaker = (1.0 - speaker_mask.astype(dtype)) * NEG_INF
                speaker_additive = mx.reshape(inv_speaker, (batch, 1, 1, t_speaker))
                parts.append(mx.broadcast_to(speaker_additive, (batch, 1, t_query, t_speaker)))

        return mx.concatenate(parts, axis=-1)

    def _quantize_predicate(self, path: str, module: Any) -> bool:
        parts = [p for p in path.split(".") if p]
        if any(p in _QUANTIZE_SKIP for p in parts):
            return False
        if parts and any(parts[-1].endswith(suffix) for suffix in _QUANTIZE_SKIP_SUFFIXES):
            return False
        return isinstance(module, self.nn.Linear)

    def apply_quantization(self, *, mode: str, group_size: int = 64) -> None:
        mode = _normalize_quantize_mode(mode)
        if mode == "none":
            self.quantize_mode = "none"
            self._quantized_modules = {}
            self._quantized_module_params = {}
            return

        selected: dict[str, int] = {}
        selected_params: dict[str, dict[str, int | str]] = {}

        if mode == "8bit":
            def pred(path: str, module: Any) -> bool:
                keep = self._quantize_predicate(path, module)
                if keep:
                    selected[path] = 8
                    selected_params[path] = {"bits": 8, "group_size": int(group_size), "mode": "affine"}
                return keep

            self.nn.quantize(self.tree, group_size=group_size, bits=8, class_predicate=pred)

        elif mode == "4bit":
            def pred_dit(path: str, module: Any) -> bool:
                keep = self._quantize_predicate(path, module) and (
                    path.startswith("blocks.") or path.startswith("cond_module.")
                )
                if keep:
                    selected[path] = 4
                    selected_params[path] = {"bits": 4, "group_size": int(group_size), "mode": "affine"}
                return keep

            def pred_encoder(path: str, module: Any) -> bool:
                keep = self._quantize_predicate(path, module) and (
                    path.startswith("text_encoder.")
                    or path.startswith("speaker_encoder.")
                    or path.startswith("latent_encoder.")
                )
                if keep:
                    selected[path] = 8
                    selected_params[path] = {"bits": 8, "group_size": int(group_size), "mode": "affine"}
                return keep

            self.nn.quantize(self.tree, group_size=group_size, bits=4, class_predicate=pred_dit)
            self.nn.quantize(self.tree, group_size=group_size, bits=8, class_predicate=pred_encoder)

        elif mode == "mxfp4":
            if int(group_size) != 32:
                warnings.warn("MXFP4 requires group_size=32, ignoring provided value", stacklevel=2)

            def pred(path: str, module: Any) -> bool:
                keep = self._quantize_predicate(path, module)
                if keep:
                    selected[path] = 4
                    selected_params[path] = {"bits": 4, "group_size": 32, "mode": "mxfp4"}
                return keep

            self.nn.quantize(self.tree, group_size=32, bits=4, mode="mxfp4", class_predicate=pred)

        else:
            def pred(path: str, module: Any) -> bool | dict[str, int | str]:
                if not self._quantize_predicate(path, module):
                    return False

                if (
                    path in SENSITIVE_MODULES
                    or path.startswith("text_encoder.")
                    or path.startswith("speaker_encoder.")
                    or path.startswith("latent_encoder.")
                ):
                    spec = {"bits": 8, "group_size": 64, "mode": "affine"}
                else:
                    spec = {"bits": 4, "group_size": 32, "mode": "mxfp4"}

                selected[path] = int(spec["bits"])
                selected_params[path] = dict(spec)
                return spec

            self.nn.quantize(self.tree, class_predicate=pred)

        self.quantize_mode = mode
        self._quantized_modules = selected
        self._quantized_module_params = selected_params

    def quantize_config(self, *, group_size: int = 64) -> QuantizeConfig:
        if self.quantize_mode == "none":
            raise ValueError("Model is not quantized.")
        if self.quantize_mode == "8bit":
            bits = 8
            gs = int(group_size)
        elif self.quantize_mode == "4bit":
            bits = 4
            gs = int(group_size)
        elif self.quantize_mode == "mxfp4":
            bits = 4
            gs = 32
        else:
            bits = 8
            gs = 64

        per_module = self.quantize_mode == "mixed"
        modules = dict(self._quantized_module_params) if per_module else None
        return QuantizeConfig(
            mode=self.quantize_mode,
            bits=bits,
            group_size=gs,
            quantized_modules=sorted(self._quantized_modules.keys()),
            per_module=per_module,
            modules=modules,
        )

    def _self_attention(
        self,
        x: Any,
        *,
        prefix: str,
        num_heads: int,
        causal: bool,
        key_mask: Any | None,
        query_mask: Any | None,
        half_rotary: bool,
    ) -> Any:
        mx = self.mx
        b, t, _ = x.shape

        q = self._apply_linear_path(x, f"{prefix}.wq")
        k = self._apply_linear_path(x, f"{prefix}.wk")
        v = self._apply_linear_path(x, f"{prefix}.wv")

        q = _reshape_heads(q, num_heads, mx)
        k = _reshape_heads(k, num_heads, mx)
        v = _reshape_heads(v, num_heads, mx)

        q = _rms_norm_head(q, self.t(f"{prefix}.q_norm.weight"), mx, self.config.norm_eps)
        k = _rms_norm_head(k, self.t(f"{prefix}.k_norm.weight"), mx, self.config.norm_eps)

        if half_rotary:
            q = _apply_half_rotary(q, mx)
            k = _apply_half_rotary(k, mx)
        else:
            q = _apply_rotary(q, mx)
            k = _apply_rotary(k, mx)

        mask = self._build_self_attention_mask(
            t=int(t),
            key_mask=key_mask,
            causal=causal,
            dtype=q.dtype,
        )
        y = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=1.0 / math.sqrt(int(q.shape[-1])),
            mask=mask,
        )
        y = _merge_heads(y, mx)

        y = y * _sigmoid(self._apply_linear_path(x, f"{prefix}.gate"), mx)
        y = self._apply_linear_path(y, f"{prefix}.wo")

        if query_mask is not None:
            y = y * mx.reshape(query_mask.astype(y.dtype), (b, t, 1))

        return y

    def _joint_attention(
        self,
        x: Any,
        *,
        layer_idx: int,
        start_pos: int,
        kv_latent: list[tuple[Any, Any]] | None,
        kv_text: list[tuple[Any, Any]] | None,
        kv_speaker: list[tuple[Any, Any]] | None,
        latent_mask: Any | None,
        text_mask: Any | None,
        speaker_mask: Any | None,
        joint_mask: Any | None = None,
    ) -> Any:
        mx = self.mx
        _, t, _ = x.shape

        prefix = f"blocks.{layer_idx}.attention"
        q = self._apply_linear_path(x, f"{prefix}.wq")
        k = self._apply_linear_path(x, f"{prefix}.wk")
        v = self._apply_linear_path(x, f"{prefix}.wv")

        q = _reshape_heads(q, self.config.num_heads, mx)
        k = _reshape_heads(k, self.config.num_heads, mx)
        v = _reshape_heads(v, self.config.num_heads, mx)

        q = _rms_norm_head(q, self.t(f"{prefix}.q_norm.weight"), mx, self.config.norm_eps)
        k = _rms_norm_head(k, self.t(f"{prefix}.k_norm.weight"), mx, self.config.norm_eps)

        q = _apply_half_rotary(q, mx, offset=start_pos)
        k = _apply_half_rotary(k, mx, offset=start_pos)

        keys = [k]
        values = [v]

        if kv_latent is not None:
            k_latent, v_latent = kv_latent[layer_idx]
            keys.append(k_latent)
            values.append(v_latent)

        if kv_text is not None:
            k_text, v_text = kv_text[layer_idx]
            keys.append(k_text)
            values.append(v_text)

        if kv_speaker is not None:
            k_speaker, v_speaker = kv_speaker[layer_idx]
            keys.append(k_speaker)
            values.append(v_speaker)

        k_full = mx.concatenate(keys, axis=2)
        v_full = mx.concatenate(values, axis=2)

        if joint_mask is None:
            t_latent = int(kv_latent[layer_idx][0].shape[2]) if kv_latent is not None else 0
            t_text = int(kv_text[layer_idx][0].shape[2]) if kv_text is not None else 0
            t_speaker = int(kv_speaker[layer_idx][0].shape[2]) if kv_speaker is not None else 0
            joint_mask = self._build_joint_attention_mask(
                batch=int(x.shape[0]),
                t_query=int(t),
                t_latent=t_latent,
                t_text=t_text,
                t_speaker=t_speaker,
                latent_mask=latent_mask,
                text_mask=text_mask,
                speaker_mask=speaker_mask,
                dtype=q.dtype,
            )

        y = mx.fast.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            scale=1.0 / math.sqrt(self.head_dim),
            mask=joint_mask,
        )
        y = _merge_heads(y, mx)
        y = y * _sigmoid(self._apply_linear_path(x, f"{prefix}.gate"), mx)
        y = self._apply_linear_path(y, f"{prefix}.wo")
        return y

    def _lowrank_adaln(self, x: Any, *, prefix: str, shift: Any, scale: Any, gate: Any) -> tuple[Any, Any]:
        mx = self.mx

        def _lr(vec: Any, down_key: str, up_w_key: str, up_b_key: str) -> Any:
            h = _silu(vec, mx)
            h = self._apply_linear_path(h, down_key.removesuffix(".weight"))
            h = self._apply_linear_path(h, up_w_key.removesuffix(".weight"), bias=True)
            return vec + h

        shift = _lr(
            shift,
            f"{prefix}.shift_down.weight",
            f"{prefix}.shift_up.weight",
            f"{prefix}.shift_up.bias",
        )
        scale = _lr(
            scale,
            f"{prefix}.scale_down.weight",
            f"{prefix}.scale_up.weight",
            f"{prefix}.scale_up.bias",
        )
        gate = _lr(
            gate,
            f"{prefix}.gate_down.weight",
            f"{prefix}.gate_up.weight",
            f"{prefix}.gate_up.bias",
        )

        x_f32 = x.astype(mx.float32)
        norm = 1.0 / mx.sqrt(mx.mean(x_f32 * x_f32, axis=-1, keepdims=True) + self.config.norm_eps)
        h = (x_f32 * norm).astype(x.dtype)
        h = h * (1.0 + mx.reshape(scale, (scale.shape[0], 1, scale.shape[1])))
        h = h + mx.reshape(shift, (shift.shape[0], 1, shift.shape[1]))
        g = mx.tanh(mx.reshape(gate, (gate.shape[0], 1, gate.shape[1])))
        return h, g

    def _encoder_block(
        self,
        x: Any,
        *,
        prefix: str,
        num_heads: int,
        causal: bool,
        mask: Any,
        mlp_w1_key: str,
        mlp_w2_key: str,
        mlp_w3_key: str,
    ) -> Any:
        mx = self.mx

        h = _rms_norm(x, self.t(f"{prefix}.attention_norm.weight"), mx, self.config.norm_eps)
        h = self._self_attention(
            h,
            prefix=f"{prefix}.attention",
            num_heads=num_heads,
            causal=causal,
            key_mask=mask,
            query_mask=mask,
            half_rotary=False,
        )
        x = x + h

        h = _rms_norm(x, self.t(f"{prefix}.mlp_norm.weight"), mx, self.config.norm_eps)
        h = self._swiglu_paths(
            h,
            w1_path=mlp_w1_key.removesuffix(".weight"),
            w2_path=mlp_w2_key.removesuffix(".weight"),
            w3_path=mlp_w3_key.removesuffix(".weight"),
        )
        h = h * mx.reshape(mask.astype(h.dtype), (mask.shape[0], mask.shape[1], 1))
        x = x + h
        return x

    def _encode_text(self, text_ids: Any, text_mask: Any | None) -> tuple[Any, Any]:
        mx = self.mx
        text_ids = _to_mx_array(text_ids, mx)
        tokens = text_ids.astype(mx.int32)
        b, t = int(tokens.shape[0]), int(tokens.shape[1])
        mask = _ensure_mask(text_mask, batch=b, length=t, mx=mx)

        emb = self.t("text_encoder.text_embedding.weight")
        x = mx.take(emb, tokens, axis=0)
        # Run encoder in float32 to avoid precision drift in KV caches
        x = x.astype(mx.float32)
        x = x * mx.reshape(mask.astype(x.dtype), (b, t, 1))

        for i in range(self.config.text_num_layers):
            p = f"text_encoder.blocks.{i}"
            x = self._encoder_block(
                x,
                prefix=p,
                num_heads=self.config.text_num_heads,
                causal=False,
                mask=mask,
                mlp_w1_key=f"{p}.mlp.w1.weight",
                mlp_w2_key=f"{p}.mlp.w2.weight",
                mlp_w3_key=f"{p}.mlp.w3.weight",
            )

        x = _rms_norm(x, self.t("text_norm.weight"), mx, self.config.norm_eps)
        x = x * mx.reshape(mask.astype(x.dtype), (b, t, 1))
        return x, mask

    def _patch_speaker_latents(self, latents: Any, mask: Any | None) -> tuple[Any, Any]:
        mx = self.mx
        latents = _to_mx_array(latents, mx)

        b, t, d = int(latents.shape[0]), int(latents.shape[1]), int(latents.shape[2])

        if d == self.config.speaker_patch_size * self.config.latent_size:
            patched = latents
            patched_t = int(patched.shape[1])
            patched_mask = _ensure_mask(mask, batch=b, length=patched_t, mx=mx) if mask is not None else mx.ones((b, patched_t), dtype=mx.bool_)
            return patched, patched_mask

        if d != self.config.latent_size:
            raise ValueError(
                "speaker_latents last dim must be latent_size (80) or patched dim (320); "
                f"got {d}"
            )

        patch = self.config.speaker_patch_size
        pad = (-t) % patch

        if pad > 0:
            latents = mx.pad(latents, ((0, 0), (0, pad), (0, 0)))

        patched_t = (t + pad) // patch
        patched = mx.reshape(latents, (b, patched_t, patch * d))

        if mask is None:
            raw_mask = mx.ones((b, t), dtype=mx.bool_)
        else:
            raw_mask = _ensure_mask(mask, batch=b, length=t, mx=mx)

        if pad > 0:
            raw_mask = mx.pad(raw_mask.astype(mx.int32), ((0, 0), (0, pad))) > 0

        m = mx.reshape(raw_mask.astype(mx.int32), (b, patched_t, patch))
        patched_mask = mx.max(m, axis=-1) > 0
        return patched, patched_mask

    def _encode_speaker(self, speaker_latents: Any, speaker_mask: Any | None) -> tuple[Any, Any]:
        mx = self.mx
        x, mask = self._patch_speaker_latents(speaker_latents, speaker_mask)

        # Run speaker encoder in float32 to avoid precision drift in KV caches
        x = self._apply_linear_path(x.astype(mx.float32), "speaker_encoder.in_proj", bias=True)
        x = x / 6.0
        x = x * mx.reshape(mask.astype(x.dtype), (mask.shape[0], mask.shape[1], 1))

        for i in range(self.config.speaker_num_layers):
            p = f"speaker_encoder.blocks.{i}"
            x = self._encoder_block(
                x,
                prefix=p,
                num_heads=self.config.speaker_num_heads,
                causal=True,
                mask=mask,
                mlp_w1_key=f"{p}.mlp.w1.weight",
                mlp_w2_key=f"{p}.mlp.w2.weight",
                mlp_w3_key=f"{p}.mlp.w3.weight",
            )

        x = _rms_norm(x, self.t("speaker_norm.weight"), mx, self.config.norm_eps)
        x = x * mx.reshape(mask.astype(x.dtype), (mask.shape[0], mask.shape[1], 1))
        return x, mask

    def _encode_latent(self, prefix_latents: Any) -> tuple[Any, Any]:
        """Encode latent prefix frames for blockwise generation.

        Mirrors the speaker encoder architecture (same depth, causal masking,
        and `/6.0` scaling) using latent-specific weights.
        Upstream reference:
        https://github.com/jordandare/echo-tts/blob/main/inference_blockwise.py
        """
        mx = self.mx
        x, mask = self._patch_speaker_latents(prefix_latents, None)

        x = self._apply_linear_path(x.astype(mx.float32), "latent_encoder.in_proj", bias=True)
        x = x / 6.0
        x = x * mx.reshape(mask.astype(x.dtype), (mask.shape[0], mask.shape[1], 1))

        for i in range(self.config.speaker_num_layers):
            p = f"latent_encoder.blocks.{i}"
            x = self._encoder_block(
                x,
                prefix=p,
                num_heads=self.config.speaker_num_heads,
                causal=True,
                mask=mask,
                mlp_w1_key=f"{p}.mlp.w1.weight",
                mlp_w2_key=f"{p}.mlp.w2.weight",
                mlp_w3_key=f"{p}.mlp.w3.weight",
            )

        x = _rms_norm(x, self.t("latent_norm.weight"), mx, self.config.norm_eps)
        x = x * mx.reshape(mask.astype(x.dtype), (mask.shape[0], mask.shape[1], 1))
        return x, mask

    def _project_kv(self, hidden: Any, *, k_weight: str, v_weight: str, k_norm_weight: Any) -> tuple[Any, Any]:
        mx = self.mx
        k = self._apply_linear_path(hidden, k_weight)
        v = self._apply_linear_path(hidden, v_weight)
        k = _rms_norm_head(_reshape_heads(k, self.config.num_heads, mx), k_norm_weight, mx, self.config.norm_eps)
        v = _reshape_heads(v, self.config.num_heads, mx)
        # Upcast KV caches to float32 so they match the DiT residual stream precision.
        # Without this, fp16 KV values cause precision drift that compounds across
        # 32 Euler diffusion steps (butterfly effect in the sampler).
        k = k.astype(mx.float32)
        v = v.astype(mx.float32)
        return k, v

    def get_kv_cache_text(self, text_ids: Any, text_mask: Any | None = None) -> tuple[list[tuple[Any, Any]], Any]:
        hidden, mask = self._encode_text(text_ids, text_mask)
        kv: list[tuple[Any, Any]] = []
        for i in range(self.config.num_layers):
            k, v = self._project_kv(
                hidden,
                k_weight=f"blocks.{i}.attention.wk_text",
                v_weight=f"blocks.{i}.attention.wv_text",
                k_norm_weight=self.t(f"blocks.{i}.attention.k_norm.weight"),
            )
            kv.append((k, v))
        return kv, mask

    def get_kv_cache_speaker(self, speaker_latents: Any, speaker_mask: Any | None = None) -> tuple[list[tuple[Any, Any]], Any]:
        hidden, mask = self._encode_speaker(speaker_latents, speaker_mask)
        kv: list[tuple[Any, Any]] = []
        for i in range(self.config.num_layers):
            k, v = self._project_kv(
                hidden,
                k_weight=f"blocks.{i}.attention.wk_speaker",
                v_weight=f"blocks.{i}.attention.wv_speaker",
                k_norm_weight=self.t(f"blocks.{i}.attention.k_norm.weight"),
            )
            kv.append((k, v))
        return kv, mask

    def get_kv_cache_latent(self, prefix_latent: Any) -> list[tuple[Any, Any]]:
        if not self.has_blockwise_modules:
            raise RuntimeError(
                "Blockwise generation requires weights converted with --include-blockwise."
            )

        mx = self.mx
        hidden, _ = self._encode_latent(prefix_latent)
        seq_len = int(hidden.shape[1])
        positions = mx.arange(seq_len, dtype=mx.int32) * int(self.config.speaker_patch_size)

        kv: list[tuple[Any, Any]] = []
        for i in range(self.config.num_layers):
            k = self._apply_linear_path(hidden, f"blocks.{i}.attention.wk_latent")
            v = self._apply_linear_path(hidden, f"blocks.{i}.attention.wv_latent")
            k = _reshape_heads(k, self.config.num_heads, mx)
            v = _reshape_heads(v, self.config.num_heads, mx)
            k = _rms_norm_head(k, self.t(f"blocks.{i}.attention.k_norm.weight"), mx, self.config.norm_eps)
            k = _apply_half_rotary_at_positions(k, positions, mx)
            k = k.astype(mx.float32)
            v = v.astype(mx.float32)
            kv.append((k, v))
        return kv

    def _cond_vectors(self, timesteps: Any, batch: int) -> tuple[Any, Any, Any]:
        mx = self.mx
        timesteps = _to_mx_array(timesteps, mx)
        t = mx.reshape(timesteps.astype(self.dtype), (batch,))

        h = _timestep_embedding(t, self.config.timestep_embed_size, mx, self.dtype)
        h = self._apply_linear_path(h, "cond_module.0")
        h = _silu(h, mx)
        h = self._apply_linear_path(h, "cond_module.2")
        h = _silu(h, mx)
        h = self._apply_linear_path(h, "cond_module.4")

        shift, scale, gate = mx.split(h, 3, axis=-1)
        return shift, scale, gate

    def forward(
        self,
        latents: Any,
        timesteps: Any,
        *,
        kv_text: list[tuple[Any, Any]] | None = None,
        kv_speaker: list[tuple[Any, Any]] | None = None,
        text_mask: Any | None = None,
        speaker_mask: Any | None = None,
        start_pos: int = 0,
        kv_latent: list[tuple[Any, Any]] | None = None,
    ) -> Any:
        mx = self.mx
        latents = _to_mx_array(latents, mx)

        # Residual stream in float32 to avoid fp16 overflow across 24 blocks.
        # Weights stay in fp16; matmuls upcast automatically.
        x = latents.astype(mx.float32)
        b, t, d = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        if d != self.config.latent_size:
            raise ValueError(f"Latent dim mismatch: expected {self.config.latent_size}, got {d}")

        x = self._apply_linear_path(x, "in_proj", bias=True)
        shift, scale, gate = self._cond_vectors(timesteps, b)
        # Keep conditioning in float32 too
        shift, scale, gate = shift.astype(mx.float32), scale.astype(mx.float32), gate.astype(mx.float32)
        t_latent = int(kv_latent[0][0].shape[2]) if kv_latent is not None else 0
        t_text = int(kv_text[0][0].shape[2]) if kv_text is not None else 0
        t_speaker = int(kv_speaker[0][0].shape[2]) if kv_speaker is not None else 0
        latent_mask = None
        if t_latent > 0:
            # Latent visibility is based on absolute frame position:
            # positions < start_pos are prior/continuation context and visible.
            latent_positions = mx.arange(t_latent, dtype=mx.int32) * int(self.config.speaker_patch_size)
            latent_visible = latent_positions < int(start_pos)
            latent_mask = mx.broadcast_to(mx.reshape(latent_visible, (1, t_latent)), (b, t_latent))
        joint_mask = self._build_joint_attention_mask(
            batch=b,
            t_query=t,
            t_latent=t_latent,
            t_text=t_text,
            t_speaker=t_speaker,
            latent_mask=latent_mask,
            text_mask=text_mask,
            speaker_mask=speaker_mask,
            dtype=x.dtype,
        )

        for i in range(self.config.num_layers):
            h, g = self._lowrank_adaln(
                x,
                prefix=f"blocks.{i}.attention_adaln",
                shift=shift,
                scale=scale,
                gate=gate,
            )
            h = self._joint_attention(
                h,
                layer_idx=i,
                start_pos=start_pos,
                kv_latent=kv_latent,
                kv_text=kv_text,
                kv_speaker=kv_speaker,
                latent_mask=latent_mask,
                text_mask=text_mask,
                speaker_mask=speaker_mask,
                joint_mask=joint_mask,
            )
            x = x + g * h

            h, g = self._lowrank_adaln(
                x,
                prefix=f"blocks.{i}.mlp_adaln",
                shift=shift,
                scale=scale,
                gate=gate,
            )
            h = self._swiglu_paths(
                h,
                w1_path=f"blocks.{i}.mlp.w1",
                w2_path=f"blocks.{i}.mlp.w2",
                w3_path=f"blocks.{i}.mlp.w3",
            )
            x = x + g * h

        x = _rms_norm(x, self.t("out_norm.weight"), mx, self.config.norm_eps)
        x = self._apply_linear_path(x, "out_proj", bias=True)
        return x

    def forward_step(
        self,
        latents: Any,
        timesteps: Any,
        *,
        text_ids: Any,
        speaker_latents: Any,
        text_mask: Any | None = None,
        speaker_mask: Any | None = None,
    ) -> Any:
        kv_text, text_k_mask = self.get_kv_cache_text(text_ids, text_mask)
        kv_speaker, speaker_k_mask = self.get_kv_cache_speaker(speaker_latents, speaker_mask)
        return self.forward(
            latents,
            timesteps,
            kv_text=kv_text,
            kv_speaker=kv_speaker,
            text_mask=text_k_mask,
            speaker_mask=speaker_k_mask,
        )
