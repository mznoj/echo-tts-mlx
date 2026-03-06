"""Checkpoint conversion utilities and weight-norm folding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import struct
from typing import Any

import numpy as np
from safetensors import safe_open


DEFAULT_DAC_CHECKPOINT = Path("weights/upstream/dac_model.safetensors")
DEFAULT_DIT_CHECKPOINT = Path("weights/upstream/dit_model.safetensors")
DEFAULT_PCA_CHECKPOINT = Path("weights/upstream/pca_state.safetensors")


# Registered buffers that should be recomputed at runtime.
SKIP_BUFFER_KEYS = {
    "quantizer.pre_module.causal_mask",
    "quantizer.pre_module.freqs_cis",
    "quantizer.post_module.causal_mask",
    "quantizer.post_module.freqs_cis",
    "encoder.block.4.block.5.causal_mask",
    "encoder.block.4.block.5.freqs_cis",
}


@dataclass(frozen=True)
class WNFoldStats:
    """Counts of folded/special keys while preparing a runtime-ready state dict."""

    total_keys: int
    skipped_buffers: int
    folded_new_style: int
    folded_old_style: int


@dataclass(frozen=True)
class TensorMeta:
    """Minimal safetensors metadata for shape/dtype inspection without tensor decode."""

    key: str
    dtype: str
    shape: tuple[int, ...]


def read_safetensor_header(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read the safetensors JSON header without loading tensor payloads."""
    with Path(path).open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    return {k: v for k, v in header.items() if k != "__metadata__"}


def read_tensor_meta(path: str | Path) -> list[TensorMeta]:
    """Return tensor metadata for diagnostics/reporting."""
    header = read_safetensor_header(path)
    return [
        TensorMeta(key=k, dtype=str(v["dtype"]), shape=tuple(int(x) for x in v["shape"]))
        for k, v in sorted(header.items())
    ]


def _safe_open_np(path: str | Path) -> dict[str, np.ndarray]:
    """Load tensors with numpy backend, skipping unsupported dtypes (e.g. BF16 buffers)."""
    out: dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="np") as f:
        for key in f.keys():
            try:
                out[key] = np.asarray(f.get_tensor(key))
            except TypeError:
                # Numpy backend cannot decode bf16 directly. This only
                # affects registered buffers we recompute anyway.
                continue
    return out


def fold_weight_norm(g: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Fold weight norm parameters into a single convolution weight tensor."""
    if g.ndim == 1:
        g = g[:, None, None]
    denom = np.sqrt(np.sum(v * v, axis=(1, 2), keepdims=True) + eps)
    return g * (v / denom)


def load_and_fold_dac_state(path: str | Path) -> tuple[dict[str, np.ndarray], WNFoldStats]:
    """Load DAC checkpoint and fold both weight-norm formats into plain `.weight` keys.

    Supported formats:
    - New parametrization API: `...parametrizations.weight.original0/1`
    - Legacy API: `...weight_g/weight_v`
    """
    raw = _safe_open_np(path)
    out: dict[str, np.ndarray] = {}

    skipped_buffers = 0
    folded_new = 0
    folded_old = 0

    for key in sorted(raw):
        if key in SKIP_BUFFER_KEYS:
            skipped_buffers += 1
            continue

        if key.endswith("parametrizations.weight.original0"):
            # consumed by original1 branch
            continue

        if key.endswith("parametrizations.weight.original1"):
            base = key[: -len("parametrizations.weight.original1")]
            g_key = base + "parametrizations.weight.original0"
            if g_key not in raw:
                raise KeyError(f"Missing paired weight norm key: {g_key}")
            out[base + "weight"] = fold_weight_norm(raw[g_key], raw[key])
            folded_new += 1
            continue

        if key.endswith(".weight_g"):
            # consumed by weight_v branch
            continue

        if key.endswith(".weight_v"):
            base = key[: -len(".weight_v")]
            g_key = base + ".weight_g"
            if g_key not in raw:
                raise KeyError(f"Missing paired legacy weight norm key: {g_key}")
            out[base + ".weight"] = fold_weight_norm(raw[g_key], raw[key])
            folded_old += 1
            continue

        out[key] = raw[key]

    stats = WNFoldStats(
        total_keys=len(raw),
        skipped_buffers=skipped_buffers,
        folded_new_style=folded_new,
        folded_old_style=folded_old,
    )
    return out, stats


def to_torch_state(np_state: dict[str, np.ndarray]) -> dict[str, "torch.Tensor"]:
    """Convert numpy state to torch tensors lazily (import inside function)."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only when torch missing
        raise RuntimeError("PyTorch is required for torch tensor conversion.") from exc

    return {k: torch.from_numpy(v.copy()) for k, v in np_state.items()}


def to_mlx_state(np_state: dict[str, np.ndarray]) -> dict[str, "mx.array"]:
    """Convert numpy state to MLX arrays lazily (import inside function)."""
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - exercised only when mlx missing
        raise RuntimeError("MLX is required for MLX tensor conversion.") from exc

    return {k: mx.array(v) for k, v in np_state.items()}
