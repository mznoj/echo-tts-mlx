"""Weight conversion utilities and CLI.

Converts local upstream checkpoints into runtime-ready safetensors for MLX.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.numpy import save_file as save_file_np

from ._conversion_utils import (
    DEFAULT_DAC_CHECKPOINT,
    DEFAULT_DIT_CHECKPOINT,
    DEFAULT_PCA_CHECKPOINT,
    SKIP_BUFFER_KEYS,
    read_safetensor_header,
)


DEFAULT_OUTPUT_DIR = Path("weights/converted")

PRUNE_PREFIXES = ("latent_encoder.", "latent_norm")
PRUNE_CONTAINS = (".wk_latent", ".wv_latent")

DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "model_type": "echo-dit",
    "latent_size": 80,
    "model_size": 2048,
    "num_layers": 24,
    "num_heads": 16,
    "intermediate_size": 5888,
    "norm_eps": 1e-5,
    "text_vocab_size": 256,
    "text_model_size": 1280,
    "text_num_layers": 14,
    "text_num_heads": 10,
    "text_intermediate_size": 3328,
    "speaker_patch_size": 4,
    "speaker_model_size": 1280,
    "speaker_num_layers": 14,
    "speaker_num_heads": 10,
    "speaker_intermediate_size": 3328,
    "timestep_embed_size": 512,
    "adaln_rank": 256,
    "sample_rate": 44100,
    "ae_downsample_factor": 2048,
    "max_latent_length": 640,
    "max_text_length": 768,
    "max_speaker_latent_length": 6400,
    "pca_latent_dim": 1024,
}


@dataclass(frozen=True)
class ConversionSettings:
    dit_dtype: str
    dac_dtype: str
    prune_blockwise: bool
    components: tuple[str, ...]


@dataclass(frozen=True)
class ConversionSummary:
    dit_total: int = 0
    dit_written: int = 0
    dit_pruned: int = 0
    dac_total: int = 0
    dac_written: int = 0
    dac_skipped_buffers: int = 0
    dac_folded_new: int = 0
    dac_folded_old: int = 0
    pca_total: int = 0
    pca_written: int = 0


TORCH_DTYPE_NAMES = {
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
}


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for conversion (install with `pip install \".[convert]\"`)."
        ) from exc
    return torch


def _torch_dtype(torch_mod, dtype_name: str):
    if dtype_name not in TORCH_DTYPE_NAMES:
        raise ValueError(f"Unsupported dtype '{dtype_name}'.")
    return getattr(torch_mod, TORCH_DTYPE_NAMES[dtype_name])


def _normalize_components(raw: str) -> tuple[str, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("components cannot be empty")

    allowed = {"dit", "dac", "pca"}
    unknown = sorted(set(parts) - allowed)
    if unknown:
        raise ValueError(f"Unknown components: {', '.join(unknown)}")

    ordered = tuple(p for p in ("dit", "dac", "pca") if p in parts)
    return ordered


def _should_prune_dit_key(key: str) -> bool:
    if key.startswith(PRUNE_PREFIXES):
        return True
    return any(token in key for token in PRUNE_CONTAINS)


def _to_numpy(tensor) -> Any:
    return tensor.detach().cpu().numpy()


def _cast_float_tensor(tensor, dtype):
    if getattr(tensor, "is_floating_point", lambda: False)():
        return tensor.to(dtype=dtype)
    return tensor


def _fold_weight_norm_torch(g, v, torch_mod, eps: float = 1e-12):
    if g.ndim == 1:
        g = g[:, None, None]
    denom = torch_mod.sqrt(torch_mod.sum(v * v, dim=(1, 2), keepdim=True) + eps)
    return g * (v / denom)


def _convert_dit(
    dit_path: Path,
    output_path: Path,
    target_dtype: str,
    prune_blockwise: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert DiT weights, optionally pruning blockwise-only modules."""
    torch = _require_torch()

    src_header = read_safetensor_header(dit_path)
    out_state: dict[str, Any] = {}
    map_entries: list[dict[str, Any]] = []

    t_dtype = _torch_dtype(torch, target_dtype)

    total = 0
    written = 0
    pruned = 0

    with safe_open(str(dit_path), framework="pt", device="cpu") as f:
        for key in sorted(f.keys()):
            total += 1
            src_dtype = str(src_header[key]["dtype"])
            src_shape = [int(x) for x in src_header[key]["shape"]]

            if prune_blockwise and _should_prune_dit_key(key):
                pruned += 1
                map_entries.append(
                    {
                        "source": [key],
                        "target": None,
                        "action": "prune_blockwise",
                        "source_dtype": src_dtype,
                        "target_dtype": None,
                        "shape": src_shape,
                    }
                )
                continue

            tensor = f.get_tensor(key)
            tensor = _cast_float_tensor(tensor, dtype=t_dtype)

            out_state[key] = _to_numpy(tensor)
            written += 1
            map_entries.append(
                {
                    "source": [key],
                    "target": key,
                    "action": "copy",
                    "source_dtype": src_dtype,
                    "target_dtype": str(tensor.dtype).replace("torch.", "").upper(),
                    "shape": src_shape,
                }
            )

    save_file_np(out_state, str(output_path))

    summary = {
        "dit_total": total,
        "dit_written": written,
        "dit_pruned": pruned,
    }
    return summary, {"dit": map_entries}


def _convert_dac(
    dac_path: Path,
    output_path: Path,
    target_dtype: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert DAC weights and fold weight normalization."""
    torch = _require_torch()

    src_header = read_safetensor_header(dac_path)
    out_state: dict[str, Any] = {}
    map_entries: list[dict[str, Any]] = []

    t_dtype = _torch_dtype(torch, target_dtype)

    total = 0
    written = 0
    skipped_buffers = 0
    folded_new = 0
    folded_old = 0

    with safe_open(str(dac_path), framework="pt", device="cpu") as f:
        keys = sorted(f.keys())
        for key in keys:
            total += 1

            if key in SKIP_BUFFER_KEYS:
                skipped_buffers += 1
                map_entries.append(
                    {
                        "source": [key],
                        "target": None,
                        "action": "skip_buffer",
                        "source_dtype": str(src_header[key]["dtype"]),
                        "target_dtype": None,
                        "shape": [int(x) for x in src_header[key]["shape"]],
                    }
                )
                continue

            if key.endswith("parametrizations.weight.original0"):
                continue

            if key.endswith("parametrizations.weight.original1"):
                base = key[: -len("parametrizations.weight.original1")]
                g_key = base + "parametrizations.weight.original0"
                if g_key not in src_header:
                    raise KeyError(f"Missing paired weight norm key: {g_key}")

                g = f.get_tensor(g_key).to(dtype=torch.float32)
                v = f.get_tensor(key).to(dtype=torch.float32)
                folded = _fold_weight_norm_torch(g, v, torch)
                folded = _cast_float_tensor(folded, dtype=t_dtype)

                out_key = base + "weight"
                out_state[out_key] = _to_numpy(folded)
                written += 1
                folded_new += 1

                map_entries.append(
                    {
                        "source": [g_key, key],
                        "target": out_key,
                        "action": "fold_weight_norm",
                        "source_dtype": "F32",
                        "target_dtype": str(folded.dtype).replace("torch.", "").upper(),
                        "shape": [int(x) for x in src_header[key]["shape"]],
                    }
                )
                continue

            if key.endswith(".weight_g"):
                continue

            if key.endswith(".weight_v"):
                base = key[: -len(".weight_v")]
                g_key = base + ".weight_g"
                if g_key not in src_header:
                    raise KeyError(f"Missing paired legacy weight norm key: {g_key}")

                g = f.get_tensor(g_key).to(dtype=torch.float32)
                v = f.get_tensor(key).to(dtype=torch.float32)
                folded = _fold_weight_norm_torch(g, v, torch)
                folded = _cast_float_tensor(folded, dtype=t_dtype)

                out_key = base + ".weight"
                out_state[out_key] = _to_numpy(folded)
                written += 1
                folded_old += 1

                map_entries.append(
                    {
                        "source": [g_key, key],
                        "target": out_key,
                        "action": "fold_weight_norm",
                        "source_dtype": "F32",
                        "target_dtype": str(folded.dtype).replace("torch.", "").upper(),
                        "shape": [int(x) for x in src_header[key]["shape"]],
                    }
                )
                continue

            tensor = f.get_tensor(key)
            tensor = _cast_float_tensor(tensor, dtype=t_dtype)

            out_state[key] = _to_numpy(tensor)
            written += 1

            map_entries.append(
                {
                    "source": [key],
                    "target": key,
                    "action": "copy",
                    "source_dtype": str(src_header[key]["dtype"]),
                    "target_dtype": str(tensor.dtype).replace("torch.", "").upper(),
                    "shape": [int(x) for x in src_header[key]["shape"]],
                }
            )

    save_file_np(out_state, str(output_path))

    summary = {
        "dac_total": total,
        "dac_written": written,
        "dac_skipped_buffers": skipped_buffers,
        "dac_folded_new": folded_new,
        "dac_folded_old": folded_old,
    }
    return summary, {"dac": map_entries}


def _convert_pca(pca_path: Path, output_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy PCA state (always float32)."""
    src_header = read_safetensor_header(pca_path)

    out_state: dict[str, Any] = {}
    map_entries: list[dict[str, Any]] = []

    total = 0
    written = 0

    with safe_open(str(pca_path), framework="np") as f:
        for key in sorted(f.keys()):
            total += 1
            tensor = f.get_tensor(key)
            tensor = tensor.astype("float32")
            out_state[key] = tensor
            written += 1

            map_entries.append(
                {
                    "source": [key],
                    "target": key,
                    "action": "copy",
                    "source_dtype": str(src_header[key]["dtype"]),
                    "target_dtype": "F32",
                    "shape": [int(x) for x in src_header[key]["shape"]],
                }
            )

    save_file_np(out_state, str(output_path))

    summary = {
        "pca_total": total,
        "pca_written": written,
    }
    return summary, {"pca": map_entries}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def convert_weights(
    *,
    dit_path: Path,
    dac_path: Path,
    pca_path: Path,
    output_dir: Path,
    settings: ConversionSettings,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "dit": output_dir / "dit_weights.safetensors",
        "dac": output_dir / "dac_weights.safetensors",
        "pca": output_dir / "pca_state.safetensors",
        "config": output_dir / "config.json",
        "weight_map": output_dir / "weight_map.json",
    }

    summary: dict[str, Any] = {}
    weight_map: dict[str, list[dict[str, Any]]] = {"dit": [], "dac": [], "pca": []}

    if "dit" in settings.components:
        dit_summary, dit_map = _convert_dit(
            dit_path=dit_path,
            output_path=outputs["dit"],
            target_dtype=settings.dit_dtype,
            prune_blockwise=settings.prune_blockwise,
        )
        summary.update(dit_summary)
        weight_map["dit"] = dit_map["dit"]

    if "dac" in settings.components:
        dac_summary, dac_map = _convert_dac(
            dac_path=dac_path,
            output_path=outputs["dac"],
            target_dtype=settings.dac_dtype,
        )
        summary.update(dac_summary)
        weight_map["dac"] = dac_map["dac"]

    if "pca" in settings.components:
        pca_summary, pca_map = _convert_pca(
            pca_path=pca_path,
            output_path=outputs["pca"],
        )
        summary.update(pca_summary)
        weight_map["pca"] = pca_map["pca"]

    # Always emit config + weight map for selected conversion set.
    _write_json(outputs["config"], DEFAULT_MODEL_CONFIG)

    map_payload = {
        "format_version": 1,
        "inputs": {
            "dit": str(dit_path),
            "dac": str(dac_path),
            "pca": str(pca_path),
        },
        "outputs": {k: str(v) for k, v in outputs.items()},
        "settings": {
            "dit_dtype": settings.dit_dtype,
            "dac_dtype": settings.dac_dtype,
            "prune_blockwise": settings.prune_blockwise,
            "components": list(settings.components),
        },
        "summary": summary,
        "mappings": weight_map,
    }
    _write_json(outputs["weight_map"], map_payload)

    return {
        "output_dir": str(output_dir),
        "outputs": {k: str(v) for k, v in outputs.items()},
        "summary": summary,
        "settings": asdict(settings),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert local Echo-TTS/Fish-S1-DAC checkpoints to MLX-ready safetensors.")
    p.add_argument("--dit", type=Path, default=DEFAULT_DIT_CHECKPOINT, help="Path to upstream DiT safetensors")
    p.add_argument("--dac", type=Path, default=DEFAULT_DAC_CHECKPOINT, help="Path to upstream DAC safetensors")
    p.add_argument("--pca", type=Path, default=DEFAULT_PCA_CHECKPOINT, help="Path to upstream PCA safetensors")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for converted files")
    p.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Target dtype for DiT weights",
    )
    p.add_argument(
        "--dac-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float32",
        help="Target dtype for DAC weights",
    )

    p.add_argument("--prune-blockwise", dest="prune_blockwise", action="store_true", default=True)
    p.add_argument("--no-prune-blockwise", dest="prune_blockwise", action="store_false")
    p.add_argument(
        "--include-blockwise",
        dest="prune_blockwise",
        action="store_false",
        help="Include latent_encoder/wk_latent/wv_latent weights (equivalent to --no-prune-blockwise)",
    )
    p.add_argument(
        "--components",
        default="dit,dac,pca",
        help="Comma-separated component list from: dit,dac,pca",
    )
    p.add_argument(
        "--quantize",
        choices=("none", "8bit", "4bit", "mxfp4", "mixed"),
        default="none",
        help="Optional post-conversion quantization mode for DiT weights",
    )
    p.add_argument(
        "--save-quantized",
        type=Path,
        default=None,
        help="Optional output directory to save a quantized checkpoint package",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = ConversionSettings(
        dit_dtype=args.dtype,
        dac_dtype=args.dac_dtype,
        prune_blockwise=bool(args.prune_blockwise),
        components=_normalize_components(args.components),
    )

    if "dit" in settings.components and not args.dit.exists():
        raise FileNotFoundError(f"DiT checkpoint not found: {args.dit}")
    if "dac" in settings.components and not args.dac.exists():
        raise FileNotFoundError(f"DAC checkpoint not found: {args.dac}")
    if "pca" in settings.components and not args.pca.exists():
        raise FileNotFoundError(f"PCA checkpoint not found: {args.pca}")

    result = convert_weights(
        dit_path=args.dit,
        dac_path=args.dac,
        pca_path=args.pca,
        output_dir=args.output,
        settings=settings,
    )

    if args.save_quantized is not None:
        if args.quantize == "none":
            raise ValueError("--save-quantized requires --quantize 8bit, 4bit, mxfp4, or mixed.")
        from .pipeline import EchoTTS

        pipe = EchoTTS.from_pretrained(args.output, dtype=args.dtype, quantize=args.quantize)
        saved_dir = pipe.save_quantized(args.save_quantized)
        print(f"Saved quantized package: {saved_dir}")

    print("Conversion complete")
    print(f"Output dir: {result['output_dir']}")
    for key, value in result["summary"].items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
