"""Model config definitions and loaders for converted Echo-TTS weights."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Static model hyperparameters loaded from `weights/converted/config.json`."""

    model_type: str
    latent_size: int
    model_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    norm_eps: float
    text_vocab_size: int
    text_model_size: int
    text_num_layers: int
    text_num_heads: int
    text_intermediate_size: int
    speaker_patch_size: int
    speaker_model_size: int
    speaker_num_layers: int
    speaker_num_heads: int
    speaker_intermediate_size: int
    timestep_embed_size: int
    adaln_rank: int
    sample_rate: int
    ae_downsample_factor: int
    max_latent_length: int
    max_text_length: int
    max_speaker_latent_length: int
    pca_latent_dim: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        return cls(
            model_type=str(payload["model_type"]),
            latent_size=int(payload["latent_size"]),
            model_size=int(payload["model_size"]),
            num_layers=int(payload["num_layers"]),
            num_heads=int(payload["num_heads"]),
            intermediate_size=int(payload["intermediate_size"]),
            norm_eps=float(payload["norm_eps"]),
            text_vocab_size=int(payload["text_vocab_size"]),
            text_model_size=int(payload["text_model_size"]),
            text_num_layers=int(payload["text_num_layers"]),
            text_num_heads=int(payload["text_num_heads"]),
            text_intermediate_size=int(payload["text_intermediate_size"]),
            speaker_patch_size=int(payload["speaker_patch_size"]),
            speaker_model_size=int(payload["speaker_model_size"]),
            speaker_num_layers=int(payload["speaker_num_layers"]),
            speaker_num_heads=int(payload["speaker_num_heads"]),
            speaker_intermediate_size=int(payload["speaker_intermediate_size"]),
            timestep_embed_size=int(payload["timestep_embed_size"]),
            adaln_rank=int(payload["adaln_rank"]),
            sample_rate=int(payload["sample_rate"]),
            ae_downsample_factor=int(payload["ae_downsample_factor"]),
            max_latent_length=int(payload["max_latent_length"]),
            max_text_length=int(payload["max_text_length"]),
            max_speaker_latent_length=int(payload["max_speaker_latent_length"]),
            pca_latent_dim=int(payload["pca_latent_dim"]),
        )


def resolve_converted_paths(weights_dir: str | Path) -> tuple[Path, Path]:
    """Return `(config_path, dit_weights_path)` for a converted weights directory."""
    root = Path(weights_dir)
    config_path = root / "config.json"
    dit_path = root / "dit_weights.safetensors"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing converted config: {config_path}")
    if not dit_path.exists():
        raise FileNotFoundError(f"Missing converted DiT weights: {dit_path}")

    return config_path, dit_path


def load_model_config(path: str | Path) -> ModelConfig:
    """Load model config from a JSON file path or converted weights directory."""
    raw_path = Path(path)
    config_path = raw_path / "config.json" if raw_path.is_dir() else raw_path
    payload = json.loads(config_path.read_text())
    return ModelConfig.from_dict(payload)
