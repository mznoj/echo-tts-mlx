"""PCA state loading and tensor transforms for Echo-TTS latents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors import safe_open


DEFAULT_CONVERTED_WEIGHTS_DIR = Path("weights/converted")


@dataclass(frozen=True)
class PCAState:
    """PCA parameters used between DAC latents and DiT latents."""

    pca_components: np.ndarray  # (80, 1024), float32
    pca_mean: np.ndarray  # (1024,), float32
    latent_scale: float


def resolve_converted_pca_path(weights_dir: str | Path = DEFAULT_CONVERTED_WEIGHTS_DIR) -> Path:
    """Resolve the converted PCA safetensors path from a weights directory."""
    root = Path(weights_dir)
    if root.is_file():
        raise ValueError(
            f"Expected converted weights directory, got file path: {root}. "
            "Pass the directory containing `pca_state.safetensors`."
        )
    pca_path = root / "pca_state.safetensors"
    if not pca_path.exists():
        raise FileNotFoundError(f"Missing converted PCA state: {pca_path}")
    return pca_path


def load_pca_state(weights_dir: str | Path = DEFAULT_CONVERTED_WEIGHTS_DIR) -> PCAState:
    """Load converted PCA tensors as float32."""
    pca_path = resolve_converted_pca_path(weights_dir)
    with safe_open(str(pca_path), framework="np") as f:
        required = {"pca_components", "pca_mean", "latent_scale"}
        missing = required.difference(set(f.keys()))
        if missing:
            raise KeyError(f"Missing required PCA keys: {sorted(missing)}")

        pca_components = np.asarray(f.get_tensor("pca_components"), dtype=np.float32)
        pca_mean = np.asarray(f.get_tensor("pca_mean"), dtype=np.float32)
        latent_scale_arr = np.asarray(f.get_tensor("latent_scale"), dtype=np.float32).reshape(-1)
        if latent_scale_arr.size != 1:
            raise ValueError(
                "Expected `latent_scale` to contain exactly one value, "
                f"got shape {tuple(latent_scale_arr.shape)}."
            )

    if pca_components.shape != (80, 1024):
        raise ValueError(f"Expected pca_components shape (80, 1024), got {tuple(pca_components.shape)}")
    if pca_mean.shape != (1024,):
        raise ValueError(f"Expected pca_mean shape (1024,), got {tuple(pca_mean.shape)}")

    return PCAState(
        pca_components=pca_components,
        pca_mean=pca_mean,
        latent_scale=float(latent_scale_arr[0]),
    )


def pca_encode_np(z_q: np.ndarray, state: PCAState) -> np.ndarray:
    """Project DAC latent tensor `(B, 1024, T)` to DiT latent tensor `(B, T, 80)`."""
    z_q = np.asarray(z_q, dtype=np.float32)
    if z_q.ndim != 3 or z_q.shape[1] != 1024:
        raise ValueError(f"z_q must have shape (B, 1024, T), got {tuple(z_q.shape)}")

    z = np.transpose(z_q, (0, 2, 1))
    z = (z - state.pca_mean.reshape(1, 1, 1024)) @ state.pca_components.T
    return z * np.float32(state.latent_scale)


def pca_decode_np(z: np.ndarray, state: PCAState) -> np.ndarray:
    """Project DiT latent tensor `(B, T, 80)` back to DAC latent tensor `(B, 1024, T)`."""
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 3 or z.shape[2] != 80:
        raise ValueError(f"z must have shape (B, T, 80), got {tuple(z.shape)}")

    z_q = (z / np.float32(state.latent_scale)) @ state.pca_components + state.pca_mean.reshape(1, 1, 1024)
    return np.transpose(z_q, (0, 2, 1))
