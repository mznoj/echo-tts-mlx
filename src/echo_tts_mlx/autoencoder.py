# Copyright 2025 Jordan Darefsky (original Echo-TTS)
# Copyright 2026 Matthew Znoj (MLX port)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from the original Echo-TTS autoencoder.py:
# - Ported from PyTorch/CUDA to MLX for Apple Silicon
# - Refactored for converted checkpoint loading
# - Added float32-only path for MLX compatibility

"""Fish S1-DAC MLX wrapper (converted checkpoints only, float32 path)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open


DEFAULT_CONVERTED_WEIGHTS_DIR = Path("weights/converted")


def resolve_converted_dac_paths(weights_dir: str | Path) -> tuple[Path, Path]:
    """Return `(config_path, dac_weights_path)` from a converted weights directory."""
    root = Path(weights_dir)
    if root.is_file():
        raise ValueError(
            f"Expected converted weights directory, got file path: {root}. "
            "Pass the directory containing `config.json` and `dac_weights.safetensors`."
        )

    config_path = root / "config.json"
    dac_path = root / "dac_weights.safetensors"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing converted config: {config_path}")
    if not dac_path.exists():
        raise FileNotFoundError(f"Missing converted DAC weights: {dac_path}")
    return config_path, dac_path


def load_converted_dac_state(weights_dir: str | Path) -> dict[str, np.ndarray]:
    """Load converted DAC state tensors as NumPy arrays."""
    _, dac_path = resolve_converted_dac_paths(weights_dir)
    out: dict[str, np.ndarray] = {}
    with safe_open(str(dac_path), framework="np") as f:
        for key in f.keys():
            tensor = np.asarray(f.get_tensor(key))
            if np.issubdtype(tensor.dtype, np.floating) and tensor.dtype != np.float32:
                tensor = tensor.astype(np.float32)
            out[key] = tensor
    return out


class MlxFishS1DAC:
    """MLX Fish S1-DAC autoencoder wrapper around the validated DAC implementation.

    This class is converted-checkpoint only and enforces float32 DAC execution.
    """

    def __init__(self, np_state: dict[str, np.ndarray]) -> None:
        from ._dac_core import MlxFishS1DAC as _DacCoreFishS1DAC

        self._impl = _DacCoreFishS1DAC(np_state)

    @classmethod
    def from_pretrained(
        cls,
        weights_dir: str | Path = DEFAULT_CONVERTED_WEIGHTS_DIR,
        *,
        dtype: str = "float32",
    ) -> "MlxFishS1DAC":
        if dtype != "float32":
            raise ValueError(
                f"DAC path is float32-only. Received dtype={dtype!r}. "
                "Use dtype='float32'."
            )
        state = load_converted_dac_state(weights_dir)
        return cls(state)

    def _to_mx_array(self, x: Any) -> Any:
        import mlx.core as mx

        if hasattr(x, "detach"):
            # torch.Tensor path
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return mx.array(x, dtype=mx.float32)

        if x.__class__.__module__.startswith("mlx"):
            return x.astype(mx.float32)

        return mx.array(np.asarray(x), dtype=mx.float32)

    def encode_zq(self, audio: Any, n_quantizers: int | None = None):
        """Encode waveform tensor to quantized latent frames."""
        audio_mx = self._to_mx_array(audio)
        return self._impl.encode_zq(audio_mx, n_quantizers=n_quantizers)

    def decode_zq(self, z_q: Any):
        """Decode quantized latent frames to waveform tensor."""
        z_q_mx = self._to_mx_array(z_q)
        return self._impl.decode_zq(z_q_mx)
