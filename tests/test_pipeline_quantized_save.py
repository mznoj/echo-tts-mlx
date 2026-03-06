from __future__ import annotations

from pathlib import Path

import pytest

from echo_tts_mlx.model import QuantizeConfig
from echo_tts_mlx.pipeline import EchoTTS


def _make_pipeline_stub(tmp_path: Path, *, mode: str) -> EchoTTS:
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "dac_weights.safetensors", "pca_state.safetensors", "weight_map.json"):
        (src / name).write_bytes(name.encode("utf-8"))

    class _FakeModel:
        quantize_mode = mode

        def __init__(self) -> None:
            self.saved_to: Path | None = None

        def save_weights(self, path: str | Path) -> Path:
            out = Path(path)
            out.write_bytes(b"DIT")
            self.saved_to = out
            return out

        def quantize_config(self, *, group_size: int = 64) -> QuantizeConfig:
            return QuantizeConfig(
                mode=mode,
                bits=(8 if mode == "8bit" else 4),
                group_size=group_size,
                quantized_modules=["blocks.0.attention.wq"],
            )

    pipe = EchoTTS.__new__(EchoTTS)
    pipe.model = _FakeModel()
    pipe.weights_dir = src
    return pipe


def test_save_quantized_rejects_unquantized_model(tmp_path: Path) -> None:
    pipe = _make_pipeline_stub(tmp_path, mode="none")
    with pytest.raises(ValueError, match="not quantized"):
        pipe.save_quantized(tmp_path / "out")


def test_save_quantized_writes_package(tmp_path: Path) -> None:
    pipe = _make_pipeline_stub(tmp_path, mode="8bit")
    out_dir = pipe.save_quantized(tmp_path / "quantized")

    assert (out_dir / "dit_weights.safetensors").exists()
    assert (out_dir / "quantize_config.json").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "dac_weights.safetensors").exists()
    assert (out_dir / "pca_state.safetensors").exists()
    assert (out_dir / "weight_map.json").exists()
