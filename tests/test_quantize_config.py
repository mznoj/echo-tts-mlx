from __future__ import annotations

import json
from pathlib import Path

import pytest

from echo_tts_mlx.model import (
    QuantizeConfig,
    detect_quantize_config,
    load_quantize_config,
    save_quantize_config,
)


def test_quantize_config_roundtrip(tmp_path: Path) -> None:
    out_dir = tmp_path / "quantized-8bit"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = QuantizeConfig(
        mode="8bit",
        bits=8,
        group_size=64,
        quantized_modules=["blocks.0.attention.wq", "blocks.0.attention.wk", "cond_module.0"],
    )
    save_quantize_config(out_dir, cfg)

    assert detect_quantize_config(out_dir)
    loaded = load_quantize_config(out_dir)
    assert loaded == cfg


def test_quantize_config_detect_false(tmp_path: Path) -> None:
    assert not detect_quantize_config(tmp_path)


def test_quantize_config_missing_key_raises(tmp_path: Path) -> None:
    out_dir = tmp_path / "quantized-bad"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quantize_config.json").write_text(json.dumps({"mode": "8bit"}) + "\n")

    with pytest.raises(ValueError, match="quantize config"):
        load_quantize_config(out_dir)


def test_quantize_config_per_module_roundtrip(tmp_path: Path) -> None:
    out_dir = tmp_path / "quantized-mixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = QuantizeConfig(
        mode="mixed",
        bits=8,
        group_size=64,
        quantized_modules=["blocks.18.attention.wq", "blocks.0.attention.wq"],
        per_module=True,
        modules={
            "blocks.18.attention.wq": {"bits": 8, "group_size": 64, "mode": "affine"},
            "blocks.0.attention.wq": {"bits": 4, "group_size": 32, "mode": "mxfp4"},
        },
    )
    save_quantize_config(out_dir, cfg)
    loaded = load_quantize_config(out_dir)
    assert loaded == cfg
