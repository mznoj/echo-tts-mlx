from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file as save_file_np

from echo_tts_mlx.conversion import (
    ConversionSettings,
    _convert_dac,
    _convert_dit,
    _normalize_components,
    _should_prune_dit_key,
    main as conversion_main,
    convert_weights,
)


HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_normalize_components() -> None:
    assert _normalize_components("dit,dac,pca") == ("dit", "dac", "pca")
    assert _normalize_components("pca,dit") == ("dit", "pca")
    with pytest.raises(ValueError):
        _normalize_components("")
    with pytest.raises(ValueError):
        _normalize_components("dit,unknown")


def test_should_prune_dit_key() -> None:
    assert _should_prune_dit_key("latent_encoder.blocks.0.weight")
    assert _should_prune_dit_key("latent_norm.weight")
    assert _should_prune_dit_key("blocks.0.attention.wk_latent.weight")
    assert _should_prune_dit_key("blocks.0.attention.wv_latent.weight")
    assert not _should_prune_dit_key("blocks.0.attention.wk_text.weight")


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_convert_dit_prunes_blockwise_keys(tmp_path: Path) -> None:
    src = tmp_path / "dit_src.safetensors"
    out = tmp_path / "dit_out.safetensors"

    state = {
        "in_proj.weight": np.ones((2, 2), dtype=np.float32),
        "latent_encoder.weight": np.full((2, 2), 2.0, dtype=np.float32),
        "blocks.0.attention.wk_latent.weight": np.full((2, 2), 3.0, dtype=np.float32),
        "blocks.0.attention.wk_text.weight": np.full((2, 2), 4.0, dtype=np.float32),
    }
    save_file_np(state, str(src))

    summary, mapping = _convert_dit(src, out, target_dtype="float16", prune_blockwise=True)

    assert summary["dit_total"] == 4
    assert summary["dit_pruned"] == 2
    assert summary["dit_written"] == 2

    with safe_open(str(out), framework="np") as f:
        keys = sorted(f.keys())
        assert keys == ["blocks.0.attention.wk_text.weight", "in_proj.weight"]
        assert f.get_tensor("in_proj.weight").dtype == np.float16

    pruned_entries = [m for m in mapping["dit"] if m["action"] == "prune_blockwise"]
    assert len(pruned_entries) == 2


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_convert_dit_include_blockwise_keeps_keys(tmp_path: Path) -> None:
    src = tmp_path / "dit_src.safetensors"
    out = tmp_path / "dit_out.safetensors"

    state = {
        "in_proj.weight": np.ones((2, 2), dtype=np.float32),
        "latent_encoder.weight": np.full((2, 2), 2.0, dtype=np.float32),
        "blocks.0.attention.wk_latent.weight": np.full((2, 2), 3.0, dtype=np.float32),
        "blocks.0.attention.wk_text.weight": np.full((2, 2), 4.0, dtype=np.float32),
    }
    save_file_np(state, str(src))

    summary, mapping = _convert_dit(src, out, target_dtype="float16", prune_blockwise=False)
    assert summary["dit_pruned"] == 0
    assert summary["dit_written"] == 4
    assert not [m for m in mapping["dit"] if m["action"] == "prune_blockwise"]

    with safe_open(str(out), framework="np") as f:
        keys = sorted(f.keys())
        assert keys == sorted(state.keys())


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_convert_dac_folds_new_and_old_weight_norm(tmp_path: Path) -> None:
    src = tmp_path / "dac_src.safetensors"
    out = tmp_path / "dac_out.safetensors"

    g_new = np.array([[[2.0]], [[3.0]]], dtype=np.float32)
    v_new = np.array(
        [
            [[1.0], [2.0], [3.0]],
            [[2.0], [4.0], [4.0]],
        ],
        dtype=np.float32,
    )

    g_old = np.array([[[1.5]], [[0.5]]], dtype=np.float32)
    v_old = np.array(
        [
            [[1.0], [0.0], [1.0]],
            [[2.0], [0.0], [0.0]],
        ],
        dtype=np.float32,
    )

    state = {
        "layer.conv.parametrizations.weight.original0": g_new,
        "layer.conv.parametrizations.weight.original1": v_new,
        "vq.in_proj.weight_g": g_old,
        "vq.in_proj.weight_v": v_old,
        "foo.bias": np.array([1.0, -1.0], dtype=np.float32),
        "quantizer.pre_module.causal_mask": np.zeros((2, 2), dtype=np.bool_),
    }
    save_file_np(state, str(src))

    summary, mapping = _convert_dac(src, out, target_dtype="float32")

    assert summary["dac_total"] == 6
    assert summary["dac_skipped_buffers"] == 1
    assert summary["dac_folded_new"] == 1
    assert summary["dac_folded_old"] == 1
    assert summary["dac_written"] == 3

    with safe_open(str(out), framework="np") as f:
        keys = sorted(f.keys())
        assert keys == ["foo.bias", "layer.conv.weight", "vq.in_proj.weight"]

        expected_new = g_new * (v_new / np.sqrt(np.sum(v_new * v_new, axis=(1, 2), keepdims=True) + 1e-12))
        expected_old = g_old * (v_old / np.sqrt(np.sum(v_old * v_old, axis=(1, 2), keepdims=True) + 1e-12))

        assert np.allclose(f.get_tensor("layer.conv.weight"), expected_new, atol=1e-6)
        assert np.allclose(f.get_tensor("vq.in_proj.weight"), expected_old, atol=1e-6)

    folded_entries = [m for m in mapping["dac"] if m["action"] == "fold_weight_norm"]
    assert len(folded_entries) == 2


def test_convert_weights_pca_only(tmp_path: Path) -> None:
    pca_src = tmp_path / "pca_src.safetensors"
    save_file_np(
        {
            "pca_components": np.ones((2, 4), dtype=np.float32),
            "pca_mean": np.zeros((4,), dtype=np.float32),
            "latent_scale": np.array([1.0], dtype=np.float32),
        },
        str(pca_src),
    )

    out_dir = tmp_path / "out"
    result = convert_weights(
        dit_path=tmp_path / "unused_dit.safetensors",
        dac_path=tmp_path / "unused_dac.safetensors",
        pca_path=pca_src,
        output_dir=out_dir,
        settings=ConversionSettings(
            dit_dtype="float16",
            dac_dtype="float32",
            prune_blockwise=True,
            components=("pca",),
        ),
    )

    assert Path(result["outputs"]["pca"]).exists()
    assert Path(result["outputs"]["config"]).exists()
    assert Path(result["outputs"]["weight_map"]).exists()

    with safe_open(result["outputs"]["pca"], framework="np") as f:
        assert sorted(f.keys()) == ["latent_scale", "pca_components", "pca_mean"]
        assert f.get_tensor("pca_components").dtype == np.float32


def test_conversion_main_save_quantized_requires_quantize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pca_src = tmp_path / "pca_src.safetensors"
    save_file_np(
        {
            "pca_components": np.ones((2, 4), dtype=np.float32),
            "pca_mean": np.zeros((4,), dtype=np.float32),
            "latent_scale": np.array([1.0], dtype=np.float32),
        },
        str(pca_src),
    )

    monkeypatch.setattr(
        "echo_tts_mlx.conversion.convert_weights",
        lambda **_kwargs: {"output_dir": str(tmp_path / "out"), "outputs": {}, "summary": {}},
    )

    with pytest.raises(ValueError, match="--save-quantized requires --quantize"):
        conversion_main(
            [
                "--components",
                "pca",
                "--pca",
                str(pca_src),
                "--output",
                str(tmp_path / "out"),
                "--quantize",
                "none",
                "--save-quantized",
                str(tmp_path / "quantized"),
            ]
        )


def test_conversion_main_save_quantized_calls_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pca_src = tmp_path / "pca_src.safetensors"
    save_file_np(
        {
            "pca_components": np.ones((2, 4), dtype=np.float32),
            "pca_mean": np.zeros((4,), dtype=np.float32),
            "latent_scale": np.array([1.0], dtype=np.float32),
        },
        str(pca_src),
    )

    seen: dict[str, str] = {}

    class _FakePipe:
        def save_quantized(self, out_dir: Path):
            seen["out_dir"] = str(out_dir)
            return out_dir

    monkeypatch.setattr(
        "echo_tts_mlx.conversion.convert_weights",
        lambda **_kwargs: {"output_dir": str(tmp_path / "out"), "outputs": {}, "summary": {}},
    )
    monkeypatch.setattr(
        "echo_tts_mlx.pipeline.EchoTTS.from_pretrained",
        lambda weights_dir, dtype="float16", quantize="none": _FakePipe(),
    )

    rc = conversion_main(
        [
            "--components",
            "pca",
            "--pca",
            str(pca_src),
            "--output",
            str(tmp_path / "out"),
            "--quantize",
            "8bit",
            "--save-quantized",
            str(tmp_path / "quantized"),
        ]
    )
    assert rc == 0
    assert seen["out_dir"] == str(tmp_path / "quantized")


def test_conversion_main_accepts_new_quantize_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pca_src = tmp_path / "pca_src.safetensors"
    save_file_np(
        {
            "pca_components": np.ones((2, 4), dtype=np.float32),
            "pca_mean": np.zeros((4,), dtype=np.float32),
            "latent_scale": np.array([1.0], dtype=np.float32),
        },
        str(pca_src),
    )

    monkeypatch.setattr(
        "echo_tts_mlx.conversion.convert_weights",
        lambda **_kwargs: {"output_dir": str(tmp_path / "out"), "outputs": {}, "summary": {}},
    )

    rc = conversion_main(
        [
            "--components",
            "pca",
            "--pca",
            str(pca_src),
            "--output",
            str(tmp_path / "out"),
            "--quantize",
            "mixed",
        ]
    )
    assert rc == 0


def test_conversion_main_include_blockwise_disables_pruning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pca_src = tmp_path / "pca_src.safetensors"
    save_file_np(
        {
            "pca_components": np.ones((2, 4), dtype=np.float32),
            "pca_mean": np.zeros((4,), dtype=np.float32),
            "latent_scale": np.array([1.0], dtype=np.float32),
        },
        str(pca_src),
    )

    seen: dict[str, bool] = {}

    def _fake_convert_weights(*, settings, **_kwargs):  # type: ignore[no-untyped-def]
        seen["prune_blockwise"] = bool(settings.prune_blockwise)
        return {"output_dir": str(tmp_path / "out"), "outputs": {}, "summary": {}}

    monkeypatch.setattr("echo_tts_mlx.conversion.convert_weights", _fake_convert_weights)

    rc = conversion_main(
        [
            "--components",
            "pca",
            "--pca",
            str(pca_src),
            "--output",
            str(tmp_path / "out"),
            "--include-blockwise",
        ]
    )
    assert rc == 0
    assert seen["prune_blockwise"] is False
