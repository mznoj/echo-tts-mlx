from __future__ import annotations

import warnings

from echo_tts_mlx.model import MlxEchoDiT


class _FakeNN:
    class Linear:
        pass

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.modules = {
            "blocks.0.attention.wq": self.Linear(),
            "blocks.18.attention.wq": self.Linear(),
            "blocks.0.attention.wk_text": self.Linear(),
            "blocks.0.mlp.w1": self.Linear(),
            "cond_module.0": self.Linear(),
            "text_encoder.blocks.0.attention.wq": self.Linear(),
            "speaker_encoder.blocks.0.attention.wq": self.Linear(),
            "blocks.0.attention_adaln.shift_down": self.Linear(),
            "out_proj": self.Linear(),
        }

    def quantize(self, tree, *, group_size=64, bits=8, mode="affine", class_predicate=None):  # type: ignore[no-untyped-def]
        selected = {}
        if class_predicate is not None:
            for path, module in self.modules.items():
                keep = class_predicate(path, module)
                if keep:
                    selected[path] = keep
        self.calls.append(
            {
                "tree": tree,
                "group_size": group_size,
                "bits": bits,
                "mode": mode,
                "selected": selected,
            }
        )


def _fake_model() -> MlxEchoDiT:
    model = MlxEchoDiT.__new__(MlxEchoDiT)
    model.nn = _FakeNN()
    model.tree = object()
    model.quantize_mode = "none"
    model._quantized_modules = {}
    model._quantized_module_params = {}
    return model


def test_apply_quantization_mxfp4_forces_group_size_32() -> None:
    model = _fake_model()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        MlxEchoDiT.apply_quantization(model, mode="mxfp4", group_size=64)

    assert any("group_size=32" in str(w.message) for w in seen)
    call = model.nn.calls[0]
    assert call["bits"] == 4
    assert call["mode"] == "mxfp4"
    assert call["group_size"] == 32
    assert "blocks.0.attention_adaln.shift_down" not in model._quantized_modules
    assert "out_proj" not in model._quantized_modules
    assert model._quantized_modules["blocks.0.attention.wq"] == 4
    assert model._quantized_module_params["blocks.0.attention.wq"]["mode"] == "mxfp4"


def test_apply_quantization_mixed_uses_per_module_policy() -> None:
    model = _fake_model()
    MlxEchoDiT.apply_quantization(model, mode="mixed", group_size=64)

    call = model.nn.calls[0]
    assert call["selected"]["blocks.18.attention.wq"]["bits"] == 8
    assert call["selected"]["blocks.18.attention.wq"]["mode"] == "affine"
    assert call["selected"]["blocks.0.attention.wq"]["bits"] == 4
    assert call["selected"]["blocks.0.attention.wq"]["mode"] == "mxfp4"
    assert call["selected"]["text_encoder.blocks.0.attention.wq"]["bits"] == 8
    assert call["selected"]["speaker_encoder.blocks.0.attention.wq"]["bits"] == 8
    assert call["selected"]["cond_module.0"]["bits"] == 4
    assert model._quantized_modules["blocks.18.attention.wq"] == 8
    assert model._quantized_modules["blocks.0.attention.wq"] == 4
    assert model._quantized_module_params["blocks.18.attention.wq"]["group_size"] == 64
    assert model._quantized_module_params["blocks.0.attention.wq"]["group_size"] == 32


def test_quantize_config_mixed_contains_per_module() -> None:
    model = _fake_model()
    model.quantize_mode = "mixed"
    model._quantized_modules = {
        "blocks.18.attention.wq": 8,
        "blocks.0.attention.wq": 4,
    }
    model._quantized_module_params = {
        "blocks.18.attention.wq": {"bits": 8, "group_size": 64, "mode": "affine"},
        "blocks.0.attention.wq": {"bits": 4, "group_size": 32, "mode": "mxfp4"},
    }

    cfg = model.quantize_config(group_size=999)
    assert cfg.mode == "mixed"
    assert cfg.bits == 8
    assert cfg.group_size == 64
    assert cfg.per_module is True
    assert cfg.modules is not None
    assert cfg.modules["blocks.0.attention.wq"]["mode"] == "mxfp4"
