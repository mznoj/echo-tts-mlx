from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from echo_tts_mlx.cli import main


WEIGHTS_DIR = Path("weights/converted")
CONFIG_PATH = WEIGHTS_DIR / "config.json"
DIT_PATH = WEIGHTS_DIR / "dit_weights.safetensors"
DAC_PATH = WEIGHTS_DIR / "dac_weights.safetensors"
PCA_PATH = WEIGHTS_DIR / "pca_state.safetensors"
HAS_CONVERTED = CONFIG_PATH.exists() and DIT_PATH.exists() and DAC_PATH.exists() and PCA_PATH.exists()


def _mlx_runtime_available() -> bool:
    if importlib.util.find_spec("mlx") is None:
        return False
    proc = subprocess.run(
        [sys.executable, "-c", "import mlx.core as mx; _ = mx.array([0], dtype=mx.float16)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


HAS_MLX = _mlx_runtime_available()


def test_generate_passes_quantize_mode_to_pipeline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, str] = {}

    class _FakePipeline:
        class _Cfg:
            sample_rate = 44100

        config = _Cfg()

        def generate(self, **_kwargs):
            return np.ones((44100,), dtype=np.float32) * 0.1

        def save_audio(self, _audio, output_path: Path):
            output_path.write_bytes(b"RIFF")
            return output_path

    def _fake_from_pretrained(_weights, *, dtype: str = "float16", quantize: str = "none"):
        seen["dtype"] = dtype
        seen["quantize"] = quantize
        return _FakePipeline()

    monkeypatch.setattr("echo_tts_mlx.cli.EchoTTS.from_pretrained", _fake_from_pretrained)

    rc = main(
        [
            "generate",
            "--text",
            "hello",
            "--output",
            str(tmp_path / "tmp.wav"),
            "--quantize",
            "8bit",
            "--steps",
            "1",
        ]
    )
    _ = capsys.readouterr()
    assert rc == 0
    assert seen["dtype"] == "float16"
    assert seen["quantize"] == "8bit"


def test_generate_requires_speaker_with_force_flag(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "generate",
            "--text",
            "hello",
            "--output",
            "tmp.wav",
            "--force-speaker",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "requires --speaker" in captured.err


def test_generate_continuation_requires_blockwise(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "generate",
            "--text",
            "hello",
            "--output",
            "tmp.wav",
            "--continuation",
            "existing.wav",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "requires --blockwise" in captured.err


def test_generate_preset_overrides_steps_and_truncation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, float | int] = {}

    class _FakePipeline:
        class _Cfg:
            sample_rate = 44100

        config = _Cfg()

        def generate(self, **kwargs):
            seen["num_steps"] = int(kwargs["num_steps"])
            seen["truncation_factor"] = float(kwargs["truncation_factor"])
            return np.ones((44100,), dtype=np.float32) * 0.1

        def save_audio(self, _audio, output_path: Path):
            output_path.write_bytes(b"RIFF")
            return output_path

    def _fake_from_pretrained(_weights, *, dtype: str = "float16", quantize: str = "none"):
        return _FakePipeline()

    monkeypatch.setattr("echo_tts_mlx.cli.EchoTTS.from_pretrained", _fake_from_pretrained)

    rc = main(
        [
            "generate",
            "--text",
            "hello",
            "--output",
            str(tmp_path / "tmp.wav"),
            "--preset",
            "balanced",
            "--steps",
            "99",
            "--truncation-factor",
            "0.1",
        ]
    )
    _ = capsys.readouterr()
    assert rc == 0
    assert seen["num_steps"] == 16
    assert seen["truncation_factor"] == pytest.approx(0.8)


def test_generate_accepts_auto_truncation_factor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, float] = {}

    class _FakePipeline:
        class _Cfg:
            sample_rate = 44100

        config = _Cfg()

        def generate(self, **kwargs):
            seen["truncation_factor"] = float(kwargs["truncation_factor"])
            return np.ones((44100,), dtype=np.float32) * 0.1

        def save_audio(self, _audio, output_path: Path):
            output_path.write_bytes(b"RIFF")
            return output_path

    monkeypatch.setattr(
        "echo_tts_mlx.cli.EchoTTS.from_pretrained",
        lambda _weights, *, dtype="float16", quantize="none": _FakePipeline(),
    )
    monkeypatch.setattr("echo_tts_mlx.cli.resolve_adaptive_truncation", lambda seq: 0.73)

    rc = main(
        [
            "generate",
            "--text",
            "hello",
            "--output",
            str(tmp_path / "tmp.wav"),
            "--truncation-factor",
            "auto",
            "--max-length",
            "333",
        ]
    )
    _ = capsys.readouterr()
    assert rc == 0
    assert seen["truncation_factor"] == pytest.approx(0.73)


def test_generate_blockwise_routes_and_uses_default_cfg_speaker(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _FakePipeline:
        class _Cfg:
            sample_rate = 44100

        config = _Cfg()

        def generate(self, **_kwargs):
            raise AssertionError("standard generate() should not be called in blockwise mode")

        def generate_blockwise(self, **kwargs):
            seen["block_sizes"] = kwargs["block_sizes"]
            seen["cfg_scale_speaker"] = float(kwargs["cfg_scale_speaker"])
            return np.ones((44100,), dtype=np.float32) * 0.1

        def save_audio(self, _audio, output_path: Path):
            output_path.write_bytes(b"RIFF")
            return output_path

    monkeypatch.setattr(
        "echo_tts_mlx.cli.EchoTTS.from_pretrained",
        lambda _weights, *, dtype="float16", quantize="none": _FakePipeline(),
    )

    rc = main(
        [
            "generate",
            "--text",
            "hello",
            "--output",
            str(tmp_path / "tmp.wav"),
            "--blockwise",
            "128,128,64",
            "--steps",
            "1",
        ]
    )
    _ = capsys.readouterr()
    assert rc == 0
    assert seen["block_sizes"] == [128, 128, 64]
    assert seen["cfg_scale_speaker"] == pytest.approx(5.0)


@pytest.mark.skipif(not HAS_CONVERTED, reason="converted weights not present")
@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed or runtime unavailable")
def test_cli_generate_gate_deterministic_and_valid(tmp_path: Path) -> None:
    import soundfile as sf

    out1 = tmp_path / "out1.wav"
    out2 = tmp_path / "out2.wav"

    env = os.environ.copy()
    src_dir = str(Path.cwd() / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_dir if not existing else f"{src_dir}{os.pathsep}{existing}"

    base_cmd = [
        sys.executable,
        "-m",
        "echo_tts_mlx.cli",
        "generate",
        "--text",
        "CLI determinism gate.",
        "--weights",
        str(WEIGHTS_DIR),
        "--steps",
        "2",
        "--seed",
        "123",
        "--max-length",
        "16",
        "--no-trim",
        "--quantize",
        "none",
    ]

    first = subprocess.run(base_cmd + ["--output", str(out1)], env=env, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr

    second = subprocess.run(base_cmd + ["--output", str(out2)], env=env, capture_output=True, text=True, check=False)
    assert second.returncode == 0, second.stderr

    assert out1.read_bytes() == out2.read_bytes(), "seeded outputs are not bitwise-identical"

    audio, sample_rate = sf.read(str(out1), dtype="float32")
    assert int(sample_rate) == 44100

    wave = np.asarray(audio, dtype=np.float32)
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    assert wave.shape[0] / float(sample_rate) >= 0.5
    assert float(np.max(np.abs(wave))) > 0.01
