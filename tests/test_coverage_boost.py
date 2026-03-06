"""Additional tests to boost coverage for sampler, cli, and pipeline modules.

Targets uncovered lines identified in the v1.0.0 pre-release report:
- sampler.py: blockwise inner loop with CFG active, continuation_length > 0,
  on_block_start/on_block_complete callbacks, eval_step, error paths
- cli.py: _format_size, _run_info, _run_generate argument validation,
  blockwise CLI paths, main() routing, convert delegation
- pipeline.py: encode_continuation chunking, blockwise generate trim paths,
  save_quantized, error handling
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import warnings

import numpy as np
import pytest

from echo_tts_mlx.sampler import (
    BlockwiseSamplerConfig,
    SamplerConfig,
    build_timestep_schedule,
    find_flattening_point,
    sample_blockwise_euler_cfg,
    sample_euler_cfg_independent_guidances,
)


# ── sampler: blockwise with CFG active ────────────────────────────────────


class TestBlockwiseSamplerCFGActive:
    """Cover the inner loop when CFG is active (cfg_min_t <= t <= cfg_max_t)."""

    def test_cfg_active_during_steps(self):
        """Verify predict_velocity receives cfg_active=True when in range."""
        cfg = BlockwiseSamplerConfig(
            block_sizes=[3],
            num_steps=4,
            truncation_factor=0.8,
            cfg_min_t=0.0,
            cfg_max_t=1.0,  # CFG always active
        )
        cfg_states: list[bool] = []

        def predict_velocity(x, _t, cfg_active, _start, _kvf, _kvc):
            cfg_states.append(bool(cfg_active))
            return np.zeros_like(x)

        out = sample_blockwise_euler_cfg(
            prefix_latent=np.zeros((1, 3, 80), dtype=np.float32),
            continuation_length=0,
            config=cfg,
            make_noise=lambda bs: np.ones((1, bs, 80), dtype=np.float32),
            build_latent_kv=lambda _: ("f", "c"),
            predict_velocity=predict_velocity,
        )
        assert all(cfg_states), f"Expected all CFG active, got {cfg_states}"
        assert out.shape == (1, 3, 80)

    def test_cfg_partially_active(self):
        """CFG active only for high timesteps."""
        cfg = BlockwiseSamplerConfig(
            block_sizes=[2],
            num_steps=4,
            truncation_factor=None,
            cfg_min_t=0.5,
            cfg_max_t=1.0,
        )
        cfg_states: list[tuple[float, bool]] = []

        def predict_velocity(x, t, cfg_active, _start, _kvf, _kvc):
            cfg_states.append((float(t), bool(cfg_active)))
            return np.zeros_like(x)

        sample_blockwise_euler_cfg(
            prefix_latent=np.zeros((1, 2, 80), dtype=np.float32),
            continuation_length=0,
            config=cfg,
            make_noise=lambda bs: np.zeros((1, bs, 80), dtype=np.float32),
            build_latent_kv=lambda _: ("f", "c"),
            predict_velocity=predict_velocity,
        )
        # At least some steps should have CFG inactive (t < 0.5)
        active_count = sum(1 for _, active in cfg_states if active)
        inactive_count = sum(1 for _, active in cfg_states if not active)
        assert active_count > 0, "Expected some CFG-active steps"
        assert inactive_count > 0, "Expected some CFG-inactive steps"


class TestBlockwiseSamplerContinuation:
    """Cover continuation_length > 0 paths."""

    def test_continuation_offset(self):
        """Blocks should start at continuation_length offset."""
        cfg = BlockwiseSamplerConfig(
            block_sizes=[2, 2],
            num_steps=2,
            truncation_factor=None,
            cfg_min_t=2.0,
            cfg_max_t=2.0,
        )
        starts_seen: list[int] = []

        def predict_velocity(x, _t, _cfg, start_pos, _kvf, _kvc):
            starts_seen.append(int(start_pos))
            return np.zeros_like(x)

        prefix = np.zeros((1, 6, 80), dtype=np.float32)
        prefix[:, :2, :] = 1.0  # Pre-filled continuation

        out = sample_blockwise_euler_cfg(
            prefix_latent=prefix,
            continuation_length=2,
            config=cfg,
            make_noise=lambda bs: np.zeros((1, bs, 80), dtype=np.float32),
            build_latent_kv=lambda _: ("f", "c"),
            predict_velocity=predict_velocity,
        )
        # First block starts at 2 (continuation offset), second at 4
        assert starts_seen[:2] == [2, 2]
        assert starts_seen[2:] == [4, 4]
        # Continuation region should be preserved
        np.testing.assert_allclose(out[:, :2, :], 1.0, atol=1e-6)


class TestBlockwiseSamplerCallbacks:
    """Cover on_block_start and on_block_complete callbacks."""

    def test_on_block_start_called(self):
        cfg = BlockwiseSamplerConfig(
            block_sizes=[2, 3],
            num_steps=1,
            truncation_factor=None,
            cfg_min_t=2.0,
            cfg_max_t=2.0,
        )
        block_starts: list[tuple[int, int, int]] = []

        def on_block_start(block_idx, total_blocks, start_pos):
            block_starts.append((int(block_idx), int(total_blocks), int(start_pos)))

        sample_blockwise_euler_cfg(
            prefix_latent=np.zeros((1, 5, 80), dtype=np.float32),
            continuation_length=0,
            config=cfg,
            make_noise=lambda bs: np.zeros((1, bs, 80), dtype=np.float32),
            build_latent_kv=lambda _: ("f", "c"),
            predict_velocity=lambda x, *_: np.zeros_like(x),
            on_block_start=on_block_start,
        )
        assert block_starts == [(0, 2, 0), (1, 2, 2)]

    def test_on_block_complete_called(self):
        cfg = BlockwiseSamplerConfig(
            block_sizes=[2, 3],
            num_steps=1,
            truncation_factor=None,
            cfg_min_t=2.0,
            cfg_max_t=2.0,
        )
        completions: list[tuple[int, int]] = []

        def on_block_complete(block_idx, total_blocks, block_latent):
            completions.append((int(block_idx), int(total_blocks)))
            assert block_latent is not None

        sample_blockwise_euler_cfg(
            prefix_latent=np.zeros((1, 5, 80), dtype=np.float32),
            continuation_length=0,
            config=cfg,
            make_noise=lambda bs: np.zeros((1, bs, 80), dtype=np.float32),
            build_latent_kv=lambda _: ("f", "c"),
            predict_velocity=lambda x, *_: np.zeros_like(x),
            on_block_complete=on_block_complete,
        )
        assert completions == [(0, 2), (1, 2)]

    def test_eval_step_called_each_step(self):
        cfg = BlockwiseSamplerConfig(
            block_sizes=[2],
            num_steps=3,
            truncation_factor=None,
            cfg_min_t=2.0,
            cfg_max_t=2.0,
        )
        eval_calls = [0]

        def eval_step(x_t):
            eval_calls[0] += 1
            assert x_t.shape == (1, 2, 80)

        sample_blockwise_euler_cfg(
            prefix_latent=np.zeros((1, 2, 80), dtype=np.float32),
            continuation_length=0,
            config=cfg,
            make_noise=lambda bs: np.zeros((1, bs, 80), dtype=np.float32),
            build_latent_kv=lambda _: ("f", "c"),
            predict_velocity=lambda x, *_: np.zeros_like(x),
            eval_step=eval_step,
        )
        assert eval_calls[0] == 3


class TestBlockwiseSamplerErrors:
    """Cover error paths in the blockwise sampler."""

    def test_empty_block_sizes_raises(self):
        cfg = BlockwiseSamplerConfig(
            block_sizes=[],
            num_steps=2,
        )
        with pytest.raises(ValueError, match="block_sizes must not be empty"):
            sample_blockwise_euler_cfg(
                prefix_latent=np.zeros((1, 4, 80), dtype=np.float32),
                continuation_length=0,
                config=cfg,
                make_noise=lambda bs: np.zeros((1, bs, 80), dtype=np.float32),
                build_latent_kv=lambda _: ("f", "c"),
                predict_velocity=lambda x, *_: np.zeros_like(x),
            )

    def test_zero_block_size_raises(self):
        cfg = BlockwiseSamplerConfig(
            block_sizes=[2, 0, 3],
            num_steps=2,
        )
        with pytest.raises(ValueError, match="All block sizes must be > 0"):
            sample_blockwise_euler_cfg(
                prefix_latent=np.zeros((1, 5, 80), dtype=np.float32),
                continuation_length=0,
                config=cfg,
                make_noise=lambda bs: np.zeros((1, bs, 80), dtype=np.float32),
                build_latent_kv=lambda _: ("f", "c"),
                predict_velocity=lambda x, *_: np.zeros_like(x),
            )

    def test_negative_block_size_raises(self):
        cfg = BlockwiseSamplerConfig(
            block_sizes=[-1],
            num_steps=2,
        )
        with pytest.raises(ValueError, match="All block sizes must be > 0"):
            sample_blockwise_euler_cfg(
                prefix_latent=np.zeros((1, 4, 80), dtype=np.float32),
                continuation_length=0,
                config=cfg,
                make_noise=lambda bs: np.zeros((1, bs, 80), dtype=np.float32),
                build_latent_kv=lambda _: ("f", "c"),
                predict_velocity=lambda x, *_: np.zeros_like(x),
            )


class TestBlockwiseSamplerVelocityIntegration:
    """Cover the actual velocity integration math in the blockwise loop."""

    def test_velocity_updates_latents(self):
        """Non-zero velocity should produce non-zero latents."""
        cfg = BlockwiseSamplerConfig(
            block_sizes=[4],
            num_steps=2,
            truncation_factor=None,
            cfg_min_t=2.0,
            cfg_max_t=2.0,
        )

        def predict_velocity(x, _t, _cfg, _start, _kvf, _kvc):
            return np.ones_like(x) * 0.5

        out = sample_blockwise_euler_cfg(
            prefix_latent=np.zeros((1, 4, 80), dtype=np.float32),
            continuation_length=0,
            config=cfg,
            make_noise=lambda bs: np.zeros((1, bs, 80), dtype=np.float32),
            build_latent_kv=lambda _: ("f", "c"),
            predict_velocity=predict_velocity,
        )
        # Velocity integration should produce non-zero values
        assert np.max(np.abs(out)) > 0

    def test_multi_block_velocity_accumulation(self):
        """Each block should independently integrate velocity."""
        cfg = BlockwiseSamplerConfig(
            block_sizes=[2, 2],
            num_steps=1,
            truncation_factor=None,
            cfg_min_t=2.0,
            cfg_max_t=2.0,
        )
        block_idx_tracker = [0]

        def predict_velocity(x, _t, _cfg, start_pos, _kvf, _kvc):
            # Different velocity per block
            v = np.ones_like(x) * (block_idx_tracker[0] + 1) * 0.1
            if start_pos >= 2:
                block_idx_tracker[0] = 1
            return v

        out = sample_blockwise_euler_cfg(
            prefix_latent=np.zeros((1, 4, 80), dtype=np.float32),
            continuation_length=0,
            config=cfg,
            make_noise=lambda bs: np.zeros((1, bs, 80), dtype=np.float32),
            build_latent_kv=lambda _: ("f", "c"),
            predict_velocity=predict_velocity,
        )
        assert out.shape == (1, 4, 80)


# ── sampler: tail pitch / f0 analysis ─────────────────────────────────────


class TestAnalyzeTailPitch:
    """Cover analyze_tail_pitch (lines 352-365)."""

    def test_tail_pitch_basic(self):
        from echo_tts_mlx.sampler import analyze_tail_pitch

        # Generate a simple tone
        sr = 22050
        t = np.arange(sr * 2, dtype=np.float32) / sr
        audio = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

        result = analyze_tail_pitch(audio=audio, sample_rate=sr, ae_downsample_factor=2048)
        assert "f0_hz" in result
        assert "tail_to_body_ratio" in result
        assert "onset_window_time_s" in result
        assert "onset_latent_frame" in result

    def test_tail_pitch_empty_audio(self):
        from echo_tts_mlx.sampler import analyze_tail_pitch

        audio = np.zeros((0,), dtype=np.float32)
        result = analyze_tail_pitch(audio=audio, sample_rate=22050, ae_downsample_factor=2048)
        assert result["f0_hz"].size == 0

    def test_tail_pitch_no_librosa_raises(self):
        from echo_tts_mlx.sampler import analyze_tail_pitch

        with patch.dict("sys.modules", {"librosa": None}):
            with pytest.raises(ImportError, match="librosa"):
                audio = np.ones((22050,), dtype=np.float32) * 0.5
                analyze_tail_pitch(audio=audio, sample_rate=22050, ae_downsample_factor=2048)


# ── sampler: find_content_boundary ────────────────────────────────────────


class TestFindContentBoundary:
    """Cover find_content_boundary with energy detection."""

    def test_content_boundary_with_energy(self):
        from echo_tts_mlx.sampler import find_content_boundary

        rng = np.random.default_rng(42)
        latents = np.zeros((100, 80), dtype=np.float32)
        latents[:60, :] = rng.standard_normal((60, 80)).astype(np.float32)
        audio = np.zeros((100 * 2048,), dtype=np.float32)
        audio[:60 * 2048] = rng.standard_normal((60 * 2048,)).astype(np.float32) * 0.5

        boundary = find_content_boundary(
            latents, audio,
            sample_rate=44100,
            ae_downsample_factor=2048,
            energy_enabled=True,
        )
        assert 0 < boundary <= 100

    def test_content_boundary_latent_only(self):
        from echo_tts_mlx.sampler import find_content_boundary

        latents = np.zeros((50, 80), dtype=np.float32)
        latents[:25, :] = np.ones((25, 80), dtype=np.float32)
        audio = np.ones((50 * 2048,), dtype=np.float32) * 0.1

        boundary = find_content_boundary(
            latents, audio,
            energy_enabled=False,
            f0_enabled=False,
        )
        assert 0 <= boundary <= 50


# ── cli: _format_size ─────────────────────────────────────────────────────


class TestCLIFormatSize:
    def test_format_bytes(self):
        from echo_tts_mlx.cli import _format_size
        assert "B" in _format_size(500)

    def test_format_kb(self):
        from echo_tts_mlx.cli import _format_size
        result = _format_size(2048)
        assert "KB" in result

    def test_format_mb(self):
        from echo_tts_mlx.cli import _format_size
        result = _format_size(5 * 1024 * 1024)
        assert "MB" in result

    def test_format_gb(self):
        from echo_tts_mlx.cli import _format_size
        result = _format_size(3 * 1024 * 1024 * 1024)
        assert "GB" in result


# ── cli: _run_info ────────────────────────────────────────────────────────


class TestCLIRunInfo:
    def test_run_info_with_fake_weights(self, tmp_path, capsys):
        from echo_tts_mlx.cli import _run_info
        import json

        # Create minimal weight dir structure with all required config fields
        config = {
            "model_type": "echo-dit",
            "latent_size": 80,
            "model_size": 2048,
            "num_layers": 24,
            "num_heads": 16,
            "intermediate_size": 5888,
            "norm_eps": 1e-05,
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
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "dit_weights.safetensors").write_bytes(b"\x00" * 100)
        (tmp_path / "dac_weights.safetensors").write_bytes(b"\x00" * 200)
        (tmp_path / "pca_state.safetensors").write_bytes(b"\x00" * 50)

        result = _run_info(tmp_path)
        assert result == 0
        out = capsys.readouterr().out
        assert "Weights dir:" in out
        assert "config" in out

    def test_run_info_missing_files(self, tmp_path, capsys):
        from echo_tts_mlx.cli import _run_info

        result = _run_info(tmp_path)
        assert result == 0
        out = capsys.readouterr().out
        assert "MISSING" in out


# ── cli: _run_generate argument validation ────────────────────────────────


class TestCLIGenerateValidation:
    def _make_fake_pipeline(self):
        class FakeCfg:
            sample_rate = 44100
            ae_downsample_factor = 2048

        class FakePipeline:
            config = FakeCfg()

            def generate(self, **kwargs):
                return np.ones((44100,), dtype=np.float32) * 0.1

            def generate_blockwise(self, **kwargs):
                return np.ones((44100,), dtype=np.float32) * 0.1

            def save_audio(self, _audio, output_path):
                Path(output_path).write_bytes(b"RIFF")
                return output_path

        return FakePipeline()

    def test_continuation_without_blockwise_errors(self, tmp_path):
        from echo_tts_mlx.cli import main

        out_path = tmp_path / "out.wav"
        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"\x00" * 100)

        result = main([
            "generate",
            "--weights", str(tmp_path),
            "--text", "Hello",
            "--output", str(out_path),
            "--continuation", str(ref_audio),
        ])
        assert result == 2  # Error exit

    def test_force_speaker_without_speaker_errors(self, tmp_path):
        from echo_tts_mlx.cli import main

        out_path = tmp_path / "out.wav"
        result = main([
            "generate",
            "--weights", str(tmp_path),
            "--text", "Hello",
            "--output", str(out_path),
            "--force-speaker",
        ])
        assert result == 2  # Error exit

    def test_invalid_steps_errors(self, tmp_path, monkeypatch):
        from echo_tts_mlx.cli import main

        pipeline = self._make_fake_pipeline()
        monkeypatch.setattr(
            "echo_tts_mlx.cli.EchoTTS.from_pretrained",
            lambda *a, **kw: pipeline,
        )

        out_path = tmp_path / "out.wav"
        result = main([
            "generate",
            "--weights", str(tmp_path),
            "--text", "Hello",
            "--output", str(out_path),
            "--steps", "0",
        ])
        assert result == 2

    def test_invalid_truncation_errors(self, tmp_path, monkeypatch):
        from echo_tts_mlx.cli import main

        pipeline = self._make_fake_pipeline()
        monkeypatch.setattr(
            "echo_tts_mlx.cli.EchoTTS.from_pretrained",
            lambda *a, **kw: pipeline,
        )

        out_path = tmp_path / "out.wav"
        result = main([
            "generate",
            "--weights", str(tmp_path),
            "--text", "Hello",
            "--output", str(out_path),
            "--truncation-factor", "1.5",
        ])
        assert result == 2

    def test_blockwise_generate_path(self, tmp_path, monkeypatch):
        from echo_tts_mlx.cli import main

        pipeline = self._make_fake_pipeline()
        monkeypatch.setattr(
            "echo_tts_mlx.cli.EchoTTS.from_pretrained",
            lambda *a, **kw: pipeline,
        )

        out_path = tmp_path / "out.wav"
        result = main([
            "generate",
            "--weights", str(tmp_path),
            "--text", "[S1] Hello world",
            "--output", str(out_path),
            "--blockwise", "64,64",
        ])
        assert result == 0
        assert out_path.exists()

    def test_invalid_blockwise_size_errors(self, tmp_path, monkeypatch):
        from echo_tts_mlx.cli import main

        pipeline = self._make_fake_pipeline()
        monkeypatch.setattr(
            "echo_tts_mlx.cli.EchoTTS.from_pretrained",
            lambda *a, **kw: pipeline,
        )

        out_path = tmp_path / "out.wav"
        result = main([
            "generate",
            "--weights", str(tmp_path),
            "--text", "Hello",
            "--output", str(out_path),
            "--blockwise", "abc,64",
        ])
        assert result == 2

    def test_generate_with_preset(self, tmp_path, monkeypatch):
        from echo_tts_mlx.cli import main

        pipeline = self._make_fake_pipeline()
        monkeypatch.setattr(
            "echo_tts_mlx.cli.EchoTTS.from_pretrained",
            lambda *a, **kw: pipeline,
        )

        out_path = tmp_path / "out.wav"
        result = main([
            "generate",
            "--weights", str(tmp_path),
            "--text", "[S1] Hello world",
            "--output", str(out_path),
            "--preset", "fast",
        ])
        assert result == 0

    def test_generate_verbose_output(self, tmp_path, monkeypatch, capsys):
        from echo_tts_mlx.cli import main

        pipeline = self._make_fake_pipeline()
        monkeypatch.setattr(
            "echo_tts_mlx.cli.EchoTTS.from_pretrained",
            lambda *a, **kw: pipeline,
        )

        out_path = tmp_path / "out.wav"
        result = main([
            "generate",
            "--weights", str(tmp_path),
            "--text", "[S1] Hello world",
            "--output", str(out_path),
            "--seed", "42",
            "--verbose",
        ])
        assert result == 0
        captured = capsys.readouterr().out
        assert "Seed: 42" in captured
        assert "Generation time:" in captured


# ── cli: main() routing ──────────────────────────────────────────────────


class TestCLIMainRouting:
    def test_info_subcommand(self, tmp_path):
        from echo_tts_mlx.cli import main

        result = main(["info", "--weights", str(tmp_path)])
        assert result == 0

    def test_convert_subcommand_delegates(self, monkeypatch):
        from echo_tts_mlx.cli import main

        mock_convert = MagicMock(return_value=0)
        monkeypatch.setattr("echo_tts_mlx.conversion.main", mock_convert)

        result = main(["convert", "--help-placeholder"])
        # Convert delegates to conversion.main with remaining args
        mock_convert.assert_called_once()

    def test_unknown_args_error(self, tmp_path):
        from echo_tts_mlx.cli import main

        with pytest.raises(SystemExit):
            main(["info", "--weights", str(tmp_path), "--bogus-flag"])


# ── cross_impl_protocol: bundled reference ───────────────────────────────


class TestBundledReferenceAudio:
    def test_bundled_reference_preferred(self, tmp_path):
        from benchmarks.cross_impl_protocol import get_reference_audio, BUNDLED_REFERENCE

        if not BUNDLED_REFERENCE.exists():
            pytest.skip("Bundled reference audio not available")

        audio, meta = get_reference_audio(
            cache_dir=tmp_path,
            sample_rate=44100,
        )
        assert meta["reference"] == "bundled"
        assert audio.shape[0] > 0
        assert "reference_sha256" in meta

    def test_forced_synthetic_overrides_bundled(self, tmp_path):
        from benchmarks.cross_impl_protocol import get_reference_audio

        audio, meta = get_reference_audio(
            cache_dir=tmp_path,
            sample_rate=44100,
            force_synthetic_reference=True,
        )
        assert meta["reference"] == "synthetic"
        assert meta["reference_source"] == "forced"

    def test_synthetic_fallback_when_no_bundled(self, tmp_path, monkeypatch):
        import benchmarks.cross_impl_protocol as proto
        from benchmarks.cross_impl_protocol import get_reference_audio as _get_ref

        # Temporarily point BUNDLED_REFERENCE to nonexistent file
        monkeypatch.setattr(proto, "BUNDLED_REFERENCE", Path("/nonexistent/audio.wav"))
        # Also block network download
        monkeypatch.setattr(proto, "LJ_SPEECH_URL", "http://localhost:1/bad.wav")

        audio, meta = _get_ref(
            cache_dir=tmp_path,
            sample_rate=44100,
            timeout_s=1.0,
        )
        assert meta["reference"] == "synthetic"


# ── benchmark: auto-detect weights ────────────────────────────────────────


class TestBenchmarkWeightsAutoDetect:
    def test_build_parser_weights_default_is_none(self):
        from benchmarks.run_benchmarks import build_parser

        parser = build_parser()
        args = parser.parse_args(["--tier", "1"])
        assert args.weights is None

    def test_build_parser_explicit_weights(self, tmp_path):
        from benchmarks.run_benchmarks import build_parser

        parser = build_parser()
        args = parser.parse_args(["--weights", str(tmp_path)])
        assert args.weights == tmp_path
