from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from benchmarks.cross_impl_protocol import (
    AbstractBenchmarkRunner,
    build_blockwise_cases,
    build_standard_cases,
    get_reference_audio,
    make_synthetic_reference_audio,
    run_cross_impl_suite,
    validate_cross_impl_report,
)


def test_build_standard_cases_has_expected_ids() -> None:
    cases = build_standard_cases()
    assert [case.case_id for case in cases] == ["case_a", "case_b", "case_c", "case_d"]
    assert cases[0].speaker_seconds is None
    assert cases[1].speaker_seconds == 5
    assert cases[2].seq_length == 300
    assert cases[3].seq_length == 640


def test_build_blockwise_cases_has_expected_ids_and_block_sizes() -> None:
    cases = build_blockwise_cases()
    assert [case.case_id for case in cases] == [
        "case_a_bw",
        "case_b_bw",
        "case_c_bw",
        "case_d_bw",
        "case_d_bw_stream",
    ]
    assert cases[0].block_sizes == [100]
    assert cases[1].block_sizes == [76, 76]
    assert cases[2].block_sizes == [128, 128, 44]
    assert all(size % 4 == 0 for case in cases for size in (case.block_sizes or []))


def test_synthetic_reference_audio_is_deterministic() -> None:
    a = make_synthetic_reference_audio(sample_rate=44_100, duration_s=10.0)
    b = make_synthetic_reference_audio(sample_rate=44_100, duration_s=10.0)
    assert a.dtype == np.float32
    assert np.array_equal(a, b)


def test_get_reference_audio_falls_back_to_synthetic_when_download_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request
    from benchmarks import cross_impl_protocol

    monkeypatch.setattr(cross_impl_protocol, "BUNDLED_REFERENCE", tmp_path / "missing.wav")

    def _raise(*_args, **_kwargs):
        raise RuntimeError("network offline")

    monkeypatch.setattr(urllib.request, "urlopen", _raise)
    audio, meta = get_reference_audio(
        cache_dir=tmp_path,
        sample_rate=44_100,
        timeout_s=0.1,
    )
    assert meta["reference"] == "synthetic"
    assert "reference_error" in meta
    assert audio.shape[0] == 44_100 * 10


@dataclass
class _DummyRunner(AbstractBenchmarkRunner):
    sample_rate: int = 44_100

    def run_case(
        self,
        *,
        text: str,
        speaker_audio: np.ndarray | None,
        seq_length: int,
        num_steps: int,
        seed: int,
        cfg_scale_text: float,
        cfg_scale_speaker: float,
        truncation_factor: float,
    ) -> np.ndarray:
        _ = (text, speaker_audio, num_steps, seed, cfg_scale_text, cfg_scale_speaker, truncation_factor)
        samples = max(22_051, seq_length * 400)
        t = np.arange(samples, dtype=np.float32) / np.float32(self.sample_rate)
        return (0.1 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)


def test_run_cross_impl_suite_builds_valid_report(tmp_path: Path) -> None:
    runner = _DummyRunner(
        implementation="dummy",
        version="0.0.1",
        backend="cpu",
        device="test",
        dtype="float32",
    )
    report = run_cross_impl_suite(
        runner=runner,
        cache_dir=tmp_path,
        quality_checks=True,
        det_atol=1e-6,
        det_rtol=1e-5,
        timeout_s=0.1,
        force_synthetic_reference=True,
    )
    validate_cross_impl_report(report)
    assert report["metadata"]["reference"] == "synthetic"
    assert all(report["tier3"][k]["quality_ok"] for k in ("case_a", "case_b", "case_c", "case_d"))
    assert report["tier3_blockwise"]["skipped"] == "blockwise cases are not supported by this runner"


@dataclass
class _DummyBlockwiseRunner(_DummyRunner):
    def run_case_blockwise(
        self,
        *,
        text: str,
        speaker_audio: np.ndarray | None,
        block_sizes: list[int],
        num_steps: int,
        seed: int,
        cfg_scale_text: float,
        cfg_scale_speaker: float,
        truncation_factor: float,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
        _ = (text, speaker_audio, block_sizes, num_steps, seed, cfg_scale_text, cfg_scale_speaker, truncation_factor)
        samples = 44_100
        t = np.arange(samples, dtype=np.float32) / np.float32(self.sample_rate)
        audio = (0.1 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        return audio, {"ttfb_audio_s": 0.5, "ttfb_diffusion_s": 0.4}


def test_run_cross_impl_suite_includes_tier3_blockwise_when_supported(tmp_path: Path) -> None:
    runner = _DummyBlockwiseRunner(
        implementation="dummy",
        version="0.0.2",
        backend="cpu",
        device="test",
        dtype="float32",
    )
    report = run_cross_impl_suite(
        runner=runner,
        cache_dir=tmp_path,
        quality_checks=True,
        det_atol=1e-6,
        det_rtol=1e-5,
        timeout_s=0.1,
        force_synthetic_reference=True,
    )
    assert "tier3_blockwise" in report
    assert "case_d_bw_stream" in report["tier3_blockwise"]
    assert report["tier3_blockwise"]["case_a_bw"]["ttfb_audio_s"] == pytest.approx(0.5)
    assert report["tier3_blockwise"]["case_a_bw"]["overhead_ratio"] > 0.0
