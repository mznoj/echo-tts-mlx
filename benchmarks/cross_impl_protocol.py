from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any
import urllib.request

import numpy as np


SCHEMA_VERSION = 1
LJ_SPEECH_URL = "https://keithito.com/LJ-Speech-Dataset/LJ001-0001.wav"
LJ_SPEECH_FILENAME = "LJ001-0001.wav"
BUNDLED_REFERENCE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "reference_audio.wav"


@dataclass(frozen=True)
class StandardCase:
    case_id: str
    text: str
    seq_length: int
    num_steps: int
    speaker_seconds: int | None
    block_sizes: list[int] | None = None
    """Blockwise frame counts per block.

    Only set for cases produced by build_blockwise_cases(). Standard cases use None.
    """


def build_standard_cases() -> list[StandardCase]:
    return [
        StandardCase(
            case_id="case_a",
            text="[S1] Hello world.",
            seq_length=100,
            num_steps=32,
            speaker_seconds=None,
            block_sizes=None,
        ),
        StandardCase(
            case_id="case_b",
            text=(
                "[S1] The quick brown fox jumps over the lazy dog near the riverbank "
                "on a warm summer afternoon."
            ),
            seq_length=150,
            num_steps=32,
            speaker_seconds=5,
            block_sizes=None,
        ),
        StandardCase(
            case_id="case_c",
            text=(
                "[S1] In the beginning, there was nothing but silence. Then came the sound, "
                "a low hum that grew louder and louder until it filled every corner of the room. "
                "Nobody could explain where it came from, but everyone agreed it was beautiful."
            ),
            seq_length=300,
            num_steps=32,
            speaker_seconds=10,
            block_sizes=None,
        ),
        StandardCase(
            case_id="case_d",
            text=(
                "[S1] In the beginning, there was nothing but silence. Then came the sound, "
                "a low hum that grew louder and louder until it filled every corner of the room. "
                "Nobody could explain where it came from, but everyone agreed it was beautiful."
            ),
            seq_length=640,
            num_steps=32,
            speaker_seconds=10,
            block_sizes=None,
        ),
    ]


def build_blockwise_cases() -> list[StandardCase]:
    standard = {case.case_id: case for case in build_standard_cases()}
    return [
        StandardCase(
            case_id="case_a_bw",
            text=standard["case_a"].text,
            seq_length=100,
            num_steps=32,
            speaker_seconds=None,
            block_sizes=[100],
        ),
        StandardCase(
            case_id="case_b_bw",
            text=standard["case_b"].text,
            seq_length=150,
            num_steps=32,
            speaker_seconds=5,
            block_sizes=[76, 76],
        ),
        StandardCase(
            case_id="case_c_bw",
            text=standard["case_c"].text,
            seq_length=300,
            num_steps=32,
            speaker_seconds=10,
            block_sizes=[128, 128, 44],
        ),
        StandardCase(
            case_id="case_d_bw",
            text=standard["case_d"].text,
            seq_length=640,
            num_steps=32,
            speaker_seconds=10,
            block_sizes=[160, 160, 160, 160],
        ),
        StandardCase(
            case_id="case_d_bw_stream",
            text=standard["case_d"].text,
            seq_length=640,
            num_steps=32,
            speaker_seconds=10,
            block_sizes=[64, 192, 192, 192],
        ),
    ]


def make_synthetic_reference_audio(*, sample_rate: int = 44_100, duration_s: float = 10.0) -> np.ndarray:
    n = int(round(sample_rate * duration_s))
    t = np.arange(n, dtype=np.float32) / np.float32(sample_rate)
    signal = 0.5 * np.sin(2.0 * np.pi * 220.0 * t) + 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
    return signal.astype(np.float32)


def _to_mono(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x.mean(axis=1, dtype=np.float32)
    raise ValueError(f"Unsupported audio shape: {tuple(x.shape)}")


def _resample(samples: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    if in_sr == out_sr:
        return np.asarray(samples, dtype=np.float32)

    try:
        import soxr

        return np.asarray(soxr.resample(samples, int(in_sr), int(out_sr)), dtype=np.float32)
    except Exception:
        src_t = np.arange(samples.shape[0], dtype=np.float64) / float(in_sr)
        out_n = int(round(samples.shape[0] * (float(out_sr) / float(in_sr))))
        dst_t = np.arange(out_n, dtype=np.float64) / float(out_sr)
        return np.interp(dst_t, src_t, samples).astype(np.float32)


def _load_audio_file(path: Path, *, sample_rate: int) -> np.ndarray:
    import soundfile as sf

    samples, in_sr = sf.read(str(path), dtype="float32", always_2d=False)
    mono = _to_mono(np.asarray(samples, dtype=np.float32))
    return _resample(mono, int(in_sr), int(sample_rate))


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_to_path(*, url: str, destination: Path, timeout_s: float) -> None:
    with urllib.request.urlopen(url, timeout=float(timeout_s)) as response:
        payload = response.read()
    destination.write_bytes(payload)


def get_reference_audio(
    *,
    cache_dir: Path,
    sample_rate: int,
    timeout_s: float = 20.0,
    force_synthetic_reference: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    wav_path = cache_dir / LJ_SPEECH_FILENAME

    meta: dict[str, Any] = {
        "reference_url": LJ_SPEECH_URL,
        "reference_cache": str(wav_path),
    }

    if force_synthetic_reference:
        meta["reference"] = "synthetic"
        meta["reference_source"] = "forced"
        return make_synthetic_reference_audio(sample_rate=sample_rate, duration_s=10.0), meta

    # 1. Prefer bundled reference audio (no network required)
    if BUNDLED_REFERENCE.exists():
        try:
            audio = _load_audio_file(BUNDLED_REFERENCE, sample_rate=sample_rate)
            meta["reference"] = "bundled"
            meta["reference_source"] = str(BUNDLED_REFERENCE)
            meta["reference_sha256"] = _sha256(BUNDLED_REFERENCE)
            return audio, meta
        except Exception as exc:
            meta["bundled_error"] = f"{type(exc).__name__}: {exc}"
            # Fall through to download/synthetic

    # 2. Try cached or downloaded LJ Speech
    if not wav_path.exists():
        try:
            _download_to_path(url=LJ_SPEECH_URL, destination=wav_path, timeout_s=timeout_s)
            meta["reference_downloaded"] = True
        except Exception as exc:
            meta["reference"] = "synthetic"
            meta["reference_source"] = "download_failed"
            meta["reference_error"] = f"{type(exc).__name__}: {exc}"
            return make_synthetic_reference_audio(sample_rate=sample_rate, duration_s=10.0), meta

    try:
        audio = _load_audio_file(wav_path, sample_rate=sample_rate)
        meta["reference"] = "lj_speech"
        meta["reference_sha256"] = _sha256(wav_path)
        return audio, meta
    except Exception as exc:
        meta["reference"] = "synthetic"
        meta["reference_source"] = "decode_failed"
        meta["reference_error"] = f"{type(exc).__name__}: {exc}"
        return make_synthetic_reference_audio(sample_rate=sample_rate, duration_s=10.0), meta


def _slice_seconds(audio: np.ndarray, *, sample_rate: int, seconds: int | None) -> np.ndarray | None:
    if seconds is None:
        return None
    n = int(sample_rate * seconds)
    return np.asarray(audio[:n], dtype=np.float32)


def _quality_for_pair(
    *,
    a: np.ndarray,
    b: np.ndarray,
    sample_rate: int,
    det_atol: float,
    det_rtol: float,
) -> dict[str, bool]:
    same_shape = a.shape == b.shape
    determinism_ok = bool(same_shape and np.allclose(a, b, atol=det_atol, rtol=det_rtol))
    non_silence_ok = bool((a.size > 0) and (b.size > 0) and np.max(np.abs(a)) > 0.01 and np.max(np.abs(b)) > 0.01)
    duration_ok = bool(
        (a.shape[0] / float(sample_rate)) > 0.5 and (b.shape[0] / float(sample_rate)) > 0.5
    )
    return {
        "determinism_ok": determinism_ok,
        "non_silence_ok": non_silence_ok,
        "duration_ok": duration_ok,
        "quality_ok": bool(determinism_ok and non_silence_ok and duration_ok),
    }


def _safe_float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64))


@dataclass
class AbstractBenchmarkRunner(abc.ABC):
    implementation: str
    version: str
    backend: str
    device: str
    dtype: str
    seed: int = 0
    cfg_scale_text: float = 3.0
    cfg_scale_speaker: float = 8.0
    truncation_factor: float = 0.8
    sample_rate: int = 44_100

    @abc.abstractmethod
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
        """Run one inference case and return mono waveform `(samples,)` float32."""

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
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError("blockwise cases are not supported by this runner")


def _split_case_output(output: Any) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
        audio = np.asarray(output[0], dtype=np.float32).reshape(-1)
        return audio, dict(output[1])
    return np.asarray(output, dtype=np.float32).reshape(-1), {}


def _base_case_id_for_blockwise(case_id: str) -> str | None:
    mapping = {
        "case_a_bw": "case_a",
        "case_b_bw": "case_b",
        "case_c_bw": "case_c",
        "case_d_bw": "case_d",
        "case_d_bw_stream": "case_d",
    }
    return mapping.get(case_id)


def _build_metadata(
    *,
    runner: AbstractBenchmarkRunner,
    reference_meta: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "implementation": runner.implementation,
        "version": runner.version,
        "backend": runner.backend,
        "device": runner.device,
        "os": platform.platform(),
        "dtype": runner.dtype,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "python_version": platform.python_version(),
        "seed": int(runner.seed),
        "cfg_scale_text": float(runner.cfg_scale_text),
        "cfg_scale_speaker": float(runner.cfg_scale_speaker),
        "truncation_factor": float(runner.truncation_factor),
        "num_steps": 32,
    }
    out.update(reference_meta)
    return out


def run_cross_impl_suite(
    *,
    runner: AbstractBenchmarkRunner,
    cache_dir: Path,
    quality_checks: bool = True,
    det_atol: float = 1e-5,
    det_rtol: float = 1e-4,
    timeout_s: float = 20.0,
    force_synthetic_reference: bool = False,
) -> dict[str, Any]:
    reference_audio, reference_meta = get_reference_audio(
        cache_dir=cache_dir,
        sample_rate=runner.sample_rate,
        timeout_s=timeout_s,
        force_synthetic_reference=force_synthetic_reference,
    )

    tier3: dict[str, Any] = {}
    standard_wall_times: dict[str, float] = {}
    for case in build_standard_cases():
        speaker_audio = _slice_seconds(
            reference_audio,
            sample_rate=runner.sample_rate,
            seconds=case.speaker_seconds,
        )

        t0 = time.perf_counter()
        output_a = runner.run_case(
            text=case.text,
            speaker_audio=speaker_audio,
            seq_length=case.seq_length,
            num_steps=case.num_steps,
            seed=runner.seed,
            cfg_scale_text=runner.cfg_scale_text,
            cfg_scale_speaker=runner.cfg_scale_speaker,
            truncation_factor=runner.truncation_factor,
        )
        wall_time_s = time.perf_counter() - t0

        audio_a, extra_a = _split_case_output(output_a)
        audio_duration_s = float(audio_a.shape[0] / float(runner.sample_rate))

        result: dict[str, Any] = {
            "wall_time_s": _safe_float(wall_time_s),
            "realtime_factor": _safe_float(audio_duration_s / max(wall_time_s, 1e-12)),
            "audio_duration_s": _safe_float(audio_duration_s),
        }
        for key, value in extra_a.items():
            if isinstance(value, (float, int, np.floating, np.integer, bool)):
                result[key] = _safe_float(value) if not isinstance(value, bool) else bool(value)

        if quality_checks:
            output_b = runner.run_case(
                text=case.text,
                speaker_audio=speaker_audio,
                seq_length=case.seq_length,
                num_steps=case.num_steps,
                seed=runner.seed,
                cfg_scale_text=runner.cfg_scale_text,
                cfg_scale_speaker=runner.cfg_scale_speaker,
                truncation_factor=runner.truncation_factor,
            )
            audio_b, _ = _split_case_output(output_b)
            checks = _quality_for_pair(
                a=audio_a,
                b=audio_b,
                sample_rate=runner.sample_rate,
                det_atol=det_atol,
                det_rtol=det_rtol,
            )
            result.update(checks)
            result["status"] = "PASS" if checks["quality_ok"] else "FAIL"
        else:
            result["quality_ok"] = True
            result["status"] = "PASS"

        tier3[case.case_id] = result
        standard_wall_times[case.case_id] = float(result["wall_time_s"])

    tier3_blockwise: dict[str, Any] = {}
    blockwise_supported = True
    for case in build_blockwise_cases():
        speaker_audio = _slice_seconds(
            reference_audio,
            sample_rate=runner.sample_rate,
            seconds=case.speaker_seconds,
        )
        if case.block_sizes is None:
            continue

        t0 = time.perf_counter()
        try:
            output_a = runner.run_case_blockwise(
                text=case.text,
                speaker_audio=speaker_audio,
                block_sizes=list(case.block_sizes),
                num_steps=case.num_steps,
                seed=runner.seed,
                cfg_scale_text=runner.cfg_scale_text,
                cfg_scale_speaker=runner.cfg_scale_speaker,
                truncation_factor=runner.truncation_factor,
            )
        except NotImplementedError as exc:
            tier3_blockwise = {"skipped": str(exc) or "blockwise not supported"}
            blockwise_supported = False
            break
        wall_time_s = time.perf_counter() - t0

        audio_a, extra_a = _split_case_output(output_a)
        audio_duration_s = float(audio_a.shape[0] / float(runner.sample_rate))
        result = {
            "wall_time_s": _safe_float(wall_time_s),
            "realtime_factor": _safe_float(audio_duration_s / max(wall_time_s, 1e-12)),
            "audio_duration_s": _safe_float(audio_duration_s),
            "block_sizes": list(case.block_sizes),
        }
        for key, value in extra_a.items():
            if isinstance(value, (float, int, np.floating, np.integer, bool)):
                result[key] = _safe_float(value) if not isinstance(value, bool) else bool(value)

        base_case_id = _base_case_id_for_blockwise(case.case_id)
        if base_case_id is not None and base_case_id in standard_wall_times:
            result["overhead_ratio"] = _safe_float(wall_time_s / max(standard_wall_times[base_case_id], 1e-12))

        if quality_checks:
            output_b = runner.run_case_blockwise(
                text=case.text,
                speaker_audio=speaker_audio,
                block_sizes=list(case.block_sizes),
                num_steps=case.num_steps,
                seed=runner.seed,
                cfg_scale_text=runner.cfg_scale_text,
                cfg_scale_speaker=runner.cfg_scale_speaker,
                truncation_factor=runner.truncation_factor,
            )
            audio_b, _ = _split_case_output(output_b)
            checks = _quality_for_pair(
                a=audio_a,
                b=audio_b,
                sample_rate=runner.sample_rate,
                det_atol=det_atol,
                det_rtol=det_rtol,
            )
            result.update(checks)
            result["status"] = "PASS" if checks["quality_ok"] else "FAIL"
        else:
            result["quality_ok"] = True
            result["status"] = "PASS"
        tier3_blockwise[case.case_id] = result

    if not blockwise_supported and "skipped" not in tier3_blockwise:
        tier3_blockwise = {"skipped": "blockwise not supported"}

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metadata": _build_metadata(runner=runner, reference_meta=reference_meta),
        "tier3": tier3,
    }
    if tier3_blockwise:
        report["tier3_blockwise"] = tier3_blockwise
    validate_cross_impl_report(report)
    return report


def validate_cross_impl_report(report: dict[str, Any]) -> None:
    if int(report.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"Invalid schema_version: {report.get('schema_version')}")

    metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")

    for key in (
        "implementation",
        "version",
        "backend",
        "device",
        "dtype",
        "timestamp",
        "seed",
        "cfg_scale_text",
        "cfg_scale_speaker",
        "truncation_factor",
        "reference",
    ):
        if key not in metadata:
            raise ValueError(f"metadata.{key} is required")

    tier3 = report.get("tier3")
    if not isinstance(tier3, dict):
        raise ValueError("tier3 must be an object")

    for case_id in ("case_a", "case_b", "case_c", "case_d"):
        if case_id not in tier3:
            raise ValueError(f"tier3.{case_id} is required")
        entry = tier3[case_id]
        if not isinstance(entry, dict):
            raise ValueError(f"tier3.{case_id} must be an object")
        for key in ("wall_time_s", "realtime_factor", "audio_duration_s", "quality_ok"):
            if key not in entry:
                raise ValueError(f"tier3.{case_id}.{key} is required")


def _build_cli_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Cross-implementation benchmark protocol")
    parser.add_argument("--dump-cases", action="store_true", help="Print standardized case definitions as JSON")
    parser.add_argument("--output", type=Path, default=Path("cross_impl_cases.json"), help="Output path for --dump-cases")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    if args.dump_cases:
        payload = [
            {
                "case_id": c.case_id,
                "speaker_seconds": c.speaker_seconds,
                "text": c.text,
                "seq_length": c.seq_length,
                "num_steps": c.num_steps,
            }
            for c in build_standard_cases()
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"Wrote standardized case definitions: {args.output}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
