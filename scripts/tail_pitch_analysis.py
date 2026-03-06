#!/usr/bin/env python3
"""Tail pitch diagnostics and truncation sweep tooling.

Run with:
  python scripts/tail_pitch_analysis.py --mode diagnose
  python scripts/tail_pitch_analysis.py --mode sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _parse_seeds(raw: str) -> list[int]:
    out = []
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    if not out:
        raise ValueError("Expected at least one seed.")
    return out


def _parse_truncation(raw: str, *, sequence_length: int) -> float | None:
    from echo_tts_mlx.pipeline import resolve_adaptive_truncation

    key = str(raw).strip().lower()
    if key == "none":
        return None
    if key == "auto":
        return resolve_adaptive_truncation(sequence_length)
    return float(raw)


def _cases(*, include_cloned: bool) -> list[dict[str, Any]]:
    short_prompt = "A clear day sharpens every sound."
    medium_prompt = (
        "Reading passage: The city library opens before sunrise, and by noon the atrium is full of quiet footsteps, "
        "soft page turns, and the steady rhythm of notes taken by hand."
    )
    long_prompt = (
        "Long paragraph: At the edge of the harbor, fog lifted in slow bands while ferries traced careful arcs "
        "between pylons. A street musician tuned a worn guitar, paused, and then played a melody that carried "
        "across the water. Commuters slowed without realizing it, and even the gulls seemed to circle in time "
        "with the final chord."
    )

    base = [
        {"name": "short_uncond", "text": short_prompt, "seq": 100, "speaker": False},
        {"name": "medium_uncond", "text": medium_prompt, "seq": 200, "speaker": False},
        {"name": "long_uncond", "text": long_prompt, "seq": 400, "speaker": False},
        {"name": "max_uncond", "text": long_prompt, "seq": 640, "speaker": False},
    ]
    if include_cloned:
        base.extend(
            [
                {"name": "short_cloned", "text": short_prompt, "seq": 150, "speaker": True},
                {"name": "medium_cloned", "text": medium_prompt, "seq": 300, "speaker": True},
                {"name": "max_cloned", "text": long_prompt, "seq": 640, "speaker": True},
            ]
        )
    return base


def _compute_mcd(ref_audio: np.ndarray, test_audio: np.ndarray, *, sr: int, n_mfcc: int = 13) -> float:
    import librosa

    ref_mfcc = librosa.feature.mfcc(y=np.asarray(ref_audio, dtype=np.float32), sr=sr, n_mfcc=n_mfcc)
    test_mfcc = librosa.feature.mfcc(y=np.asarray(test_audio, dtype=np.float32), sr=sr, n_mfcc=n_mfcc)
    n = min(ref_mfcc.shape[1], test_mfcc.shape[1])
    if n == 0:
        return float("nan")
    diff = ref_mfcc[:, :n] - test_mfcc[:, :n]
    return float(np.mean(np.sqrt(np.sum(diff * diff, axis=0))))


def _plot_f0(sample_dir: Path, stem: str, analysis: dict[str, Any]) -> Path:
    import matplotlib.pyplot as plt

    f0 = np.asarray(analysis["f0_hz"], dtype=np.float32)
    times = np.asarray(analysis["f0_times_s"], dtype=np.float32)
    centers = np.asarray(analysis["window_centers_s"], dtype=np.float32)
    variances = np.asarray(analysis["window_variances"], dtype=np.float32)
    onset_s = analysis.get("onset_time_s")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    axes[0].plot(times, f0, linewidth=1.0)
    if onset_s is not None:
        axes[0].axvline(float(onset_s), color="red", linestyle="--", linewidth=1.0)
    axes[0].set_title("F0 contour (Hz)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Hz")
    axes[0].set_ylim(0, max(550.0, float(np.nanmax(f0)) if np.isfinite(f0).any() else 550.0))

    axes[1].plot(centers, variances, linewidth=1.0)
    if onset_s is not None:
        axes[1].axvline(float(onset_s), color="red", linestyle="--", linewidth=1.0)
    axes[1].set_title("Windowed F0 variance")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Variance")

    out = sample_dir / f"{stem}_f0.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _prepare_speaker_audio(speaker_ref: Path | None, *, sample_rate: int) -> np.ndarray | None:
    if speaker_ref is None:
        return None
    if not speaker_ref.exists():
        raise FileNotFoundError(f"Speaker reference not found: {speaker_ref}")
    from echo_tts_mlx.utils import load_audio

    audio, _ = load_audio(speaker_ref, target_sample_rate=sample_rate)
    return audio


def _run_diagnose(args: argparse.Namespace) -> int:
    from echo_tts_mlx.pipeline import EchoTTS
    from echo_tts_mlx.sampler import analyze_tail_pitch
    from echo_tts_mlx.utils import flatten_audio_for_write, save_audio

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds)
    model = EchoTTS.from_pretrained(args.weights, dtype=args.dtype, quantize=args.quantize)
    speaker_audio = _prepare_speaker_audio(args.speaker_ref, sample_rate=model.config.sample_rate)

    include_cloned = speaker_audio is not None
    if not include_cloned:
        print("warning: --speaker-ref not provided, skipping cloned cases.")
    cases = _cases(include_cloned=include_cloned)

    report_samples: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = {}

    for case in cases:
        for seed in seeds:
            truncation = _parse_truncation(args.truncation_factor, sequence_length=int(case["seq"]))
            wav = model.generate(
                text=case["text"],
                speaker_audio=(speaker_audio if case["speaker"] else None),
                sequence_length=int(case["seq"]),
                seed=int(seed),
                num_steps=int(args.steps),
                truncation_factor=truncation,
                trim_latents=True,
                trim_mode=str(args.trim_mode),
            )
            wave = flatten_audio_for_write(wav)
            analysis = analyze_tail_pitch(
                audio=wave,
                sample_rate=model.config.sample_rate,
                ae_downsample_factor=model.config.ae_downsample_factor,
                f0_variance_ratio_threshold=float(args.f0_ratio_threshold),
                min_voiced_ratio=float(args.min_voiced_ratio),
            )

            sample_id = f"{case['name']}_seed{seed}"
            sample_dir = out_dir / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            wav_path = save_audio(sample_dir / f"{sample_id}.wav", wave, sample_rate=model.config.sample_rate)

            onset_frame = analysis.get("onset_latent_frame")
            onset_sec = analysis.get("onset_window_time_s")
            payload = {
                "sample_id": sample_id,
                "case": case["name"],
                "seed": int(seed),
                "sequence_length": int(case["seq"]),
                "truncation_factor": truncation,
                "trim_mode": str(args.trim_mode),
                "tail_to_body_ratio": float(analysis["tail_to_body_ratio"]),
                "onset_latent_frame": (None if onset_frame is None else int(onset_frame)),
                "onset_seconds": (None if onset_sec is None else float(onset_sec)),
                "audio_path": str(wav_path),
                "f0_times_s": np.asarray(analysis["f0_times_s"], dtype=np.float32).tolist(),
                "f0_hz": np.asarray(analysis["f0_hz"], dtype=np.float32).tolist(),
                "window_centers_s": np.asarray(analysis["window_centers_s"], dtype=np.float32).tolist(),
                "window_variances": np.asarray(analysis["window_variances"], dtype=np.float32).tolist(),
            }
            png_path = _plot_f0(sample_dir, sample_id, payload)
            payload["plot_path"] = str(png_path)
            (sample_dir / f"{sample_id}.json").write_text(json.dumps(payload, indent=2) + "\n")

            report_samples.append(
                {
                    "sample_id": sample_id,
                    "case": case["name"],
                    "seed": int(seed),
                    "sequence_length": int(case["seq"]),
                    "tail_to_body_ratio": float(analysis["tail_to_body_ratio"]),
                    "onset_latent_frame": (None if onset_frame is None else int(onset_frame)),
                    "onset_seconds": (None if onset_sec is None else float(onset_sec)),
                    "audio_path": str(wav_path),
                    "plot_path": str(png_path),
                }
            )
            grouped.setdefault(case["name"], []).append(float(analysis["tail_to_body_ratio"]))
            print(
                f"[diagnose] {sample_id}: ratio={float(analysis['tail_to_body_ratio']):.3f}, "
                f"onset_frame={onset_frame}, onset_s={onset_sec}"
            )

    summary = {}
    for key, values in grouped.items():
        arr = np.asarray(values, dtype=np.float32)
        summary[key] = {
            "count": int(arr.size),
            "tail_ratio_median": float(np.nanmedian(arr)) if arr.size else float("nan"),
            "tail_ratio_mean": float(np.nanmean(arr)) if arr.size else float("nan"),
        }

    report = {
        "mode": "diagnose",
        "weights": str(args.weights),
        "quantize": str(args.quantize),
        "trim_mode": str(args.trim_mode),
        "steps": int(args.steps),
        "seeds": [int(s) for s in seeds],
        "speaker_ref": (None if args.speaker_ref is None else str(args.speaker_ref)),
        "samples": report_samples,
        "summary": summary,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote diagnostic report: {report_path}")
    return 0


def _run_sweep(args: argparse.Namespace) -> int:
    from echo_tts_mlx.pipeline import EchoTTS
    from echo_tts_mlx.sampler import analyze_tail_pitch
    from echo_tts_mlx.utils import flatten_audio_for_write

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds)
    factors = np.arange(float(args.sweep_start), float(args.sweep_end) + 1e-8, float(args.sweep_step), dtype=np.float32)
    if factors.size == 0:
        raise ValueError("Sweep range is empty.")

    model = EchoTTS.from_pretrained(args.weights, dtype=args.dtype, quantize=args.quantize)
    speaker_audio = _prepare_speaker_audio(args.speaker_ref, sample_rate=model.config.sample_rate)

    include_cloned = speaker_audio is not None
    if not include_cloned:
        print("warning: --speaker-ref not provided, skipping cloned cases.")
    cases = _cases(include_cloned=include_cloned)

    baseline_trunc = _parse_truncation(str(args.baseline_truncation), sequence_length=640)
    baseline_audio: dict[str, np.ndarray] = {}
    for case in cases:
        for seed in seeds:
            key = f"{case['name']}_seed{seed}"
            wav = model.generate(
                text=case["text"],
                speaker_audio=(speaker_audio if case["speaker"] else None),
                sequence_length=int(case["seq"]),
                seed=int(seed),
                num_steps=int(args.steps),
                truncation_factor=baseline_trunc,
                trim_latents=True,
                trim_mode="latent",
            )
            baseline_audio[key] = flatten_audio_for_write(wav)

    rows: list[dict[str, Any]] = []
    by_seq: dict[int, list[dict[str, float]]] = {}

    for factor in factors:
        factor_f = float(factor)
        for case in cases:
            seq = int(case["seq"])
            for seed in seeds:
                key = f"{case['name']}_seed{seed}"
                wav = model.generate(
                    text=case["text"],
                    speaker_audio=(speaker_audio if case["speaker"] else None),
                    sequence_length=seq,
                    seed=int(seed),
                    num_steps=int(args.steps),
                    truncation_factor=factor_f,
                    trim_latents=True,
                    trim_mode="latent",
                )
                wave = flatten_audio_for_write(wav)
                analysis = analyze_tail_pitch(
                    audio=wave,
                    sample_rate=model.config.sample_rate,
                    ae_downsample_factor=model.config.ae_downsample_factor,
                    f0_variance_ratio_threshold=float(args.f0_ratio_threshold),
                    min_voiced_ratio=float(args.min_voiced_ratio),
                )

                ref = baseline_audio[key]
                body_samples = max(1, int((max(0.0, (len(ref) / model.config.sample_rate) - 2.0)) * model.config.sample_rate))
                mcd_body = _compute_mcd(ref[:body_samples], wave[:body_samples], sr=model.config.sample_rate)

                rows.append(
                    {
                        "case": case["name"],
                        "seed": int(seed),
                        "sequence_length": seq,
                        "truncation_factor": factor_f,
                        "tail_to_body_ratio": float(analysis["tail_to_body_ratio"]),
                        "onset_latent_frame": (
                            None if analysis.get("onset_latent_frame") is None else int(analysis["onset_latent_frame"])
                        ),
                        "onset_seconds": (
                            None if analysis.get("onset_window_time_s") is None else float(analysis["onset_window_time_s"])
                        ),
                        "mcd_body_db": float(mcd_body),
                    }
                )

    for seq in sorted({int(r["sequence_length"]) for r in rows}):
        seq_rows = [r for r in rows if int(r["sequence_length"]) == seq]
        for factor in sorted({float(r["truncation_factor"]) for r in seq_rows}):
            sub = [r for r in seq_rows if float(r["truncation_factor"]) == factor]
            by_seq.setdefault(seq, []).append(
                {
                    "truncation_factor": float(factor),
                    "tail_ratio_mean": float(np.nanmean([float(x["tail_to_body_ratio"]) for x in sub])),
                    "mcd_body_mean_db": float(np.nanmean([float(x["mcd_body_db"]) for x in sub])),
                }
            )

    recommended: dict[int, float] = {}
    for seq, points in by_seq.items():
        mcd_values = np.asarray([p["mcd_body_mean_db"] for p in points], dtype=np.float32)
        if mcd_values.size == 0:
            continue
        min_mcd = float(np.nanmin(mcd_values))
        feasible = [p for p in points if float(p["mcd_body_mean_db"]) <= (min_mcd + 1.0)]
        if not feasible:
            feasible = points
        feasible.sort(key=lambda p: (float(p["tail_ratio_mean"]), float(p["truncation_factor"])))
        recommended[int(seq)] = float(feasible[0]["truncation_factor"])

    payload = {
        "mode": "sweep",
        "weights": str(args.weights),
        "quantize": str(args.quantize),
        "steps": int(args.steps),
        "seeds": [int(s) for s in seeds],
        "speaker_ref": (None if args.speaker_ref is None else str(args.speaker_ref)),
        "sweep": {
            "start": float(args.sweep_start),
            "end": float(args.sweep_end),
            "step": float(args.sweep_step),
            "baseline_truncation": baseline_trunc,
        },
        "rows": rows,
        "by_sequence_length": by_seq,
        "recommended_truncation": recommended,
    }
    report_path = out_dir / "truncation_sweep.json"
    report_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote sweep report: {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tail pitch diagnostics and truncation sweep.")
    p.add_argument("--mode", choices=("diagnose", "sweep"), required=True, help="Tooling mode")
    p.add_argument("--weights", type=Path, default=Path("weights/converted"), help="Weights directory")
    p.add_argument("--output-dir", type=Path, default=Path("logs/tail_pitch_analysis"), help="Output directory")
    p.add_argument("--speaker-ref", type=Path, default=None, help="Optional speaker reference audio path")
    p.add_argument("--dtype", choices=("float16", "float32"), default="float16", help="DiT dtype")
    p.add_argument(
        "--quantize",
        choices=("none", "8bit", "4bit", "mxfp4", "mixed"),
        default="8bit",
        help="Quantization mode",
    )
    p.add_argument("--steps", type=int, default=32, help="Diffusion steps")
    p.add_argument("--seeds", type=str, default="42,123,7", help="Comma-separated seeds")
    p.add_argument("--trim-mode", choices=("latent", "energy", "f0"), default="latent", help="Trim mode for diagnose")
    p.add_argument("--truncation-factor", type=str, default="0.8", help="Truncation factor or 'auto' for diagnose")
    p.add_argument("--f0-ratio-threshold", type=float, default=2.0, help="Tail ratio threshold for onset detection")
    p.add_argument("--min-voiced-ratio", type=float, default=0.3, help="Minimum voiced-frame ratio per F0 window")
    p.add_argument("--baseline-truncation", type=str, default="0.8", help="Baseline truncation for sweep MCD references")
    p.add_argument("--sweep-start", type=float, default=0.70, help="Sweep start")
    p.add_argument("--sweep-end", type=float, default=1.00, help="Sweep end")
    p.add_argument("--sweep-step", type=float, default=0.02, help="Sweep step")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if int(args.steps) < 1:
        raise ValueError("--steps must be >= 1")

    if args.mode == "diagnose":
        return _run_diagnose(args)
    return _run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
