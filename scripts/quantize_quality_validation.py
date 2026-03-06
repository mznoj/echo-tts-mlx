#!/usr/bin/env python3
"""Quantization quality validation for f16 and quantized inference modes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import tempfile

import librosa
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quantization quality validation.")
    parser.add_argument("--weights", type=Path, default=Path("weights/converted"), help="Converted weights directory")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("quantize_validation_results"),
        help="Output directory for generated WAV files and JSON report",
    )
    parser.add_argument(
        "--speaker-ref",
        type=Path,
        default=None,
        help="Optional reference clip for cloned-voice validation (wav/m4a/etc; requires ffmpeg for non-wav).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Generation seed")
    return parser.parse_args()


def _prepare_reference_wav(path: Path) -> Path | None:
    if not path.exists():
        return None

    if path.suffix.lower() == ".wav":
        return path

    import subprocess

    tmp_dir = Path(tempfile.gettempdir())
    out_wav = tmp_dir / "echo-tts-mlx-quality-ref.wav"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-ar", "44100", "-ac", "1", str(out_wav)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return out_wav if out_wav.exists() else None


def _run(*, weights_dir: Path, results_dir: Path, seed: int, speaker_ref: Path | None) -> None:
    import mlx.core as mx
    from echo_tts_mlx.pipeline import EchoTTS
    from echo_tts_mlx.utils import flatten_audio_for_write, load_audio, save_audio

    sr = 44100
    results_dir.mkdir(parents=True, exist_ok=True)

    # Standard test cases (subset of Tier 3)
    cases: list[dict[str, str | int | None]] = [
        {"name": "short_uncond", "text": "[S1] Hello, this is a test of Echo TTS.", "seq": 100, "speaker": None},
        {
            "name": "medium_uncond",
            "text": "[S1] The quick brown fox jumps over the lazy dog near the riverbank.",
            "seq": 200,
            "speaker": None,
        },
    ]

    prepared_speaker = _prepare_reference_wav(speaker_ref) if speaker_ref is not None else None
    if prepared_speaker is not None:
        cases.append(
            {
                "name": "cloned_short",
                "text": "[S1] Hello, this is a cloned voice test.",
                "seq": 150,
                "speaker": str(prepared_speaker),
            }
        )

    modes = ["none", "8bit", "4bit", "mxfp4", "mixed"]

    def compute_mcd(ref_audio: np.ndarray, test_audio: np.ndarray, n_mfcc: int = 13) -> float:
        ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=n_mfcc)
        test_mfcc = librosa.feature.mfcc(y=test_audio, sr=sr, n_mfcc=n_mfcc)
        min_len = min(ref_mfcc.shape[1], test_mfcc.shape[1])
        diff = ref_mfcc[:, :min_len] - test_mfcc[:, :min_len]
        return float(np.mean(np.sqrt(np.sum(diff**2, axis=0))))

    def generate_audio(model: EchoTTS, case: dict[str, str | int | None]) -> np.ndarray:
        kwargs: dict[str, object] = {
            "text": case["text"],
            "num_steps": 32,
            "seed": seed,
            "sequence_length": case["seq"],
            "cfg_scale_text": 3.0,
            "cfg_scale_speaker": 8.0,
        }
        speaker_path = case["speaker"]
        if isinstance(speaker_path, str):
            audio_np, _ = load_audio(speaker_path, target_sample_rate=sr)
            kwargs["speaker_audio"] = audio_np

        audio = model.generate(**kwargs)
        mx.eval(audio)
        return flatten_audio_for_write(audio)

    print("=" * 70)
    print("QUANTIZATION QUALITY VALIDATION")
    print("=" * 70)

    results: dict[str, object] = {}
    baseline_audio: dict[str, np.ndarray] = {}

    for mode in modes:
        print(f"\n{'-' * 50}")
        print(f"Loading model: quantize={mode}")
        t0 = time.time()
        model = EchoTTS.from_pretrained(weights_dir, dtype="float16", quantize=mode)
        load_time = time.time() - t0
        active_mem = mx.get_active_memory() / 1024 / 1024
        peak_mem = mx.get_peak_memory() / 1024 / 1024
        print(f"  Load: {load_time:.1f}s | Active: {active_mem:.0f} MB | Peak: {peak_mem:.0f} MB")

        mode_results: dict[str, object] = {
            "load_time_s": round(load_time, 1),
            "active_memory_mb": round(active_mem),
            "peak_memory_mb": round(peak_mem),
            "cases": {},
        }

        for case in cases:
            name = str(case["name"])
            seq = int(case["seq"])
            print(f"\n  Case: {name} (seq={seq})")

            # Generate twice for determinism check
            t0 = time.time()
            audio1 = generate_audio(model, case)
            gen_time = time.time() - t0
            audio2 = generate_audio(model, case)

            if mode == "none":
                det_atol, det_rtol = 1e-5, 1e-4
            else:
                det_atol, det_rtol = 1e-4, 1e-3

            deterministic = bool(np.allclose(audio1, audio2, atol=det_atol, rtol=det_rtol))
            max_det_diff = float(np.max(np.abs(audio1 - audio2)))

            peak = float(np.max(np.abs(audio1)))
            duration = len(audio1) / sr

            case_result: dict[str, object] = {
                "gen_time_s": round(gen_time, 1),
                "duration_s": round(duration, 2),
                "peak_amplitude": round(peak, 4),
                "deterministic": deterministic,
                "det_max_diff": f"{max_det_diff:.2e}",
            }

            if mode == "none":
                baseline_audio[name] = audio1
                print(f"    Baseline: {duration:.2f}s, peak={peak:.4f}, det={deterministic}")
            else:
                ref = baseline_audio[name]
                mcd = compute_mcd(ref, audio1)
                peak_ratio = peak / max(float(np.max(np.abs(ref))), 1e-10)
                dur_ratio = duration / max(len(ref) / sr, 1e-10)

                case_result["mcd_db"] = round(mcd, 2)
                case_result["peak_ratio"] = round(peak_ratio, 3)
                case_result["duration_ratio"] = round(dur_ratio, 3)

                mcd_threshold = {
                    "8bit": 2.0,
                    "4bit": 4.0,
                    "mxfp4": 15.0,
                    "mixed": 10.0,
                }[mode]
                case_result["mcd_pass"] = mcd < mcd_threshold
                case_result["peak_ratio_pass"] = 0.8 <= peak_ratio <= 1.2
                case_result["duration_ratio_pass"] = 0.9 <= dur_ratio <= 1.1

                status = "PASS" if all([
                    bool(case_result["mcd_pass"]),
                    bool(case_result["peak_ratio_pass"]),
                    bool(case_result["duration_ratio_pass"]),
                    deterministic,
                ]) else "FAIL"
                print(
                    f"    {status} MCD={mcd:.2f} dB (thresh={mcd_threshold}), "
                    f"peak_ratio={peak_ratio:.3f}, dur_ratio={dur_ratio:.3f}, det={deterministic}"
                )

            save_audio(results_dir / f"{name}_{mode}.wav", audio1, sample_rate=sr)
            case_result["audio_file"] = f"{name}_{mode}.wav"
            (mode_results["cases"])[name] = case_result

        if hasattr(mx, "clear_memory_cache"):
            mx.clear_memory_cache()
        results[mode] = mode_results
        del model

    if "cloned_short" in baseline_audio:
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav

            print(f"\n{'-' * 50}")
            print("Speaker Similarity (Resemblyzer)")
            encoder = VoiceEncoder()

            ref_wav = preprocess_wav(baseline_audio["cloned_short"], source_sr=sr)
            embed_ref = encoder.embed_utterance(ref_wav)

            for mode in [m for m in modes if m != "none"]:
                quant_audio = np.array(librosa.load(str(results_dir / f"cloned_short_{mode}.wav"), sr=sr)[0])
                quant_wav = preprocess_wav(quant_audio, source_sr=sr)
                embed_quant = encoder.embed_utterance(quant_wav)
                similarity = float(np.dot(embed_ref, embed_quant))

                threshold = 0.85 if mode == "8bit" else 0.75
                passed = similarity > threshold
                status = "PASS" if passed else "FAIL"
                print(f"  {mode}: similarity={similarity:.3f} (threshold={threshold}) {status}")

                cases_dict = results[mode]["cases"]
                cases_dict["cloned_short"]["speaker_similarity"] = round(similarity, 3)
                cases_dict["cloned_short"]["speaker_sim_pass"] = passed
        except Exception as exc:
            print(f"  Resemblyzer failed: {exc}")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Mode':>8} | {'Active MB':>10} | {'Load (s)':>8} |", end="")
    for case in cases:
        print(f" {str(case['name'])[:12]:>12}", end="")
    print()
    print("-" * 70)

    for mode in modes:
        r = results[mode]
        print(f"{mode:>8} | {r['active_memory_mb']:>10} | {r['load_time_s']:>8.1f} |", end="")
        for case in cases:
            name = str(case["name"])
            cr = r["cases"][name]
            if mode == "none":
                print(f" {'baseline':>12}", end="")
            else:
                mcd = cr.get("mcd_db", "?")
                print(f" {mcd:>8.2f} dB  " if isinstance(mcd, float) else f" {mcd:>12}", end="")
        print()

    with (results_dir / "quality_validation.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_dir}/quality_validation.json")


def main() -> int:
    args = parse_args()
    _run(weights_dir=args.weights, results_dir=args.results_dir, seed=args.seed, speaker_ref=args.speaker_ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
