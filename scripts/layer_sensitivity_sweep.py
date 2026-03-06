#!/usr/bin/env python3
"""Per-layer MXFP4 sensitivity sweep for DiT blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ATTENTION_COMPONENTS = ("wq", "wk", "wv", "wo", "gate")
MLP_COMPONENTS = ("w1", "w2", "w3")
MCD_PROMOTION_THRESHOLD_DB = 5.0


def _compute_mcd(ref_audio: np.ndarray, test_audio: np.ndarray, *, sr: int, n_mfcc: int = 13) -> float:
    import librosa

    ref_mfcc = librosa.feature.mfcc(y=np.asarray(ref_audio, dtype=np.float32), sr=sr, n_mfcc=n_mfcc)
    test_mfcc = librosa.feature.mfcc(y=np.asarray(test_audio, dtype=np.float32), sr=sr, n_mfcc=n_mfcc)
    n = min(ref_mfcc.shape[1], test_mfcc.shape[1])
    if n == 0:
        return float("nan")
    diff = ref_mfcc[:, :n] - test_mfcc[:, :n]
    return float(np.mean(np.sqrt(np.sum(diff * diff, axis=0))))


def _plot_heatmap(output_dir: Path, rows: dict[str, dict[str, float]]) -> Path:
    import matplotlib.pyplot as plt

    data = np.zeros((24, 2), dtype=np.float32)
    for block in range(24):
        data[block, 0] = float(rows.get(f"blocks.{block}.attention", {}).get("max_mcd_db", np.nan))
        data[block, 1] = float(rows.get(f"blocks.{block}.mlp", {}).get("max_mcd_db", np.nan))

    fig, ax = plt.subplots(figsize=(6, 10), constrained_layout=True)
    im = ax.imshow(data, aspect="auto", cmap="magma")
    ax.set_xticks([0, 1], ["attention", "mlp"])
    ax.set_yticks(range(24), [str(i) for i in range(24)])
    ax.set_xlabel("Component")
    ax.set_ylabel("Block")
    ax.set_title("MXFP4 sensitivity (max MCD dB)")
    fig.colorbar(im, ax=ax, label="MCD (dB)")

    out = output_dir / "layer_sensitivity_heatmap.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _prepare_cases(*, include_cloned: bool) -> list[dict[str, Any]]:
    cases = [
        {
            "name": "short_uncond",
            "text": "Short sample: a steady tone of speech for measurement.",
            "seq": 120,
            "speaker": False,
        },
        {
            "name": "medium_uncond",
            "text": (
                "Medium sample: the courier crossed the plaza at dawn, reading each street sign aloud while "
                "the market stalls opened one by one."
            ),
            "seq": 220,
            "speaker": False,
        },
    ]
    if include_cloned:
        cases.append(
            {
                "name": "short_cloned",
                "text": "Cloned sample: matching cadence and speaking style matters here.",
                "seq": 150,
                "speaker": True,
            }
        )
    return cases


def _load_speaker_audio(path: Path | None, *, sample_rate: int) -> np.ndarray | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"speaker reference not found: {path}")
    from echo_tts_mlx.utils import load_audio

    audio, _ = load_audio(path, target_sample_rate=sample_rate)
    return audio


def _component_paths(block: int, component: str) -> list[str]:
    if component == "attention":
        names = ATTENTION_COMPONENTS
    elif component == "mlp":
        names = MLP_COMPONENTS
    else:
        raise ValueError(f"Unknown component: {component}")
    return [f"blocks.{block}.{component}.{name}" for name in names]


def main(argv: list[str] | None = None) -> int:
    from echo_tts_mlx.pipeline import EchoTTS
    from echo_tts_mlx.utils import flatten_audio_for_write

    parser = argparse.ArgumentParser(description="Layer sensitivity sweep for MXFP4 quantization.")
    parser.add_argument("--weights", type=Path, default=Path("weights/converted"), help="Weights directory")
    parser.add_argument("--output-dir", type=Path, default=Path("logs"), help="Output directory")
    parser.add_argument("--speaker-ref", type=Path, default=None, help="Optional speaker reference path")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16", help="DiT dtype")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed")
    parser.add_argument("--steps", type=int, default=8, help="Diffusion steps for sweep")
    parser.add_argument("--threshold-db", type=float, default=MCD_PROMOTION_THRESHOLD_DB, help="8-bit promotion threshold")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_pipe = EchoTTS.from_pretrained(args.weights, dtype=args.dtype, quantize="none")
    speaker_audio = _load_speaker_audio(args.speaker_ref, sample_rate=baseline_pipe.config.sample_rate)
    include_cloned = speaker_audio is not None
    if not include_cloned:
        print("warning: --speaker-ref not provided, skipping cloned case.")
    cases = _prepare_cases(include_cloned=include_cloned)

    baseline_audio: dict[str, np.ndarray] = {}
    for case in cases:
        audio = baseline_pipe.generate(
            text=case["text"],
            speaker_audio=(speaker_audio if case["speaker"] else None),
            sequence_length=int(case["seq"]),
            seed=int(args.seed),
            num_steps=int(args.steps),
            trim_latents=True,
            trim_mode="latent",
        )
        baseline_audio[case["name"]] = flatten_audio_for_write(audio)

    rows: dict[str, dict[str, float]] = {}
    suggested_sensitive_modules: set[str] = set()

    for block in range(24):
        for component in ("attention", "mlp"):
            key = f"blocks.{block}.{component}"
            target_paths = set(_component_paths(block, component))

            pipe = EchoTTS.from_pretrained(args.weights, dtype=args.dtype, quantize="none")

            def pred(path: str, module: Any):
                if not pipe.model._quantize_predicate(path, module):
                    return False
                if path in target_paths:
                    return {"bits": 4, "group_size": 32, "mode": "mxfp4"}
                return False

            pipe.model.nn.quantize(pipe.model.tree, class_predicate=pred)
            pipe.model.quantize_mode = "mxfp4"
            pipe.model._quantized_modules = {name: 4 for name in target_paths}

            mcds: dict[str, float] = {}
            for case in cases:
                audio = pipe.generate(
                    text=case["text"],
                    speaker_audio=(speaker_audio if case["speaker"] else None),
                    sequence_length=int(case["seq"]),
                    seed=int(args.seed),
                    num_steps=int(args.steps),
                    trim_latents=True,
                    trim_mode="latent",
                )
                wave = flatten_audio_for_write(audio)
                ref = baseline_audio[case["name"]]
                mcd = _compute_mcd(ref, wave, sr=pipe.config.sample_rate)
                mcds[case["name"]] = float(mcd)

            max_mcd = float(np.nanmax(np.asarray(list(mcds.values()), dtype=np.float32)))
            rows[key] = {f"mcd_{name}_db": float(value) for name, value in mcds.items()}
            rows[key]["max_mcd_db"] = max_mcd
            if max_mcd > float(args.threshold_db):
                suggested_sensitive_modules.update(target_paths)
            print(f"[sweep] {key}: max_mcd={max_mcd:.3f} dB")

    heatmap_path = _plot_heatmap(out_dir, rows)
    payload = {
        "weights": str(args.weights),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "threshold_db": float(args.threshold_db),
        "speaker_ref": (None if args.speaker_ref is None else str(args.speaker_ref)),
        "results": rows,
        "sensitive_modules": sorted(suggested_sensitive_modules),
        "heatmap_path": str(heatmap_path),
    }
    out_path = out_dir / "layer_sensitivity.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
