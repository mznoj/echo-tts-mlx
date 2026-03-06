"""Project CLI entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys
import time

from .config import load_model_config
from .pipeline import QUALITY_PRESETS, EchoTTS, resolve_adaptive_truncation, resolve_quality_preset
from .utils import duration_seconds, load_audio, peak_amplitude


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="echo-tts-mlx")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("convert", help="Convert local upstream checkpoints to MLX-ready weights")
    generate = sub.add_parser("generate", help="Generate speech from text")
    generate.add_argument("--text", type=str, required=True, help="Input text")
    generate.add_argument("--speaker", type=Path, default=None, help="Optional speaker reference audio path")
    generate.add_argument("--output", type=Path, required=True, help="Output WAV path")
    generate.add_argument("--weights", type=Path, default=Path("weights/converted"), help="Converted weights directory")
    generate.add_argument(
        "--preset",
        choices=tuple(QUALITY_PRESETS.keys()),
        default=None,
        help="Quality preset (overrides --steps and --truncation-factor)",
    )
    generate.add_argument("--steps", type=int, default=40, help="Diffusion steps")
    generate.add_argument("--cfg-text", type=float, default=3.0, help="Text guidance scale")
    generate.add_argument(
        "--cfg-speaker",
        type=float,
        default=None,
        help="Speaker guidance scale (default: 8.0 standard, 5.0 blockwise)",
    )
    generate.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    generate.add_argument(
        "--truncation-factor",
        type=str,
        default="0.8",
        help="Initial noise scaling factor (0.0-1.0) or 'auto'",
    )
    generate.add_argument("--dtype", choices=("float16", "float32"), default="float16", help="DiT inference dtype")
    generate.add_argument("--quantize", choices=("none", "8bit", "4bit", "mxfp4", "mixed"), default="none", help="Quantization mode")
    generate.add_argument("--force-speaker", action="store_true", help="Enable speaker KV scaling")
    generate.add_argument("--speaker-scale", type=float, default=1.5, help="Speaker KV scale when --force-speaker is set")
    generate.add_argument("--max-length", type=int, default=640, help="Maximum latent frames")
    generate.add_argument(
        "--blockwise",
        type=str,
        default=None,
        help="Comma-separated latent block sizes (e.g. 128,128,64). Enables blockwise generation.",
    )
    generate.add_argument(
        "--continuation",
        type=Path,
        default=None,
        help="Optional audio file to continue from (requires --blockwise).",
    )
    generate.add_argument("--no-trim", action="store_true", help="Disable latent tail trimming")
    generate.add_argument("--trim-mode", choices=("latent", "energy", "f0"), default="latent", help="Tail trimming mode")
    generate.add_argument("--verbose", action="store_true", help="Show progress and timing")

    info = sub.add_parser("info", help="Print local model and environment info")
    info.add_argument("--weights", type=Path, default=Path("weights/converted"), help="Converted weights directory")
    return parser


def _format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    units = ("B", "KB", "MB", "GB", "TB")
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.2f} {units[idx]}"


def _run_info(weights_dir: Path) -> int:
    files = {
        "config": weights_dir / "config.json",
        "dit": weights_dir / "dit_weights.safetensors",
        "dac": weights_dir / "dac_weights.safetensors",
        "pca": weights_dir / "pca_state.safetensors",
        "weight_map": weights_dir / "weight_map.json",
    }

    print(f"Weights dir: {weights_dir}")
    for name, path in files.items():
        exists = path.exists()
        if exists:
            print(f"  {name:<10} {path} ({_format_size(path.stat().st_size)})")
        else:
            print(f"  {name:<10} MISSING ({path})")

    if files["config"].exists():
        cfg = load_model_config(files["config"])
        print("Model config:")
        print(f"  sample_rate: {cfg.sample_rate}")
        print(f"  latent_size: {cfg.latent_size}")
        print(f"  max_latent_length: {cfg.max_latent_length}")
        print(f"  max_text_length: {cfg.max_text_length}")
        print(f"  max_speaker_latent_length: {cfg.max_speaker_latent_length}")

    print("System:")
    print(f"  platform: {platform.platform()}")
    print(f"  machine: {platform.machine()}")
    print(f"  python: {sys.version.split()[0]}")
    return 0


def _run_generate(args: argparse.Namespace) -> int:
    def _parse_block_sizes(value: str) -> list[int]:
        parts = [p.strip() for p in str(value).split(",") if p.strip()]
        if not parts:
            raise ValueError("--blockwise must provide at least one block size.")
        sizes: list[int] = []
        for part in parts:
            try:
                size = int(part)
            except ValueError as exc:
                raise ValueError(f"Invalid --blockwise size '{part}'. Expected integers like 128,128,64.") from exc
            sizes.append(size)
        return sizes

    def _resolve_cli_truncation(value: str, *, sequence_length: int) -> float | None:
        raw = str(value).strip().lower()
        if raw == "auto":
            return resolve_adaptive_truncation(sequence_length)
        out = float(value)
        return out

    block_sizes = _parse_block_sizes(args.blockwise) if args.blockwise is not None else None
    if args.continuation is not None and block_sizes is None:
        raise ValueError("--continuation requires --blockwise.")
    if args.force_speaker and args.speaker is None:
        raise ValueError("--force-speaker requires --speaker.")
    steps = int(args.steps)
    if block_sizes is None:
        sequence_length = int(args.max_length)
    else:
        sequence_length = int(sum(block_sizes))
    truncation_factor = _resolve_cli_truncation(args.truncation_factor, sequence_length=sequence_length)
    if args.preset is not None:
        steps, truncation_factor = resolve_quality_preset(args.preset, sequence_length=sequence_length)

    if steps < 1:
        raise ValueError("--steps must be >= 1.")
    if args.max_length < 1:
        raise ValueError("--max-length must be >= 1.")
    if truncation_factor is not None and not (0.0 <= truncation_factor <= 1.0):
        raise ValueError("--truncation-factor must be between 0.0 and 1.0.")

    model = EchoTTS.from_pretrained(
        args.weights,
        dtype=args.dtype,
        quantize=args.quantize,
    )

    speaker_audio = None
    if args.speaker is not None:
        speaker_audio, sample_rate = load_audio(args.speaker, target_sample_rate=model.config.sample_rate)
        if sample_rate != model.config.sample_rate:
            raise RuntimeError(
                f"Internal error: speaker sample rate mismatch ({sample_rate} != {model.config.sample_rate})."
            )
    continuation_audio = None
    if args.continuation is not None:
        continuation_audio, sample_rate = load_audio(args.continuation, target_sample_rate=model.config.sample_rate)
        if sample_rate != model.config.sample_rate:
            raise RuntimeError(
                f"Internal error: continuation sample rate mismatch ({sample_rate} != {model.config.sample_rate})."
            )
        print("WARNING: --text must include the full text of the continuation audio, not just the new portion.")

    progress_bar = None
    progress_callback = None
    total_steps = steps
    if args.verbose:
        from tqdm import tqdm

        progress_bar = tqdm(total=total_steps, desc="Diffusion", unit="step")

        def _on_step(done: int, total: int, _t: float, _cfg_active: bool) -> None:
            if progress_bar is not None:
                # Blockwise sampler reports per-block progress (1..num_steps).
                progress_bar.total = total
                progress_bar.n = done
                progress_bar.refresh()

        progress_callback = _on_step
    resolved_cfg_speaker = float(args.cfg_speaker) if args.cfg_speaker is not None else (5.0 if block_sizes is not None else 8.0)

    t0 = time.perf_counter()
    if block_sizes is None:
        audio = model.generate(
            text=args.text,
            speaker_audio=speaker_audio,
            sequence_length=args.max_length,
            seed=args.seed,
            num_steps=steps,
            cfg_scale_text=args.cfg_text,
            cfg_scale_speaker=resolved_cfg_speaker,
            truncation_factor=truncation_factor,
            speaker_kv_scale=(float(args.speaker_scale) if args.force_speaker else None),
            trim_latents=not bool(args.no_trim),
            trim_mode=args.trim_mode,
            progress_callback=progress_callback,
        )
    else:
        def _on_block_complete(block_idx: int, total_blocks: int, _block_audio) -> None:
            if args.verbose:
                print(f"Completed block {block_idx + 1}/{total_blocks}")

        audio = model.generate_blockwise(
            text=args.text,
            block_sizes=block_sizes,
            speaker_audio=speaker_audio,
            continuation_audio=continuation_audio,
            seed=args.seed,
            num_steps=steps,
            cfg_scale_text=args.cfg_text,
            cfg_scale_speaker=resolved_cfg_speaker,
            truncation_factor=truncation_factor,
            speaker_kv_scale=(float(args.speaker_scale) if args.force_speaker else None),
            trim_latents=not bool(args.no_trim),
            trim_mode=args.trim_mode,
            progress_callback=progress_callback,
            on_block_complete=_on_block_complete if args.verbose else None,
        )
    elapsed = time.perf_counter() - t0

    if progress_bar is not None:
        progress_bar.close()

    out_path = model.save_audio(audio, args.output)
    peak = peak_amplitude(audio)
    dur = duration_seconds(audio, sample_rate=model.config.sample_rate)

    print(f"Wrote: {out_path}")
    print(f"Sample rate: {model.config.sample_rate}")
    print(f"Duration: {dur:.3f} s")
    print(f"Peak amplitude: {peak:.6f}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    if args.verbose:
        print(f"Generation time: {elapsed:.3f} s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, rest = parser.parse_known_args(argv)

    if args.command == "convert":
        from .conversion import main as convert_main

        return convert_main(rest)

    if rest:
        parser.error(f"Unrecognized arguments: {' '.join(rest)}")

    if args.command == "info":
        return _run_info(args.weights)
    if args.command == "generate":
        try:
            return _run_generate(args)
        except (FileNotFoundError, RuntimeError, ValueError, NotImplementedError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
