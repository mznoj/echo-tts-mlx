# Audio Quality Controls

This document covers the audio-quality controls for tail trimming and truncation.

## Tail Trimming Modes

`echo-tts-mlx generate` supports three trimming modes:

- `latent` (default): existing latent flattening heuristic only.
- `energy`: latent heuristic plus audio RMS energy boundary detection.
- `f0`: latent + energy + F0 instability onset detection (`librosa` required).

Examples:

```bash
echo-tts-mlx generate --text "Hello world" --trim-mode latent --output out.wav
echo-tts-mlx generate --text "Hello world" --trim-mode energy --output out.wav
echo-tts-mlx generate --text "Hello world" --trim-mode f0 --output out.wav
```

`--no-trim` disables trimming regardless of `--trim-mode`.

### Boundary Logic

Signals are layered incrementally:

- `latent`: `find_flattening_point(...)`
- `energy`: `min(latent_boundary, energy_boundary)`
- `f0`: `min(latent_boundary, energy_boundary, f0_onset_boundary)`

Guardrail:

- If a multi-signal boundary would keep less than 50% of the latent-only retained duration, the pipeline falls back to the latent-only boundary.

## Truncation Factor

CLI truncation supports numeric values and `auto`:

```bash
echo-tts-mlx generate --text "Hello world" --truncation-factor 0.8 --output out.wav
echo-tts-mlx generate --text "Hello world" --truncation-factor auto --output out.wav
```

- `auto` is resolved in the CLI layer from `ADAPTIVE_TRUNCATION` using linear interpolation with endpoint clamping.
- Pipeline APIs still receive `float | None`.
- Quality presets now use `auto` truncation.

Current table values are placeholders:

- `ADAPTIVE_TRUNCATION` is intentionally set to `0.8` across sequence lengths.
- Comment marker in code: `# PLACEHOLDER — update after running truncation sweep`.

## Diagnostic And Sweep Tool

Use `scripts/tail_pitch_analysis.py`:

```bash
python scripts/tail_pitch_analysis.py --mode diagnose --quantize 8bit
python scripts/tail_pitch_analysis.py --mode sweep --quantize 8bit
```

Key options:

- `--speaker-ref <path>`: enables cloned-voice cases; without it, cloned cases are skipped with a warning.
- `--seeds 42,123,7` (default).
- `--truncation-factor auto|<float>` for diagnose.
- `--sweep-start/--sweep-end/--sweep-step` for sweep.

Outputs:

- `logs/tail_pitch_analysis/report.json` (+ per-sample JSON/WAV/plots) for `diagnose`.
- `logs/tail_pitch_analysis/truncation_sweep.json` for `sweep`.

`f0` analysis uses `librosa.pyin` and excludes unvoiced (`NaN`) frames. F0 variance windows require at least 30% voiced frames.
