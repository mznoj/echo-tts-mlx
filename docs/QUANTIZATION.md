# Quantization Guide

Echo-TTS MLX supports runtime and saved-checkpoint quantization via `mlx.nn.quantize()`.

## Modes

| Mode | Policy | Status |
|---|---|---|
| `none` | float16 weights | Default |
| `8bit` | uniform affine 8-bit | Supported |
| `4bit` | legacy mixed policy: DiT/cond 4-bit + encoders 8-bit (affine) | Experimental |
| `mxfp4` | uniform MXFP4 (`bits=4`, `group_size=32`, `mode="mxfp4"`) | Experimental |
| `mixed` | per-module: sensitive + encoders affine 8-bit, others MXFP4 | Experimental |

## CLI Usage

```bash
# Runtime quantization
echo-tts-mlx generate --text "Hello world." --quantize 8bit --output out_8bit.wav
echo-tts-mlx generate --text "Hello world." --quantize mxfp4 --output out_mxfp4.wav
echo-tts-mlx generate --text "Hello world." --quantize mixed --output out_mixed.wav

# Save quantized checkpoint package
echo-tts-mlx convert \
  --output ./weights/converted \
  --quantize mixed \
  --save-quantized ./weights/quantized-mixed
```

## MXFP4 Notes

- MXFP4 requires `group_size=32`.
- If another group size is passed, the model warns and forces 32:
  `MXFP4 requires group_size=32, ignoring provided value`.

## Mixed Mode Policy

Mixed mode uses per-module quantization config:

- Always affine 8-bit:
  - `text_encoder.*`
  - `speaker_encoder.*`
  - `SENSITIVE_MODULES` (initial conservative set: `blocks.18-23.attention.{wq,wk,wv,wo,gate}`)
- Default MXFP4:
  - `blocks.*` and `cond_module.*` modules that pass `_quantize_predicate`

`SENSITIVE_MODULES` is intentionally hardcoded as a conservative starting point and should be refined from `scripts/layer_sensitivity_sweep.py`.

## Config Format

Quantized checkpoints store `quantize_config.json`.

Legacy-compatible fields are always present:

- `mode`
- `bits`
- `group_size`
- `quantized_modules`

Mixed mode additionally stores:

- `per_module: true`
- `modules: { "<module_path>": {"bits": int, "group_size": int, "mode": str} }`

Older configs without `per_module` keep the original uniform load behavior.

## Sensitivity Sweep Tool

Use:

```bash
python scripts/layer_sensitivity_sweep.py --weights weights/converted --output-dir logs
```

Output:

- `logs/layer_sensitivity.json`
- `logs/layer_sensitivity_heatmap.png`

The sweep evaluates one DiT component at a time:

- Attention: `wq`, `wk`, `wv`, `wo`, `gate` (self-attention only)
- MLP: `w1`, `w2`, `w3`

Promotion threshold for sensitive modules is `max MCD > 5.0 dB`.
