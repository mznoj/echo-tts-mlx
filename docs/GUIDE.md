# User Guide

A practical guide to generating speech with Echo-TTS MLX on Apple Silicon.

## Setup

### 1. Install

```bash
git clone https://github.com/mattznojassist/echo-tts-mlx.git
cd echo-tts-mlx
pip install .
```

Or with requirements.txt:

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Download and convert weights

```bash
# Download from HuggingFace
huggingface-cli download jordand/echo-tts-base --local-dir weights/upstream
huggingface-cli download jordand/fish-s1-dac-min --local-dir weights/upstream

# Convert to MLX format
echo-tts-mlx convert \
  --dit weights/upstream/dit_model.safetensors \
  --dac weights/upstream/dac_model.safetensors \
  --pca weights/upstream/pca_state.safetensors \
  --output weights/converted/
```

### 3. Verify

```bash
echo-tts-mlx info --weights weights/converted
```

---

## Generating Speech

### Basic generation

```bash
echo-tts-mlx generate \
  --text "Hello, this is Echo TTS running on Apple Silicon." \
  --preset quality \
  --output hello.wav
```

### Voice cloning

Provide a reference audio clip (WAV, up to 5 minutes, but 5-10 seconds works well):

```bash
echo-tts-mlx generate \
  --text "This uses a cloned voice from the reference audio." \
  --speaker reference.wav \
  --preset quality \
  --output cloned.wav
```

### Faster generation with quantization

8-bit quantization uses ~34% less memory and runs 1.2-1.4x faster with minimal quality loss:

```bash
echo-tts-mlx generate \
  --text "Quantized inference is faster." \
  --quantize 8bit \
  --preset quality \
  --output fast.wav
```

### Reproducible output

Set a seed for deterministic generation:

```bash
echo-tts-mlx generate \
  --text "Same seed, same output every time." \
  --seed 42 \
  --preset quality \
  --output reproducible.wav
```

---

## CLI Reference

### `echo-tts-mlx generate`

| Flag | Default | Description |
|---|---|---|
| `--text` | *(required)* | Input text to synthesize |
| `--output` | *(required)* | Output WAV file path |
| `--speaker` | None | Speaker reference audio for voice cloning |
| `--weights` | `weights/converted` | Path to converted model weights |
| `--preset` | None | Quality preset: `draft`, `fast`, `balanced`, `quality`, `ultra` |
| `--steps` | 40 | Diffusion steps (overridden by `--preset`) |
| `--cfg-text` | 3.0 | Text classifier-free guidance scale |
| `--cfg-speaker` | 8.0 | Speaker classifier-free guidance scale |
| `--seed` | random | Random seed for reproducibility |
| `--truncation-factor` | auto | Initial noise scaling (0.0–1.0 or `auto`) |
| `--dtype` | `float16` | Inference precision: `float16` or `float32` |
| `--quantize` | `none` | Quantization: `none`, `8bit`, `4bit`, `mxfp4`, `mixed` |
| `--force-speaker` | off | Enable speaker KV scaling (fixes wrong-speaker issues) |
| `--speaker-scale` | 1.5 | KV scale factor when `--force-speaker` is set |
| `--max-length` | 640 | Maximum latent frames (640 ≈ 30 seconds) |
| `--trim-mode` | `latent` | Tail trimming: `latent`, `energy`, or `f0` |
| `--no-trim` | off | Disable tail trimming entirely |
| `--verbose` | off | Show progress and timing breakdown |

### `echo-tts-mlx convert`

Converts upstream PyTorch checkpoints to MLX format. Requires `pip install ".[convert]"`.

### `echo-tts-mlx info`

Prints model info and environment details. Useful for debugging.

| Flag | Default | Description |
|---|---|---|
| `--weights` | `weights/converted` | Path to converted weights |

---

## Choosing a Preset

Presets control diffusion steps and truncation. More steps = better quality but slower.

| Preset | Steps | When to use |
|---|---|---|
| `draft` | 4 | Rapid prototyping, testing pipeline |
| `fast` | 8 | Experimental |
| `balanced` | 16 | OK quality |
| `quality` | 32 | Good quality |
| `ultra` | 40 | Maximum quality, matches upstream default |

**Recommendation:** Start with `quality`. Drop to `balanced` if you need speed. Use `ultra` when quality matters most and you can wait.

You can also set steps manually — `--steps 24` works fine if you want something between presets.

---

## Choosing a Quantization Mode

| Mode | Memory | Speed | Quality | When to use |
|---|---|---|---|---|
| `none` (f16) | ~6 GB | Baseline | Best | Quality-critical work |
| `8bit` | ~4 GB | 1.2-1.4x faster | Near-baseline | **General use** — best balance |
| `mxfp4` | ~3.5 GB | 1.4-1.5x faster | Experimental | Research, memory-constrained |
| `mixed` | ~3.8 GB | 1.3-1.4x faster | Experimental | Research |
| `4bit` | ~3.5 GB | Variable | Poor | Not recommended |

**Recommendation:** Use `--quantize 8bit` unless you specifically need maximum quality. The quality difference is negligible for most use cases.

### Saving a quantized checkpoint

To avoid re-quantizing on every run:

```bash
echo-tts-mlx convert \
  --output weights/converted \
  --quantize 8bit \
  --save-quantized weights/quantized-8bit
```

Then use `--weights weights/quantized-8bit` for subsequent generation.

---

## Blockwise Generation

Blockwise mode generates audio in sequential blocks, enabling streaming playback and audio continuations. It requires weights converted with `--include-blockwise`.

### Setup

```bash
# Convert weights with blockwise modules included
echo-tts-mlx convert \
  --dit weights/upstream/dit_model.safetensors \
  --dac weights/upstream/dac_model.safetensors \
  --pca weights/upstream/pca_state.safetensors \
  --output weights/converted-blockwise/ \
  --include-blockwise
```

### Basic blockwise generation

```bash
echo-tts-mlx generate \
  --text "[S1] Hello, this is blockwise generation." \
  --weights weights/converted-blockwise/ \
  --blockwise 128,128,64 \
  --preset quality \
  --output blockwise.wav
```

Block sizes are comma-separated latent frame counts. Their sum must be ≤ `max_latent_length` (640 / ~30 seconds in current checkpoints). Recommended minimum per block: 32 frames (~1.5s).

### Audio continuation

Continue from existing audio by providing a reference file:

```bash
echo-tts-mlx generate \
  --text "[S1] Hello world. How are you today?" \
  --weights weights/converted-blockwise/ \
  --blockwise 128,128 \
  --continuation existing.wav \
  --output continued.wav
```

**Important:** `--text` must include the full text of both the continuation audio and the new portion. If the continuation says "Hello world" and you want to generate "How are you today?", the text must be the full concatenation.

### Python API

```python
from echo_tts_mlx import EchoTTS

model = EchoTTS.from_pretrained("weights/converted-blockwise", dtype="float16")

audio = model.generate_blockwise(
    text="[S1] This is generated in blocks.",
    block_sizes=[128, 128, 64],
    num_steps=32,
    cfg_scale_speaker=5.0,
    seed=42,
)

model.save_audio(audio, "blockwise_output.wav")
```

### Performance notes

Blockwise trades total throughput for lower time-to-first-audio. Each block runs its own full diffusion loop, so 2 blocks ≈ 2× total compute. Use standard mode for batch/offline generation; use blockwise for streaming or continuation use cases.

---

## Text Formatting

Echo-TTS uses [WhisperD](https://huggingface.co/jordand/whisper-d-v1a) text formatting:

- **`[S1]` prefix** is added automatically if missing
- **Commas** create pauses
- **Exclamation points** increase expressiveness (but may reduce quality)
- **Colons, semicolons, em-dashes** are normalized to commas

**Example:**
```
"[S1] Hello, and welcome. This is a demonstration of Echo TTS, running natively on Apple Silicon."
```

### Text length and speaking rate

The model generates up to **30 seconds** of audio (640 latent frames). It tries to fit your text into that duration:

- **Long text** → faster speaking rate
- **Short text** → shorter output with automatic padding

To generate only a portion of long text, reduce `--max-length`:

```bash
# Generate only the first ~15 seconds worth
echo-tts-mlx generate \
  --text "Very long text here..." \
  --max-length 320 \
  --preset quality \
  --output partial.wav
```

---

## Speaker Reference Tips

- **Length:** 5–10 seconds works well. Up to 5 minutes is supported, but longer clips don't necessarily improve quality.
- **Quality:** Clean audio with minimal background noise gives the best results.
- **Format:** WAV recommended. Other formats may work if your system has the right decoders.

### Fixing wrong-speaker output

Sometimes out-of-distribution text causes the model to generate a different speaker than the reference. Use `--force-speaker` to fix this:

```bash
echo-tts-mlx generate \
  --text "Unusual or out-of-distribution text" \
  --speaker reference.wav \
  --force-speaker \
  --preset quality \
  --output forced.wav
```

The default scale is 1.5. If that's too strong (artifacts, over-conditioning), try lower values:

```bash
--force-speaker --speaker-scale 1.1
--force-speaker --speaker-scale 1.3
```

Values above 2.0 are not recommended.

---

## Tail Trimming

The model sometimes generates trailing silence or artifacts after the speech content. Trimming modes handle this automatically:

| Mode | Method | Requirement |
|---|---|---|
| `latent` (default) | Detects flattening in latent space | None |
| `energy` | Latent + audio RMS energy boundary | None |
| `f0` | Latent + energy + pitch instability | `librosa` (`pip install librosa`) |

Each mode builds on the previous — `f0` uses all three signals for the most aggressive trimming.

Use `--no-trim` to disable trimming and get the raw model output.

---

## Python API

```python
from echo_tts_mlx import EchoTTS

# Load model (with optional quantization)
model = EchoTTS.from_pretrained(
    "weights/converted",
    dtype="float16",
    quantize="8bit",
)

# Load speaker reference (optional)
speaker_audio, sr = model.load_audio("reference.wav")

# Generate speech
audio = model.generate(
    text="Hello from the Python API.",
    speaker_audio=speaker_audio,  # None for unconditioned
    num_steps=32,
    cfg_scale_text=3.0,
    cfg_scale_speaker=8.0,
    trim_mode="energy",
    seed=42,
)

# Save output
model.save_audio(audio, "output.wav")
```

### Key parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | str | *(required)* | Input text |
| `speaker_audio` | array/None | None | Speaker reference (from `load_audio`) |
| `num_steps` | int | 40 | Diffusion steps |
| `cfg_scale_text` | float | 3.0 | Text guidance strength |
| `cfg_scale_speaker` | float | 8.0 | Speaker guidance strength |
| `sequence_length` | int | 640 | Max latent frames |
| `truncation_factor` | float/None | None | Initial noise scaling |
| `trim_latents` | bool | True | Enable tail trimming |
| `trim_mode` | str | "latent" | Trimming mode |
| `speaker_kv_scale` | float/None | None | Speaker KV scaling (force speaker) |
| `seed` | int/None | None | Random seed |

---

## Troubleshooting

### "No module named 'echo_tts_mlx'"

Install the package: `pip install .` or `pip install -e .` (editable mode).

### Out of memory

- Use `--quantize 8bit` to reduce memory by ~34%
- Reduce `--max-length` (320 ≈ 15 seconds instead of 640 ≈ 30 seconds)
- Close other memory-intensive applications

### Generated audio sounds wrong / different speaker

- Try `--force-speaker --speaker-scale 1.5`
- Use a cleaner reference clip
- Ensure reference audio is at 44.1 kHz sample rate

### Slow generation

- Use `--quantize 8bit` for 1.2-1.4x speedup
- Reduce steps: `--preset balanced` (16 steps) instead of `--preset quality` (32 steps)
- Reduce output length: `--max-length 320`
- Add `--verbose` to see which stage is the bottleneck

### "librosa not found" with --trim-mode f0

Install librosa: `pip install librosa`

### Weights not found

Ensure you've run the conversion step and the `--weights` path points to the converted directory (not the upstream checkpoints).
