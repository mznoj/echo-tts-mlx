# Architecture

Echo-TTS MLX is a diffusion-based text-to-speech system running natively on Apple Silicon.

## Pipeline Overview

```
Text → Tokenizer → Text Encoder → KV Cache ─┐
                                              ├→ DiT Diffusion (N steps) → PCA Decode → DAC Decode → Audio
Speaker Audio → DAC Encode → Speaker Encoder → KV Cache ─┘
```

### Components

| Component | Description | Module |
|---|---|---|
| **Tokenizer** | Byte-pair encoding for text input | `tokenizer.py` |
| **Text Encoder** | 14-block transformer, produces KV cache | Part of `model.py` |
| **Speaker Encoder** | 14-block transformer, encodes speaker latents | Part of `model.py` |
| **DiT (Diffusion Transformer)** | 24-block transformer, core generation model | `model.py` |
| **Sampler** | Euler flow-matching with dual CFG | `sampler.py` |
| **PCA** | Dimensionality reduction (latent ↔ codec space) | `pca.py` |
| **DAC (Autoencoder)** | Fish Speech S1-DAC, encode/decode 44.1kHz audio | `autoencoder.py` |
| **Pipeline** | Orchestrates all components end-to-end | `pipeline.py` |
| **Latent Encoder** | 14-block causal transformer, encodes prefix latents for blockwise generation | Part of `model.py` |

### Model Architecture

- **2.38B parameters** / 4.54 GB at float16
- **24 DiT blocks** (141 MB each, 3,384 MB total)
- **14 text encoder blocks** (560 MB total)
- **14 speaker encoder blocks** (560 MB total)
- **Conditioning module**: 34 MB
- **Sample rate**: 44,100 Hz

### Dual Classifier-Free Guidance

Each diffusion step runs 3 forward passes:
1. **Conditioned** (text + speaker)
2. **Unconditioned on text** (speaker only)
3. **Unconditioned on speaker** (text only)

Final velocity: `v = v_cond + scale_text × (v_cond - v_uncond_text) + scale_speaker × (v_cond - v_uncond_speaker)`

### Weight Format

Weights are converted from upstream PyTorch checkpoints to MLX-compatible safetensors:

```
weights/converted/
├── config.json              # Model configuration
├── dit_weights.safetensors  # DiT + encoders + conditioning
├── dac_weights.safetensors  # Fish Speech S1-DAC autoencoder
├── pca_state.safetensors    # PCA components and mean
└── weight_map.json          # Key mapping reference
```

### Quantized Weight Format

Quantized checkpoints add a config file:

```
weights/quantized-8bit/
├── config.json
├── dit_weights.safetensors  # Quantized DiT weights
├── dac_weights.safetensors  # Unchanged (always float32)
├── pca_state.safetensors    # Unchanged (always float32)
├── weight_map.json
└── quantize_config.json     # Quantization metadata
```

## Blockwise Generation

Blockwise generation extends the standard whole-sequence diffusion to incremental block-by-block generation:

```
Text → Tokenizer → Text Encoder → KV Cache ─┐
                                              │
Speaker Audio → DAC Encode → Speaker Encoder → KV Cache ─┤
                                              │
Previous Blocks → Latent Encoder → KV Cache ──┤
                                              ├→ DiT Diffusion (per block) → PCA Decode → DAC Decode → Audio
                                              └─ Repeat for each block
```

The latent encoder (14 causal transformer blocks, same architecture as the speaker encoder) processes previously generated clean latents to provide context for subsequent blocks. Per-layer `wk_latent`/`wv_latent` projections produce KV caches that participate in 4-way joint attention: `[self, latent, text, speaker]`.

Key properties:

- Block sizes must sum to ≤ `max_latent_length` latent frames (640 in current checkpoints)
- RoPE offset (`start_pos`) accumulates across blocks for positional continuity
- Latent KV uses dilated RoPE positions (`[0, 4, 8, ...]`) with position-based causal masking
- Supports audio continuations by prepending existing audio as a latent prefix
- Enables streaming: S1-DAC decoder is fully causal, so each block can be decoded independently

The blockwise weights (~788 MB at float16) are included in the upstream checkpoint but pruned by default during conversion. Use `--include-blockwise` during conversion to retain them.

## Performance Profile

On Mac mini M4 (16GB), typical generation at 32 steps:

| Phase | % of Wall Time |
|---|---|
| DiT Diffusion | 80–90% |
| DAC Decode | 8–15% |
| Encoding (text + speaker) | 2–5% |
| PCA | < 1% |

Diffusion dominates — optimization efforts should focus there.
