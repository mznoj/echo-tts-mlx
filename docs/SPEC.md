# Echo-TTS MLX Port — Specification & Implementation Plan

> **Source repo:** [jordandare/echo-tts](https://github.com/jordandare/echo-tts)
> **Model weights:** [jordand/echo-tts-base](https://huggingface.co/jordand/echo-tts-base) (DiT + PCA) · [jordand/fish-s1-dac-min](https://huggingface.co/jordand/fish-s1-dac-min) (autoencoder)
> **Blog:** [jordandarefsky.com/blog/2025/echo](https://jordandarefsky.com/blog/2025/echo/)

---

## 1. Overview

Echo-TTS is a diffusion-based multi-speaker text-to-speech model. It generates 44.1 kHz audio from text, optionally conditioned on a speaker reference clip. The upstream implementation is PyTorch + CUDA. This project ports the inference pipeline to Apple's **MLX** framework for native Apple Silicon execution.

### Scope

- **v1:** Inference-only port — generate speech from text + speaker reference on Apple Silicon. Includes blockwise/chunked generation, quantization (8-bit, mxfp4, mixed), quality presets, and audio continuations.
- **Future:** Fine-tuning support, audio streaming integration.
- **Non-goals:** Training, CUDA support, PyTorch runtime dependency at inference time.

### Key Constants

| Constant | Value | Notes |
|----------|-------|-------|
| Sample rate | **44,100 Hz** | All audio I/O at this rate |
| AE downsample factor | **2048** | 1 latent frame = 2048 audio samples ≈ 46.4 ms |
| Max latent length | **640** | 640 × 2048 / 44100 ≈ 29.7 seconds |
| Max text length | **768 bytes** | UTF-8 byte-level tokenization |
| Max speaker ref | **6400 latents** | ≈ 5 minutes of reference audio |
| Latent dim | **80** | After PCA projection |

### Performance Targets

| Metric | Target | Stretch | Chip | Precision |
|--------|--------|---------|------|-----------|
| Real-Time Factor (RTF) | < 2.0 | < 1.0 | M1 Pro 16GB | float16 |
| RTF (quantized) | < 2.5 | < 1.5 | M1 Pro 16GB | 8-bit |
| Peak memory (10s clip) | < 6GB | < 4GB | M1 Pro 16GB | float16 |
| Max generation length (v1) | 30s | — | Any | Any |

RTF = generation_time / audio_duration. Below 1.0 means faster than real-time. Targets are conservative given the 1.7B param model requires ~3.4GB weight streaming per forward pass × 40 steps × multiple CFG passes. Stretch targets assume MLX kernel optimizations and batched CFG.

### License

| Component | License |
|-----------|---------|
| Code (this port) | MIT |
| Upstream code (`autoencoder.py`, `_dac_core.py`) | Apache-2.0 |
| Model weights (`echo-tts-base`) | CC-BY-NC-SA-4.0 |
| Fish S1-DAC weights | CC-BY-NC-SA-4.0 |
| **Generated audio output** | **CC-BY-NC-SA-4.0** (inherited from S1-DAC) |

⚠️ All audio produced by this model is CC-BY-NC-SA-4.0 regardless of the code license. No commercial use without separate licensing.

---

## 2. Architecture

### 2.1 Pipeline Overview

```
Text ──→ [UTF-8 Tokenizer] ──→ text_ids + mask ─────────────────────┐
                                                                     │
Speaker Audio ──→ [Resample 44.1kHz] ──→ [DAC Encode] ──→           │
    [PCA Transform] ──→ speaker_latents + mask ──────────────────┐   │
                                                                 │   │
Random Noise ──→ [Euler Sampler + Dual CFG] ←── EchoDiT ←───────┴───┘
                        │                           ↑
                        ↓                      40 steps
                  final_latents
                        │
                  [Inverse PCA] ──→ [DAC Decode] ──→ [Trim Silence] ──→ WAV (44.1kHz)
```

### 2.2 Model Configuration

The `EchoDiT` model is instantiated with these exact hyperparameters (from upstream `inference.py`):

```python
EchoDiT(
    # Main DiT (diffusion transformer)
    latent_size=80,
    model_size=2048,
    num_layers=24,
    num_heads=16,
    intermediate_size=5888,
    norm_eps=1e-5,
    
    # Text encoder
    text_vocab_size=256,       # UTF-8 byte vocabulary
    text_model_size=1280,
    text_num_layers=14,
    text_num_heads=10,
    text_intermediate_size=3328,
    
    # Speaker encoder (operates on DAC latent patches)
    speaker_patch_size=4,
    speaker_model_size=1280,
    speaker_num_layers=14,
    speaker_num_heads=10,
    speaker_intermediate_size=3328,
    
    # Conditioning
    timestep_embed_size=512,
    adaln_rank=256,
)
```

These must be saved as a `config.json` alongside the converted weights to allow `from_pretrained()` loading.

**Parameter count:** ≈1.7B total (DiT + text encoder + speaker encoder).

### 2.3 Components to Port (`model.py`)

- **`LowRankAdaLN`**: Adaptive layer norm with low-rank projection (`rank=256`). Decomposes shift/scale/gate into down-projection → SiLU → up-projection with residual. Gate uses `tanh`. Implements its own RMSNorm internally (not using `nn.RMSNorm`).
- **`RMSNorm`**: Custom implementation supporting both 1D `(model_size,)` and 2D `(num_heads, head_dim)` weight shapes. Replace with `mlx.nn.RMSNorm` for 1D; implement manual version for 2D (per-head normalization).
- **`SelfAttention`**: Self-attention with **QK normalization** (per-head RMSNorm on Q and K before RoPE) and **sigmoid gating** (`output *= sigmoid(gate(x))`). Projects Q/K/V/O and a separate gate, all bias-free.
- **`JointAttention`**: The DiT's main attention — concatenates self-attention keys/values with pre-computed text and speaker KV caches. Features:
  - **Half-rotary RoPE:** Only the first half of attention heads get positional encoding; the second half is position-free. Implemented via `chunk(2, dim=-2)` then applying RoPE only to the first chunk.
  - **QK normalization:** Same per-head RMSNorm as SelfAttention.
  - **Sigmoid gating:** Same `output *= sigmoid(gate(x))` pattern.
  - **KV cache methods:** `get_kv_cache_text()`, `get_kv_cache_speaker()` project text/speaker states into K/V for each layer. These are computed **once** and reused across all diffusion steps (critical for performance — see Section 4).
- **`MLP`**: SwiGLU feed-forward: `w2(silu(w1(x)) * w3(x))`. All projections bias-free.
- **`EncoderTransformerBlock`**: Pre-norm (RMSNorm) + SelfAttention + MLP with residual connections. Used in text and speaker encoders.
- **`TransformerBlock`**: The DiT block. Uses `LowRankAdaLN` for adaptive conditioning (timestep-dependent). JointAttention + MLP, both gated by AdaLN outputs (`x += gate * attention(norm(x))`).
- **`TextEncoder`**: 14-layer transformer with `SelfAttention` (non-causal) over UTF-8 byte embeddings. Computes per-position RoPE frequencies from `head_dim=128`.
- **`SpeakerEncoder`**: 14-layer transformer with `SelfAttention` (**causal**) over patched DAC latents. Input projection: `Linear(80 * 4, 1280)` followed by hardcoded `÷6` divisor for activation stability. `head_dim=128`.
- **`EchoDiT`**: Main wrapper. Timestep conditioning via sinusoidal embedding → 3-layer MLP (`timestep_embed_size → model_size → model_size*3`), producing the shift/scale/gate vectors for all AdaLN layers.

#### RoPE Implementation

The upstream uses **complex-number RoPE**: `torch.view_as_complex()` / `torch.view_as_real()`. MLX does **not** support complex tensor views, so the implementation must use equivalent real-valued rotation:

```python
def apply_rotary_emb(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
```

Parameters: `theta=10000.0`, frequency precomputation matches standard LLaMA-style RoPE.

⚠️ Verify `mlx.nn.RoPE` output matches this formulation — if not, use custom implementation.

#### KV Caching Strategy (Critical Performance Optimization)

Text and speaker conditioning are **constant** across all diffusion steps. The upstream computes KV caches once and reuses them:

```python
# Computed ONCE before sampling loop:
kv_text = model.get_kv_cache_text(text_ids, text_mask)      # List of (K, V) per layer
kv_speaker = model.get_kv_cache_speaker(speaker_latent)      # List of (K, V) per layer

# Reused for ALL 40 steps in the sampling loop
```

Each `get_kv_cache_*` method:
1. Runs the encoder (TextEncoder or SpeakerEncoder) — forward pass through 14 transformer layers.
2. Applies final norm (`text_norm` / `speaker_norm`).
3. Projects through each DiT layer's `wk_text`/`wv_text` (or `wk_speaker`/`wv_speaker`) to produce per-layer K/V pairs.

**Without this caching**, the text encoder (14 layers) and speaker encoder (14 layers) would re-run at every diffusion step, making inference ~3× slower.

For batched CFG (see Section 4), the caches are pre-concatenated along the batch dimension:
```python
kv_text_full = concat_kv(kv_text, kv_text, kv_text)         # batch=3
kv_speaker_full = concat_kv(kv_speaker, kv_speaker, kv_speaker)
```

#### Blockwise Module Pruning

For standard (non-blockwise) inference, the following modules are unused and should be **pruned during weight conversion** to save memory:

- `latent_encoder.*`
- `latent_norm*`
- `*.wk_latent`
- `*.wv_latent`

This is the default in the upstream: `load_model_from_hf(delete_blockwise_modules=True)`.

### 2.4 Text Tokenization

The tokenizer is **raw UTF-8 byte encoding** — no external tokenizer library needed.

```python
def tokenize(text: str) -> list[int]:
    # 1. Normalize punctuation
    text = text.replace("…", "...")
    text = text.replace("\u2019", "'")   # smart right single quote → ASCII
    text = text.replace("\u201c", '"')   # smart left double quote → ASCII
    text = text.replace("\u201d", '"')   # smart right double quote → ASCII
    text = text.replace("\n", " ")
    text = text.replace(":", ",")
    text = text.replace(";", ",")
    text = text.replace("—", ", ")
    
    # 2. Add speaker tag if missing
    if (not text.startswith("[") and not text.startswith("(") 
            and "S1" not in text and "S2" not in text):
        text = "[S1] " + text
    
    # 3. Encode as UTF-8 bytes with BOS
    tokens = [0] + list(text.encode("utf-8"))  # BOS = 0
    return tokens
```

- **Vocabulary:** 256 (one per byte value)
- **Max length:** 768 bytes (truncate with warning if exceeded)
- **Speaker tag:** `[S1]` auto-prepended; `[S2]` also supported
- **Commas** act as pauses; exclamation points increase expressiveness but may reduce quality

### 2.5 Speaker Reference Pipeline

The speaker encoder does **NOT** use mel spectrograms. It processes **DAC autoencoder latents** of the reference audio:

```
Reference WAV ──→ [Resample to 44.1kHz]
                      │
                 [DAC Encode (chunked)]
                      │
                 [PCA Transform]
                      │
                 speaker_latents (B, T, 80)
                      │
                 [Patch: group by 4] ──→ (B, T/4, 320)
                      │
                 [Speaker Transformer (14 layers)]
                      │
                 speaker KV cache (used in DiT cross-attention)
```

**Details:**
1. **Load & resample** reference audio to 44,100 Hz via `soxr` (or `soundfile` + `soxr`).
2. **Normalize amplitude:** `audio = audio / max(abs(audio).max(), 1.0)` — prevents clipping by scaling to [-1, 1] range. Must replicate exactly.
3. **Chunk processing:** Reference audio is processed through DAC encode in chunks of `640 × 2048 = 1,310,720` samples (~30s each). Short final chunks are zero-padded to chunk size. Chunks are concatenated along the time dimension. Max total: 6400 latent frames (~5 minutes).
4. **PCA transform** applied to DAC latents (same transform as for diffusion latents).
5. **Truncate to actual length:** Only keep latent frames corresponding to actual audio (not padding), using `actual_length = audio_samples // AE_DOWNSAMPLE_FACTOR`.
6. **Divisibility trimming:** Trim to nearest multiple of `patch_size=4` (e.g., 101 frames → 100).
7. **Patching:** Speaker latents grouped by `patch_size=4`: reshape `(B, T, 80)` → `(B, T/4, 320)`.
8. **Input projection + divisor:** After `in_proj(patches)`, the result is divided by `6.0` (hardcoded activation stability trick from upstream training — must replicate exactly).
9. **Speaker transformer:** 14-layer **causal** self-attention encoder processes patches.
10. **Mask:** Boolean mask tracks actual vs. padding positions. Subsampled by `patch_size` for DiT attention.

**Multi-reference:** Multiple clips can be provided; they are concatenated before encoding (up to 6400 latent frames total).

#### Speaker KV Scaling ("Force Speaker")

When the model generates a different speaker than intended (common with out-of-distribution text), KV scaling forces speaker identity:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `speaker_kv_scale` | None (disabled) | Scale factor for speaker KV attention. 1.5 usually forces the speaker. |
| `speaker_kv_max_layers` | None | Limit scaling to first N DiT layers |
| `speaker_kv_min_t` | None | Only apply scaling above this timestep |

Exposed via CLI as `--force-speaker` (enables with scale=1.5) and `--speaker-scale <float>`.

### 2.6 PCA Transform

A **critical pipeline component** sitting between the DAC autoencoder and the DiT. Weights are in `pca_state.safetensors` from the `jordand/echo-tts-base` repo.

```python
@dataclass
class PCAState:
    pca_components: mx.array   # (80, 1024) — projects from DAC dim to latent dim
    pca_mean: mx.array         # (1024,)    — mean vector in DAC space (1D — broadcasts over batch/time)
    latent_scale: float        # scalar (stored as [1] tensor, extract with .item())
```

The DAC autoencoder's latent dimension is **1024**. PCA projects this down to `latent_size=80` for the DiT.

**Encoding** (audio → DiT input):
```python
z = dac_encode(audio)                              # (B, 1024, T)
z = z.transpose(1, 2)                              # (B, T, 1024)
z = (z - pca_mean) @ pca_components.T              # (B, T, 1024) @ (1024, 80) → (B, T, 80)
z = z * latent_scale                               # Normalize
```

**Decoding** (DiT output → audio):
```python
z = (z / latent_scale) @ pca_components + pca_mean  # (B, T, 80) @ (80, 1024) → (B, T, 1024)
z = z.transpose(1, 2)                               # (B, 1024, T)
audio = dac_decode(z)                                # Back to waveform
```

The PCA state must be loaded and converted alongside the model weights. It is small (~1.6 MB at float32 for the two matrices + scalar) and should remain at **float32 always**.

### 2.7 Autoencoder — Fish Speech S1-DAC

The autoencoder is a **custom variant** of Descript Audio Codec, trained from scratch by Fish Audio. It is architecturally similar to DAC but is **not** the standard `descript-audio-codec` package.

**Architecture parameters** (from `build_ae()`):

```python
DAC(
    encoder_dim=64,
    encoder_rates=[2, 4, 8, 8],           # Downsample: 2×4×8×8 = 512
    latent_dim=1024,                        # Internal dim before PCA
    decoder_dim=1536,
    decoder_rates=[8, 8, 4, 2],            # Upsample: 8×8×4×2 = 512
    quantizer=DownsampleResidualVectorQuantize(
        input_dim=1024,
        n_codebooks=9,
        codebook_size=1024,
        codebook_dim=8,
        downsample_factor=(2, 2),           # Additional 2×2 = 4× in quantizer
        semantic_codebook_size=4096,
    ),
    sample_rate=44100,
    causal=True,                            # All convolutions are causal
    encoder_transformer_layers=[0, 0, 0, 4],  # 4 transformer layers in last encoder block
    decoder_transformer_layers=[4, 0, 0, 0],  # 4 transformer layers in first decoder block
)
```

**Total downsample factor:** encoder (512) × quantizer (4) = **2048** ✓

**Key characteristics:**
- **Sample rate:** 44,100 Hz
- **Latent dim:** 1024 (before PCA projection to 80)
- **Fully causal:** All convolutions are causal (left-padded), enabling streaming decode in v2
- **Weights:** `jordand/fish-s1-dac-min` on HuggingFace (separate from DiT)
- **Default precision:** float32 (audio codecs are quality-sensitive)
- **License:** CC-BY-NC-SA-4.0

**Internal components (more complex than basic DAC):**

| Component | Layers | Notes |
|-----------|--------|-------|
| Encoder conv chain | 4 blocks (dim 64→128→256→512→1024) | Snake activations, causal Conv1d |
| Encoder transformers | 4 layers (in last block) | Full self-attention within encoder |
| Quantizer pre-transformer | 8 layers (`WindowLimitedTransformer`) | Pre-processes before VQ |
| Quantizer VQ | 1 semantic (4096) + 9 residual (1024 each) | Downsample 4× → quantize → upsample 4× |
| Quantizer post-transformer | 8 layers (`WindowLimitedTransformer`) | Post-processes after VQ |
| Decoder conv chain | 4 blocks (dim 1536→768→384→192→1) | Snake activations, causal ConvTranspose1d |
| Decoder transformers | 4 layers (in first block) | Full self-attention within decoder |

⚠️ The quantizer transformers (16 total layers) and encoder/decoder transformers (8 total layers) add significant complexity beyond a simple conv-only architecture. These must be ported correctly.

**Both encode and decode paths are needed:**
- **Encode path** (`encode_zq`): For speaker reference processing. Goes through encoder → quantizer (with VQ) → returns quantized latents `z_q` of shape `(B, 1024, T)`.
- **Decode path** (`decode_zq`): For final audio generation. Takes `z_q` → quantizer post-transformer → quantizer upsample → decoder → waveform.

**MLX porting challenges:**

| Challenge | Status (MLX 0.31+) | Approach |
|-----------|---------------------|----------|
| `Conv1d` | ✅ Native | Direct port |
| `ConvTranspose1d` | ✅ Native | Direct port |
| Snake activation | ❌ Not built-in | Manual: `x + (alpha + 1e-9).reciprocal() * sin(alpha * x)^2` |
| Weight normalization | ❌ Not built-in | **Fold at conversion time** (see Section 3) |
| `einops.rearrange` | ❌ Not available | Replace with `mx.reshape` / `mx.transpose` |
| `WindowLimitedTransformer` | ❌ Custom | Port from upstream (local attention with window) |
| Causal padding | Manual | Left-pad convolutions: `pad = (kernel_size - 1) * dilation` |

#### Feasibility Assessment

Before committing to a full port:

1. **Check `descript-mlx`** (PyPI v0.0.2) — an existing MLX DAC port. Fish S1-DAC is architecturally similar but trained from scratch, so layer names / shapes will differ. Useful as a **reference** for MLX porting patterns even if not directly compatible.
2. **Prototype Snake1d + ConvTranspose1d** in MLX. Both are straightforward with MLX 0.31+.
3. **Option C (hybrid fallback):** Run autoencoder in PyTorch CPU if MLX port proves complex. Acceptable latency: ~100ms for 10s decode on M1 Pro.

> **Go/no-go gate:** Must produce a working MLX decode of a single latent frame before proceeding.

---

## 3. Weight Conversion

Two separate checkpoints need conversion:

| Checkpoint | HuggingFace Repo | Contents |
|------------|-------------------|----------|
| DiT model | `jordand/echo-tts-base` | `pytorch_model.safetensors` (EchoDiT weights) |
| PCA state | `jordand/echo-tts-base` | `pca_state.safetensors` (PCA components, mean, scale) |
| Autoencoder | `jordand/fish-s1-dac-min` | `pytorch_model.safetensors` (Fish S1-DAC weights) |

### Conversion Script

Create `convert_weights.py` that handles all three files:

```bash
echo-tts-mlx convert \
  --dit-repo jordand/echo-tts-base \
  --dac-repo jordand/fish-s1-dac-min \
  --output ./weights/ \
  --dtype float16 \
  --prune-blockwise    # default: true
```

### Weight Layout Rules

| Layer Type | PyTorch Shape | MLX Shape | Transpose? |
|---|---|---|---|
| `nn.Linear` | `(out, in)` | `(out, in)` | **No** |
| `nn.Conv1d` | `(out_ch, in_ch, kernel)` | `(out_ch, in_ch, kernel)` | **No** |
| `nn.ConvTranspose1d` | `(in_ch, out_ch, kernel)` | Verify against `mlx.nn.ConvTranspose1d` | **Check** — verify empirically |

### Source Dtypes (Verified from Checkpoints)

| File | Stored Dtype | Notes |
|------|-------------|-------|
| DiT weights | **BF16** | NOT float16! Must cast during conversion |
| PCA state | F32 | Keep as-is |
| DAC weights | F32 | Keep as-is for audio quality |

⚠️ DiT weights are stored as **bfloat16**. Metal on M1 has limited bf16 support. Default conversion should cast to **float16**. Optionally support `--dtype bfloat16` for M2+ chips.

### Weight Normalization Folding (DAC Only)

The DAC uses the **new PyTorch parametrization API** (`torch.nn.utils.parametrizations.weight_norm`). Confirmed key pattern from checkpoint inspection:

```
{prefix}.conv.parametrizations.weight.original0  — gain (g): shape (out_ch, 1, 1)
{prefix}.conv.parametrizations.weight.original1  — weight (v): shape (out_ch, in_ch, kernel)
{prefix}.conv.bias                               — bias (not parametrized)
```

160 of 541 DAC keys use this pattern. The remaining keys (quantizer upsample/downsample convs, ConvNeXt blocks, transformer layers, codebook embeddings) use **regular weights without parametrizations**. The conversion script must handle both.

**Folding formula:**
```python
g = state[f"{prefix}.conv.parametrizations.weight.original0"]  # (out_ch, 1, 1)
v = state[f"{prefix}.conv.parametrizations.weight.original1"]  # (out_ch, in_ch, kernel)
# Per-output-channel L2 norm over dims (1, 2)
v_norm = torch.sqrt((v ** 2).sum(dim=(1, 2), keepdim=True) + 1e-12)
weight = g * (v / v_norm)
# Save as "{prefix}.conv.weight" in MLX state dict
```

**Also handle VQ in_proj/out_proj** — same pattern for `quantizer.{semantic_quantizer,quantizer}.quantizers.{i}.{in_proj,out_proj}.conv.parametrizations.weight.original{0,1}`.

### Skip Registered Buffers

These are recomputed at model init and should **not** be included in converted weights:
- `quantizer.pre_module.causal_mask`
- `quantizer.pre_module.freqs_cis`
- `quantizer.post_module.causal_mask`
- `quantizer.post_module.freqs_cis`

### Blockwise Pruning

By default, remove blockwise inference keys (unused for standard generation):

```python
PRUNE_PREFIXES = ["latent_encoder.", "latent_norm"]
PRUNE_CONTAINS = [".wk_latent", ".wv_latent"]
```

### Precision Strategy

| Component | Source dtype | Conversion dtype | Rationale |
|-----------|-------------|-----------------|-----------|
| DiT weights | **BF16** | **float16** (default) | Cast bf16→f16; bf16 optional for M2+ |
| PCA state | F32 | **float32 always** | Small, precision-critical |
| DAC autoencoder | F32 | **float32 default** | Audio quality sensitive; float16 optional with `--dac-dtype float16` |

### Validation

- Maintain `weight_map.json` documenting every PT key → MLX key mapping.
- Round-trip test: convert PT → MLX → PT, verify no corruption.
- Per-layer parity test: load converted weight, run single layer on fixed input, assert `atol=1e-5` (float32) or `atol=1e-2` (float16).

---

## 4. Inference Loop & Sampler

### Overview

The sampler (`sample_euler_cfg_independent_guidances`) implements:
1. Pre-compute text and speaker KV caches (once)
2. Initialize random noise
3. Euler ODE integration with dual CFG over `num_steps` steps
4. Return denoised latents

### Timestep Schedule

Linear schedule from 1 → 0, scaled by `INIT_SCALE = 0.999`:

```python
t_schedule = linspace(1.0, 0.0, num_steps + 1) * 0.999
# For 40 steps: [0.999, 0.974, 0.949, ..., 0.025, 0.0]
```

Each step transitions from `t_schedule[i]` to `t_schedule[i+1]`, with `dt = t_next - t` (negative, since we go from noise to signal).

### Initial Noise + Truncation

```python
x_t = mx.random.normal((1, sequence_length, 80))  # sequence_length default: 640
if truncation_factor is not None:
    x_t = x_t * truncation_factor  # default: 0.8 — reduces generation randomness
```

`truncation_factor` scales the initial noise to reduce variance. The upstream example uses `0.8`. Lower values → more predictable but potentially less diverse output. Should be exposed as CLI flag.

### KV Cache Pre-computation

Before the sampling loop, text and speaker KV caches are computed **once**:

```python
# Runs text encoder (14 layers) + projects to DiT KV space (24 layers)
kv_text_cond = model.get_kv_cache_text(text_ids, text_mask)

# Runs speaker encoder (14 layers) + projects to DiT KV space (24 layers)
kv_speaker_cond = model.get_kv_cache_speaker(speaker_latent)

# Optional: apply speaker KV scaling for "force speaker" mode
if speaker_kv_scale is not None:
    multiply_kv_cache(kv_speaker_cond, speaker_kv_scale, max_layers=speaker_kv_max_layers)
```

For batched CFG, caches are pre-concatenated along batch dimension (batch=3):
```python
kv_text_full = concat_kv(kv_text_cond, kv_text_cond, kv_text_cond)
kv_speaker_full = concat_kv(kv_speaker_cond, kv_speaker_cond, kv_speaker_cond)
```

**This is the single biggest performance optimization.** Without it, 28 extra transformer layers would run at every step.

### Dual Classifier-Free Guidance

Echo-TTS uses a **custom "independent guidances"** CFG formulation — NOT standard two-scale CFG.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cfg_scale_text` | 3.0 | Text-following strength |
| `cfg_scale_speaker` | 8.0 | Speaker similarity strength |
| `cfg_min_t` | 0.5 | CFG only active when `t >= cfg_min_t` |
| `cfg_max_t` | 1.0 | CFG only active when `t <= cfg_max_t` |

**CFG is time-dependent:** Only applied when `cfg_min_t <= t <= cfg_max_t`. At default settings, CFG is active for the first ~20 of 40 steps (t: 0.999→0.5), then disabled for the rest. When disabled, only one forward pass is needed.

**When CFG is active** — three conditions are batched in a single forward pass (batch=3):

```python
# Batched input: [fully_cond, text_uncond, speaker_uncond]
x_batch = concat([x_t, x_t, x_t], dim=0)  # (3, seq_len, 80)

# Masks control what each batch element "sees":
text_mask_batch    = concat([text_mask,   zeros_like,   text_mask  ], dim=0)
speaker_mask_batch = concat([speaker_mask, speaker_mask, zeros_like ], dim=0)

# Single forward pass through DiT (batch=3)
v_cond, v_uncond_text, v_uncond_speaker = model(x_batch, t, ...).chunk(3)
```

**The CFG formula (non-standard!):**
```python
v_pred = v_cond + cfg_text * (v_cond - v_uncond_text) + cfg_speaker * (v_cond - v_uncond_speaker)
```

This adds text guidance and speaker guidance **independently relative to the fully conditioned prediction** (`v_cond`), not relative to a single unconditional baseline. The three conditions are:
- `v_cond`: Both text and speaker conditioning active
- `v_uncond_text`: Speaker active, text mask zeroed (measures text contribution)
- `v_uncond_speaker`: Text active, speaker mask zeroed (measures speaker contribution)

**When CFG is disabled** (second half of sampling) — single forward pass:
```python
v_pred = model(x_t, t, text_mask, speaker_mask, kv_text_cond, kv_speaker_cond)
```

### Speaker KV Scale Reversal

When `speaker_kv_scale` is used, it is **reversed** when the timestep drops below `speaker_kv_min_t`:

```python
if speaker_kv_scale and t_next < speaker_kv_min_t and t >= speaker_kv_min_t:
    multiply_kv_cache(kv_speaker_cond, 1.0 / speaker_kv_scale, speaker_kv_max_layers)
    # Rebuild batched cache for remaining steps
    kv_speaker_full = concat_kv(kv_speaker_cond, kv_speaker_cond, kv_speaker_cond)
```

The scaling only applies during the noisiest part of generation where speaker identity is established.

### Euler Integration Step

```python
x_t = x_t + v_pred * (t_next - t)  # dt is negative (t decreases)
```

### Optional: Temporal Score Rescaling

From [arXiv:2510.01184](https://arxiv.org/pdf/2510.01184). Controlled by `rescale_k` and `rescale_sigma` (both default None = disabled). Adjusts the velocity prediction based on SNR to reduce artifacts at high noise levels. Not critical for v1.

### Latent Trimming

Generated latents include trailing silence/padding. A heuristic trims the output:

```python
def find_flattening_point(latents, window_size=20, std_threshold=0.05):
    """Input: (length, 80). Sliding window: if std < 0.05 and |mean| < 0.1
    for `window_size` consecutive frames, that's the end of meaningful audio."""
    # Pad with zeros to catch flattening at the very end
    padded = concat([latents, zeros(window_size, 80)])
    for i in range(len(padded) - window_size):
        window = padded[i:i + window_size]
        if window.std() < std_threshold and abs(window.mean()) < 0.1:
            return i
    return len(latents)

# Trim audio waveform: audio[..., :flattening_point * 2048]
```

Without this, output audio has noticeable trailing silence.

### `mx.eval()` Placement

**Critical:** Call `mx.eval()` after each diffusion step to prevent computation graph explosion:

```python
for i in range(num_steps):
    t, t_next = t_schedule[i], t_schedule[i + 1]
    
    if cfg_min_t <= t <= cfg_max_t:
        # Batched CFG: 3× forward pass
        v_pred = batched_cfg_step(model, x_t, t, ...)
    else:
        # Single forward pass
        v_pred = model(x_t, t, ...)
    
    x_t = x_t + v_pred * (t_next - t)
    mx.eval(x_t)  # FORCE EVALUATION — never skip
```

### Performance Note

With 40 steps and CFG active for the first ~20:
- Steps 1–20: 3× DiT forward pass (batch=3) = heavy
- Steps 21–40: 1× DiT forward pass = light
- Total DiT forward passes: ~80 (not 120), thanks to time-dependent CFG

---

## 5. Implementation Plan

### DAC Autoencoder Feasibility

> **Go/no-go gate:** Produce a working MLX single-frame decode before proceeding.

The DAC is more complex than initially assumed (24 transformer layers across quantizer + encoder + decoder, not just convolutions). The spike must assess all of these:

1. **Survey existing work:** Check `descript-mlx` (PyPI v0.0.2) as reference. Inspect actual checkpoint keys from `jordand/fish-s1-dac-min` to understand weight naming and parametrization format.
2. **Implement primitives:**
   - `Snake1d` activation (simple math)
   - Causal `Conv1d` / `ConvTranspose1d` wrappers (left-padding)
   - Weight normalization folding script (inspect actual key names: `parametrizations.weight.original0/1` vs `weight_v/weight_g`)
3. **Prototype decode chain:** Build one `DecoderBlock` (Snake + ConvTranspose1d + residuals) and verify against PyTorch.
4. **Prototype quantizer transformer:** Build one `WindowLimitedTransformer` layer (the quantizer has 16 of these). This is the riskiest component — understand window size and attention pattern.
5. **End-to-end decode test:** Load full weights (with WN folding), decode one latent frame, compare against PyTorch CPU: `np.allclose(atol=1e-3)`.
6. **If prohibitively complex:** Document hybrid fallback (PyTorch CPU for DAC) with latency measurements. Acceptable: <200ms for 10s decode.

**Deliverable:** Working single-frame decode + feasibility report + complexity assessment of the 24 transformer layers.

### Weight Conversion

1. Map full parameter trees for all three checkpoints (DiT, PCA, DAC). Produce `weight_map.json`.
2. Implement `convert_weights.py`:
   - Downloads from HuggingFace via `huggingface_hub`.
   - Renames keys per mapping.
   - Folds weight normalization in DAC layers.
   - Prunes blockwise modules.
   - Handles `ConvTranspose1d` layout verification.
   - Converts PCA state (always float32).
   - Outputs: `dit_weights.safetensors`, `dac_weights.safetensors`, `pca_state.safetensors`, `config.json`.
3. Parity tests: single-layer forward pass comparison between frameworks.

### Core DiT Model

1. Implement `model.py` in MLX with all components from Section 2.3.
2. Define `ModelConfig` dataclass matching Section 2.2 hyperparameters. Save as `config.json`.
3. Implement RoPE via `mlx.nn.RoPE` (verify upstream variant compatibility).
4. Validate individual blocks against PyTorch using **predefined input tensors** (not RNG seeds):
   - `TextEncoder`: fixed byte sequence → compare hidden states.
   - `SpeakerEncoder`: fixed latent patches → compare KV cache output.
   - `TransformerBlock`: fixed input → compare output.
   - `EchoDiT` full forward: single step → compare noise prediction.

### Autoencoder Port

Based on the DAC feasibility assessment. **Both encode and decode paths are needed** (encode for speaker refs, decode for audio output).

1. **Full port:** Implement `autoencoder.py` in MLX:
   - `Snake1d` activation
   - Causal `Conv1d` / `ConvTranspose1d` wrappers (left-padding for causality)
   - Regular conv layers (weights pre-folded from WN at conversion time)
   - Encoder: 4 conv blocks + 4 transformer layers (last block)
   - Decoder: 4 transformer layers (first block) + 4 conv blocks
   - `DownsampleResidualVectorQuantize`:
     - `WindowLimitedTransformer` (8 pre + 8 post layers)
     - Downsample/upsample via conv (factor 4)
     - Semantic codebook (1 × 4096) + residual codebooks (9 × 1024)
   - Replace `einops.rearrange` with manual `mx.reshape` / `mx.transpose`
   - `encode_zq()`: full encoder → quantizer → return z_q
   - `decode_zq()`: quantizer post-transformer → upsample → decoder → waveform
   
2. **Library integration:** If `descript-mlx` patterns can be adapted, use as base.
3. **Hybrid fallback:** PyTorch CPU wrapper with clear API boundary. Acceptable latency: <200ms per 10s clip.
4. Validate:
   - Encode: reference audio → z_q, compare against PyTorch (`atol=1e-3`)
   - Decode: z_q → waveform, compare against PyTorch (`atol=1e-3`)
   - Round-trip: audio → encode → decode → audio, compare quality

### PCA, Sampler & Pipeline

1. Implement PCA transform (matrix multiply + bias):
   - `pca_encode(z, pca_state)` — (B, 1024, T) → (B, T, 80) for speaker reference processing
   - `pca_decode(z, pca_state)` — (B, T, 80) → (B, 1024, T) for final audio generation
2. Implement UTF-8 tokenizer with exact normalization rules (Section 2.4).
3. Implement speaker reference pipeline (Section 2.5): load → resample → normalize → DAC encode (chunked) → PCA → trim → patch.
4. Implement KV cache pre-computation:
   - `model.get_kv_cache_text()` — text encoder → per-layer K/V projection
   - `model.get_kv_cache_speaker()` — speaker encoder → per-layer K/V projection
   - Batch concatenation for CFG (3×)
5. Implement Euler sampler with dual CFG (Section 4):
   - Timestep schedule (`linspace * 0.999`)
   - Truncation factor on initial noise
   - Time-dependent CFG (active only when `cfg_min_t <= t <= cfg_max_t`)
   - Batched CFG forward pass (batch=3 when active, batch=1 when not)
   - Speaker KV scale reversal at `speaker_kv_min_t`
6. Implement latent trimming heuristic.
7. Build end-to-end `generate()` pipeline:
   - Audio resampling via `soxr`
   - Audio output via `soundfile` (`sf.write` at 44100 Hz)
   - Progress bar (tqdm or similar) for diffusion steps
8. Validate: same text + speaker + predefined noise → compare output against PyTorch.

### CLI, Packaging & Polish

1. Implement CLI with progress bar (Section 10).
2. Set up `pyproject.toml` with entry points (Section 9).
3. Write README: installation, quickstart, benchmark table, license notice.
4. Run benchmark suite.
5. Publish to PyPI.

### Quantization

Implement selective weight quantization for the DiT and encoder modules to reduce memory footprint by 50–75% while preserving audio quality. DAC autoencoder and PCA state remain at full precision.

> **Prerequisite:** CLI, Packaging & Polish complete. Benchmarks established as baseline for quality regression testing.

#### Architecture & Parameter Budget

The model totals **2.38B parameters / 4.54 GB** at float16:

| Component | Parameters | Size (f16) | Quantizable? |
|---|---|---|---|
| DiT blocks (24×) | 1,774,583,808 | 3,384 MB | ✅ Yes |
| Text encoder (14 blocks) | 293,672,960 | 560 MB | ✅ Yes (8-bit only) |
| Speaker encoder (14 blocks) | 293,672,960 | 560 MB | ✅ Yes (8-bit only) |
| Conditioning MLP (`cond_module`) | 17,825,792 | 34 MB | ✅ Yes |
| In/out projections | 330,808 | 0.6 MB | ❌ No (too small) |
| Norms + embeddings | 332,288 | 0.6 MB | ❌ No (precision-sensitive) |
| DAC autoencoder | (separate) | ~500 MB | ❌ No (float32 only) |
| PCA state | (separate) | ~few MB | ❌ No (float32 only) |

#### Implementation Steps

1. **Refactor `MlxEchoDiT` to use `mlx.nn.Module`**

   The current implementation stores weights in a flat `state` dict and performs manual matmuls. `mlx.nn.quantize()` requires an `mlx.nn.Module` tree with `nn.Linear` leaf modules. The refactor should be **minimally invasive:**

   - Convert all weight-holding projections (`wq`, `wk`, `wv`, `wo`, `w1`, `w2`, `w3`, `gate`, `wk_text`, `wv_text`, `wk_speaker`, `wv_speaker`, AdaLN `*_up`/`*_down`, `cond_module` layers) to `nn.Linear` modules.
   - Organize into an `nn.Module` tree that mirrors the existing key hierarchy: `self.blocks[i].attention.wq`, `self.blocks[i].mlp.w1`, `self.text_encoder.blocks[j].attention.wq`, etc.
   - Norms (`k_norm`, `q_norm`, `attention_norm`, `mlp_norm`, `out_norm`, `text_norm`, `speaker_norm`) stay as raw weight tensors or `nn.RMSNorm` — they must NOT be `nn.Linear` (they aren't linear projections).
   - `text_embedding` becomes `nn.Embedding` (quantizable but excluded by predicate).
   - The forward logic stays unchanged — just replace `self.t("key") @ x` with `self.module(x)`.
   - Do NOT do a full architectural rewrite. The goal is the minimal diff that enables `nn.quantize()`.

   **Parity gate:** The float16 unquantized code path must produce bit-identical output before and after refactoring. Verify with the existing parity tests (all 137 must pass).

2. **Implement selective quantization in `EchoTTS.from_pretrained()`**

   Use `mlx.nn.quantize()` with `class_predicate` for selective quantization:

   ```python
   # Modules to EXCLUDE from quantization (small, precision-sensitive, or non-linear)
   _QUANTIZE_SKIP = {
       "in_proj", "out_proj",                    # Small I/O projections
       "out_norm", "text_norm", "speaker_norm",  # Normalization layers
       "text_embedding",                         # Embedding table
       "k_norm", "q_norm",                       # Per-head RMSNorm
       "attention_norm", "mlp_norm",             # Encoder block norms
   }

   # AdaLN down-projections are low-rank bottlenecks — quantizing these
   # disproportionately affects conditioning quality. Skip them.
   _QUANTIZE_SKIP_SUFFIXES = {"_down"}

   def _quantize_predicate(path: str, module: nn.Module) -> bool:
       parts = path.split(".")
       if any(s in parts for s in _QUANTIZE_SKIP):
           return False
       if any(parts[-1].endswith(suffix) for suffix in _QUANTIZE_SKIP_SUFFIXES):
           return False
       return isinstance(module, nn.Linear)

   if quantize_mode == "8bit":
       nn.quantize(model, group_size=64, bits=8, class_predicate=_quantize_predicate)
   elif quantize_mode == "4bit":
       # DiT blocks get 4-bit; encoders get 8-bit (smaller models, less tolerance)
       def _dit_only(path, m):
           if not _quantize_predicate(path, m):
               return False
           return path.startswith("blocks.") or path.startswith("cond_module.")
       def _encoder_only(path, m):
           if not _quantize_predicate(path, m):
               return False
           return path.startswith("text_encoder.") or path.startswith("speaker_encoder.")
       nn.quantize(model, group_size=64, bits=4, class_predicate=_dit_only)
       nn.quantize(model, group_size=64, bits=8, class_predicate=_encoder_only)
   ```

   **Key constraints:**
   - Quantization is applied **after** weight loading, **before** inference.
   - `group_size=64` is the MLX default and works well for transformer weights. Test `32` and `128` as alternatives during validation.
   - Do NOT quantize: norms, embeddings, in/out projections, or AdaLN down-projections (low-rank bottlenecks — quantizing these disproportionately affects conditioning).
   - Do NOT quantize any DAC or PCA weights (they live in separate `nn.Module` trees that are never passed to `nn.quantize`).

3. **Save/load quantized weights**

   Add `--save-quantized` CLI flag that quantizes and saves the resulting weights so future loads skip the quantization step:

   ```bash
   echo-tts-mlx convert --quantize 8bit --save-quantized ./weights/quantized-8bit/
   ```

   The saved directory should include a `quantize_config.json` recording `bits`, `group_size`, and which modules were quantized. `from_pretrained()` auto-detects this config and loads quantized weights directly.

4. **Wire up the existing CLI flag**

   The `--quantize` flag already exists in `cli.py` but raises `NotImplementedError`. Replace with the actual quantization call. No CLI changes needed.

5. **Update benchmarks**

   Add quantized variants to the Tier 1 and Tier 3 benchmark suites:
   - `bench_model_load_8bit`, `bench_model_load_4bit` (cold-start with quantization overhead)
   - `bench_dit_forward_single_8bit`, `bench_dit_forward_cfg_8bit` (per-step latency)
   - Tier 3 standard cases at 8-bit and 4-bit
   - Memory comparison: peak and active memory at each precision level

#### Quality Validation Protocol

Quality validation gates must pass before quantization is considered complete.

**Gate 1: Automated metrics**

For each quantization mode (`8bit`, `4bit`), generate audio for 5 standardized prompts (the Tier 3 cases plus one additional long-form case) and compare against float16 baseline:

| Metric | 8-bit Threshold | 4-bit Threshold (experimental) |
|---|---|---|
| Mel-Cepstral Distortion (MCD) | < 7.0 dB | < 90.0 dB (informational only) |
| Peak amplitude ratio | 0.9–1.1 | 0.8–1.2 |
| Duration ratio | 0.95–1.05 | 0.90–1.10 |
| Speaker similarity (Resemblyzer) | > 0.85 | > 0.75 |

> **Measured results (Mac mini M4):** 8-bit MCD = 2.9–6.5 dB, speaker similarity = 0.890.
> 4-bit MCD = 21–85 dB — DiT blocks are too sensitive for 4-bit quantization at group_size=64.
> 4-bit is marked **experimental** and not recommended for production use.

MCD computation:
```python
import librosa
def compute_mcd(ref_audio, test_audio, sr=44100, n_mfcc=13):
    ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=n_mfcc)
    test_mfcc = librosa.feature.mfcc(y=test_audio, sr=sr, n_mfcc=n_mfcc)
    min_len = min(ref_mfcc.shape[1], test_mfcc.shape[1])
    diff = ref_mfcc[:, :min_len] - test_mfcc[:, :min_len]
    return float(np.mean(np.sqrt(np.sum(diff**2, axis=0))))
```

**Gate 2: Determinism**

Quantized inference must be deterministic (same seed → same output). Verify with `np.allclose(run1, run2, atol=1e-4, rtol=1e-3)`. Note: tolerances are relaxed vs. float16 (which uses `atol=1e-5`).

**Gate 3: Speaker cloning fidelity**

Generate cloned speech at each precision using the same reference audio. Compute speaker similarity using [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) (`pip install resemblyzer`), which provides a pretrained speaker encoder that produces 256-dim speaker embeddings:

```python
from resemblyzer import VoiceEncoder, preprocess_wav
encoder = VoiceEncoder()
embed_ref = encoder.embed_utterance(preprocess_wav(ref_audio, source_sr=44100))
embed_quant = encoder.embed_utterance(preprocess_wav(quant_audio, source_sr=44100))
similarity = float(np.dot(embed_ref, embed_quant))
```

Threshold: > 0.85 for 8-bit, > 0.75 for 4-bit. If Resemblyzer is unavailable, fall back to spectral envelope correlation (MFCC cosine similarity across frames) as a proxy.

#### Expected Results

| Mode | DiT Size | Encoder Size | Total Model | Memory Savings | Expected RTF Change |
|---|---|---|---|---|---|
| float16 (baseline) | 3,384 MB | 1,120 MB | 4,541 MB | — | 1.0× |
| 8-bit | 1,692 MB | 560 MB | 2,289 MB | **50%** | ~1.0–1.1× (neutral or slight speedup) |
| 4-bit DiT + 8-bit enc | 846 MB | 560 MB | 1,443 MB | **68%** | ~1.1–1.3× (possible speedup from reduced memory bandwidth) |

#### Test Plan

| Test | What it verifies |
|---|---|
| `test_quantize_mode_none_unchanged` | Unquantized path produces identical output before/after refactor |
| `test_quantize_8bit_loads_and_runs` | 8-bit model loads, generates non-silent audio, passes duration gate |
| `test_quantize_4bit_loads_and_runs` | Same for 4-bit |
| `test_quantize_determinism_8bit` | Two runs with same seed produce `allclose` output |
| `test_quantize_determinism_4bit` | Same for 4-bit |
| `test_quantize_mcd_8bit` | MCD < 7.0 dB vs float16 baseline |
| `test_quantize_mcd_4bit` | MCD informational (4-bit is experimental) |
| `test_quantize_save_load_roundtrip` | Save quantized weights, reload, verify identical output |
| `test_quantize_memory_reduction` | Active memory with 8-bit < 60% of float16 |
| `test_quantize_excludes_dac_pca` | DAC/PCA weights remain float32 after quantization |

#### Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `nn.Module` refactor breaks parity | Medium | Run full parity test suite after refactor, before quantization |
| 4-bit quality too low for TTS | Medium | Fall back to 8-bit as minimum; 4-bit is stretch goal |
| `mlx.nn.quantize()` doesn't support custom state dict layout | Low | Refactor to `nn.Linear` modules; use `class_predicate` for selective control |
| Group size sensitivity | Low | Test 32/64/128; default to 64 |
| Quantized speaker cloning degrades more than unconditional | Medium | Keep encoders at 8-bit even in 4-bit mode |
| AdaLN down-projections degrade conditioning at low precision | Medium | Exclude `*_down` weights from quantization via predicate |
| Resemblyzer unavailable or incompatible | Low | Fall back to MFCC cosine similarity proxy for Gate 3 |

---

## 6. Quantization Strategy

### Per-Component Rules

| Component | Quantizable? | Precision | Rationale |
|-----------|-------------|-----------|-----------|
| DiT attention (Q/K/V/O) | ✅ | 4-bit or 8-bit | Transformer weights compress well |
| DiT MLPs | ✅ | 4-bit or 8-bit | Same |
| Text encoder | ✅ | 8-bit | Smaller model, less tolerance |
| Speaker encoder | ✅ | 8-bit | Same |
| DAC autoencoder | ❌ | **float32** | Audio codecs extremely quality-sensitive |
| PCA state | ❌ | **float32** | Small, precision-critical |
| Embeddings | ❌ | **float16** | Small, quantization hurts quality |

### Memory Budget

| Precision | DiT Size | DAC Size | PCA | Total |
|-----------|----------|----------|-----|-------|
| float16 | ~3.4 GB | ~varies | ~few MB | ~4 GB |
| 8-bit DiT + f32 DAC | ~1.7 GB | ~varies | ~few MB | ~2.5 GB |
| 4-bit DiT + f32 DAC | ~0.85 GB | ~varies | ~few MB | ~1.5 GB |

### Validation

- Same prompt at float16 vs quantized: compute **Mel-Cepstral Distortion (MCD)**.
- **ABX listening test** on 5 diverse prompts.
- Publish results in README.

### CLI Flag

`--quantize 4bit|8bit|none` (default: `none`). Applied via `mlx.nn.quantize()` to DiT + text/speaker encoders only.

---

## 7. Test Plan

### Test Fixture Generation

A prerequisite script (`scripts/generate_fixtures.py`, requires PyTorch) runs the upstream model and saves intermediate tensors as `.npy` files:

- `noise_seed42.npy`: Initial noise tensor (1, 640, 80)
- `text_input.npy`: Token IDs for a reference prompt
- `text_mask.npy`: Corresponding mask
- `speaker_latent.npy`: Pre-computed speaker latents from a reference clip
- `speaker_mask.npy`: Corresponding mask
- `kv_text_layer0.npy`: Text KV cache for first DiT layer
- `kv_speaker_layer0.npy`: Speaker KV cache for first DiT layer
- `dit_step0_output.npy`: DiT output after one forward step
- `final_latent.npy`: Final denoised latent (after all 40 steps)
- `dac_decode_output.npy`: Waveform from DAC decode of final latent

These fixtures enable parity testing without running PyTorch at test time.

### Unit Tests

Component parity — pass identical predefined tensors to both PyTorch (CPU) and MLX, assert `np.allclose(atol=1e-3)`:

- `test_attention.py`: SelfAttention, JointAttention, RoPE
- `test_mlp.py`: MLP, LowRankAdaLN
- `test_dit_block.py`: EncoderTransformerBlock, TransformerBlock
- `test_speaker_encoder.py`: Fixed latent patches → SpeakerEncoder KV output
- `test_text_encoder.py`: Fixed byte sequence → TextEncoder output
- `test_autoencoder.py`: Single-frame DAC encode + decode
- `test_pca.py`: PCA encode/decode round-trip; compare against PyTorch
- `test_tokenizer.py`: Normalization rules, edge cases (Unicode, long text, missing `[S1]`)
- `test_snake.py`: Snake activation numerics

### Integration Tests

- `test_weight_conversion`: Convert PT → MLX → PT, verify no corruption. Check all keys via `weight_map.json`. Verify WN folding produces correct combined weights.
- `test_kv_cache`: Compute text and speaker KV caches from fixtures, compare against saved reference. Verify batch concatenation for CFG produces expected shapes.
- `test_forward_pass`: One full DiT step with pre-computed KV caches, compare noise predictions against `dit_step0_output.npy`.
- `test_quantization_roundtrip`: Quantize, reload, run, compare quality metrics.
- `test_speaker_pipeline`: Reference audio → resample → normalize → DAC encode → PCA → trim → patch. Compare against PyTorch at each stage.

### End-to-End Tests

- Generate 5-second clip with predefined noise in both frameworks.
- **MCD target:** < 5.0 dB (allows FP divergence over 40 diffusion steps).
- Listening test: 5 diverse prompts (short, long, different speakers, forced speaker).
- Verify latent trimming produces same-length output as upstream.

### Performance Benchmarks

| Metric | How | Target | Stretch |
|--------|-----|--------|---------|
| RTF | `gen_time / audio_duration` | < 2.0 | < 1.0 |
| Peak VRAM | `mx.metal.get_peak_memory()` | < 6GB (10s) | < 4GB |
| First-step latency | Time to first `mx.eval()` | < 3s | < 2s |
| Quantized RTF | 8-bit model | < 2.5 | < 1.5 |

**Chips:** M1 8GB (baseline), M1 Pro 16GB (primary), M2/M3/M4 (stretch).

Publish benchmark table in README.

---

## 8. Rules, Guidelines & Gotchas

### Lazy Evaluation

MLX is lazy. Call `mx.eval(x_t)` inside the sampling loop at **every** diffusion step. Without this, the graph grows per-step → OOM or freeze.

### Random Number Generation

PyTorch and MLX RNGs differ for the same seed. For cross-framework tests:
- Feed **predefined noise tensors** (saved as numpy `.npy`) to both.
- Within MLX: `mx.random.seed(seed)` for reproducibility. Document MLX version.

### Memory Management

- Drop intermediate tensors immediately after use.
- Clear KV caches between diffusion steps if applicable.
- `mx.metal.clear_cache()` between major pipeline stages.
- **Memory guard:** Estimate memory before loading. Warn if > 80% of unified memory. Suggest `--quantize 8bit`.
- **Auto-fallback:** If float16 loading OOMs, retry with 8-bit and log warning.

### Audio I/O

- **Input resampling:** Use `soxr` (lightweight, no PyTorch dependency). `pip install soxr`.
- **Output writing:** `soundfile.write("out.wav", audio, 44100)`. No `torchaudio`.
- **No PyTorch at runtime.** Conversion script may import torch; inference must not.

### Long-Form Generation (Deferred to v2)

- **v1:** Hard cap at **640 latents** (≈30s). Warn + truncate if text would produce longer audio.
- **v2:** Blockwise sampling (upstream `inference_blockwise.py`) with overlap stitching. S1-DAC decoder is causal, enabling streaming.
- Approximate OOM thresholds (measure during benchmarking):
  - ~15s on 8GB M1 (float16)
  - ~45s on 16GB M1 Pro (float16)

### Error Handling

| Condition | Behavior |
|-----------|----------|
| No speaker reference (None) | Generate without speaker conditioning (unconditional voice) |
| Reference < 1s | Warn: "Short reference may produce inconsistent speaker. Recommend 3-10s." |
| Reference > 5 min | Truncate to 6400 latent frames with info message |
| Text > 768 bytes | Truncate with warning |
| Unsupported audio format | Error with format list + ffmpeg suggestion |
| Model not found locally | Error pointing to `echo-tts-mlx convert` |
| OOM on load | Suggest `--quantize 8bit` or `--dac-dtype float16` |

---

## 9. Packaging & Dependencies

### Requirements

- **Python:** >= 3.10
- **MLX:** >= 0.31.0 (requires `ConvTranspose1d` support)
- **Core:** `mlx`, `soundfile`, `numpy`, `soxr`, `huggingface_hub`, `safetensors`
- **Optional (conversion only):** `torch` — required by `convert_weights.py`, **NOT** at inference time
- **No PyTorch at runtime.**

### Project Structure

```
echo-tts-mlx/
├── pyproject.toml
├── README.md
├── LICENSE                    # MIT (code)
├── src/
│   └── echo_tts_mlx/
│       ├── __init__.py
│       ├── config.py          # ModelConfig dataclass
│       ├── model.py           # MLX EchoDiT
│       ├── autoencoder.py     # MLX Fish S1-DAC (encode + decode)
│       ├── pca.py             # PCA transform + state loading
│       ├── sampler.py         # Euler solver + dual CFG
│       ├── tokenizer.py       # UTF-8 byte tokenizer + normalization
│       ├── speaker.py         # Speaker ref pipeline (load → encode → patch)
│       ├── pipeline.py        # End-to-end generate()
│       ├── cli.py             # CLI entry point
│       └── utils.py           # Audio I/O, memory estimation, trimming
├── scripts/
│   ├── convert_weights.py     # PT → MLX conversion (requires torch)
│   └── generate_fixtures.py   # Generate test fixtures from PyTorch model
├── tests/
│   ├── test_attention.py
│   ├── test_mlp.py
│   ├── test_dit_block.py
│   ├── test_autoencoder.py
│   ├── test_speaker_encoder.py
│   ├── test_text_encoder.py
│   ├── test_pca.py
│   ├── test_tokenizer.py
│   ├── test_snake.py
│   ├── test_weight_conversion.py
│   ├── test_kv_cache.py
│   ├── test_forward_pass.py
│   ├── test_speaker_pipeline.py
│   └── test_e2e.py
├── fixtures/                   # Predefined test tensors (.npy)
│   ├── noise_seed42.npy
│   ├── text_input.npy
│   └── speaker_latent.npy
├── config.json                # Default model config
└── weight_map.json            # PT → MLX key mapping
```

### Installation

```bash
pip install echo-tts-mlx           # from PyPI (inference only)
pip install .                       # from source
pip install ".[dev]"                # with test deps (includes torch for parity tests)
pip install ".[convert]"            # with torch for weight conversion
```

---

## 10. CLI Interface

### Commands

```bash
# Generate speech
echo-tts-mlx generate \
  --text "Hello, this is a test of the Echo TTS model." \
  --speaker ref_audio.wav \
  --output out.wav \
  --weights ./weights/ \
  --steps 40 \
  --cfg-text 3.0 \
  --cfg-speaker 8.0 \
  --seed 42 \
  --truncation-factor 0.8 \
  --quantize 8bit \
  --force-speaker \
  --speaker-scale 1.5 \
  --max-length 640 \
  --verbose

# Convert weights from HuggingFace
echo-tts-mlx convert \
  --dit-repo jordand/echo-tts-base \
  --dac-repo jordand/fish-s1-dac-min \
  --output ./weights/ \
  --dtype float16 \
  --dac-dtype float32 \
  --prune-blockwise

# Print model info, memory estimate, chip details
echo-tts-mlx info [--weights ./weights/]
```

### Python API

```python
from echo_tts_mlx import EchoTTS
import mlx.core as mx

model = EchoTTS.from_pretrained("./weights/", dtype=mx.float16)

audio = model.generate(
    text="Hello, this is a test.",
    speaker="ref.wav",              # or None for unconditional
    steps=40,
    cfg_text=3.0,
    cfg_speaker=8.0,
    seed=42,
    truncation_factor=0.8,
    force_speaker=False,
    speaker_scale=1.5,
)

model.save_audio(audio, "out.wav")  # Saves at 44100 Hz
```

### CLI Defaults

| Flag | Default | Range / Notes |
|------|---------|---------------|
| `--steps` | 40 | 10–100 |
| `--cfg-text` | 3.0 | 1.0–10.0 |
| `--cfg-speaker` | 8.0 | 1.0–15.0 |
| `--seed` | random | Any int |
| `--truncation-factor` | 0.8 | 0.0–1.0; scales initial noise. Lower = more predictable |
| `--quantize` | none | none / 8bit / 4bit |
| `--dtype` | float16 | float16 / bfloat16 |
| `--max-length` | 640 | Max latent frames (640 ≈ 30s) |
| `--force-speaker` | off | Flag; enables speaker KV scaling |
| `--speaker-scale` | 1.5 | Scale when `--force-speaker` enabled |
| `--verbose` | off | Progress bar + timing |
| `--weights` | `~/.cache/echo-tts-mlx/` | Path to converted weight directory |

---

## Appendix A: Amendment Log

**Review 1 (2026-02-28):** 9 amendments from initial spec review.
1. Added Phase 0 (DAC feasibility spike)
2. Pinned weight conversion layout rules
3. Added packaging spec
4. Defined CLI and Python API
5. Added performance targets
6. Expanded quantization strategy
7. Specified speaker embedding pipeline
8. Addressed long-form generation
9. Added error handling

**Review 2 (2026-02-28):** 20 findings from upstream source code analysis.
1. 🔴 Fixed sample rate: 24kHz → 44,100 Hz
2. 🔴 Added PCA state transform (Section 2.6) — missing critical pipeline component
3. 🔴 Rewrote speaker pipeline: DAC latents, not mel spectrograms (Section 2.5)
4. 🔴 Fixed tokenizer: raw UTF-8 bytes, not Whisper/tiktoken (Section 2.4)
5. 🔴 Two CFG scales: `cfg_text` + `cfg_speaker` (Section 4)
6. 🟠 Documented model hyperparameters (Section 2.2)
7. 🟠 Added weight normalization folding (Section 3)
8. 🟠 Documented separate fish-s1-dac-min weight repo (Section 3)
9. 🟠 Autoencoder defaults to float32 (Sections 3, 6)
10. 🟠 Added latent trimming heuristic (Section 4)
11. 🟠 Added speaker KV scaling / Force Speaker (Section 2.5)
12. 🟠 Added blockwise module pruning (Sections 2.3, 3)
13. 🟠 Documented AE_DOWNSAMPLE_FACTOR = 2048 (Section 1)
14. 🟠 Replaced einops with manual reshape (Section 2.7)
15. 🟠 Added license section (Section 1)
16. 🟡 Noted descript-mlx as reference (Section 2.7)
17. 🟡 Confirmed ConvTranspose1d in MLX 0.31+ (Section 2.7); bumped min version
18. 🟡 Softened RTF target to < 2.0 with < 1.0 stretch (Section 1)
19. 🟡 Specified soxr for resampling (Sections 2.5, 8, 9)
20. 🟡 Documented MAX_TEXT_LENGTH = 768 (Section 2.4)

**Review 3 (2026-02-28):** Deep-dive into `model.py`, `inference.py`, `autoencoder.py` source code.
1. 🔴 Added KV caching strategy — text/speaker KV computed once, reused across all 40 steps (Sections 2.3, 4)
2. 🔴 Documented exact CFG formula — custom "independent guidances", NOT standard CFG (Section 4)
3. 🔴 Documented batched CFG — 3× batch in single forward pass for efficiency (Section 4)
4. 🔴 Added time-dependent CFG — only active when `cfg_min_t <= t <= cfg_max_t` (Section 4)
5. 🔴 Added timestep schedule — `linspace(1, 0, steps+1) * 0.999` (Section 4)
6. 🟠 Added truncation factor — scales initial noise, default 0.8 (Section 4, CLI)
7. 🟠 Specified DAC latent dim = 1024 → PCA components shape (80, 1024) (Section 2.6)
8. 🟠 Expanded DAC architecture — full `build_ae()` parameters including 24 transformer layers (Section 2.7)
9. 🟠 Documented RoPE variant — complex-number, half-rotary in JointAttention (Section 2.3)
10. 🟠 Documented QK normalization — per-head RMSNorm on Q and K (Section 2.3)
11. 🟠 Documented sigmoid gated attention in both attention types (Section 2.3)
12. 🟠 Documented SpeakerEncoder `/6` divisor (Section 2.3, 2.5)
13. 🟠 Added audio normalization on reference load (Section 2.5)
14. 🟠 Added speaker KV scale reversal behavior (Section 4)
15. 🟠 Expanded DAC decode path — goes through quantizer post-transformer + upsample (Section 2.7)
16. 🟡 Fixed tokenizer — added `(` prefix check and `S2` tag check (Section 2.4)
17. 🟡 Added test fixture generation script (Section 7)
18. 🟡 Added `--weights` and `--truncation-factor` CLI flags (Section 10)
19. 🟡 Added temporal score rescaling note as optional feature (Section 4)
20. 🟡 Updated Phase 0 and Phase 3 for full DAC complexity (Section 5)

**Review 4 (2026-03-01):** Phase 6 quantization implementation spec.
1. 🔴 Added Phase 6: Quantization — full implementation plan with `nn.Module` refactor strategy, selective quantization rules, save/load protocol
2. 🔴 Documented parameter budget: 2.38B params / 4.54 GB at float16, with per-component breakdown
3. 🟠 Added quality validation protocol: MCD thresholds, determinism gates, speaker cloning fidelity checks
4. 🟠 Added quantized benchmark extensions (Tier 1 + Tier 3 variants)
5. 🟠 Added 10 quantization-specific tests to test plan
6. 🟡 Documented risk matrix: `nn.Module` refactor parity, 4-bit quality risk, group size sensitivity
