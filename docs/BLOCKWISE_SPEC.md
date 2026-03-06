# Blockwise Generation — Specification

> **Reviewed for accuracy against implementation on 2026-03-05.**
> **Upstream reference:** [`inference_blockwise.py`](https://github.com/jordandare/echo-tts/blob/main/inference_blockwise.py)
> **Blog section:** [Dynamic-Block-wise Diffusion](https://jordandarefsky.com/blog/2025/echo/#dynamic-block-wise-diffusion)
> **Weights:** Already included in [`jordand/echo-tts-base`](https://huggingface.co/jordand/echo-tts-base) — fully fine-tuned (not LoRA)

---

## 1. Overview

### What It Is

Blockwise generation extends Echo-TTS from fixed 30-second whole-sequence diffusion to **incremental block-by-block generation**. Instead of denoising all 640 latent frames at once, the model generates audio in smaller blocks (e.g., 128, 256 frames), using previously generated blocks as conditioning context for subsequent ones.

The upstream model was fine-tuned for this capability with a latent-prefix encoder initialized from the speaker encoder weights, trained with Adam for 100K steps. The released HuggingFace checkpoint (`jordand/echo-tts-base`) includes these blockwise weights — the standard inference path simply deletes them via `delete_blockwise_modules=True`.

### What It Unlocks

| Capability | Description |
|---|---|
| **Streaming TTS** | Decode and play each block immediately — S1-DAC is fully causal |
| **Lower TTFB** | Generate a short first block (e.g., 64 frames ≈ 3s), decode it while generating the rest |
| **Audio continuations** | Given existing audio, generate a continuation (seconds 10–20 given seconds 0–10) |
| **Long-form up to model max** | Chain blocks up to `max_latent_length` total frames (640 for current checkpoints) |

### Constraints

- Block sizes must sum to ≤ `config.max_latent_length` latent frames (640 for current checkpoints).
- Continuation latent length + block sizes must also sum to ≤ `config.max_latent_length`.
- **Minimum block size:** 4 frames (hard floor, equals `speaker_patch_size`). Recommended minimum: 32 frames (~1.5s) for reasonable audio quality. Blocks < 16 frames (~0.75s) will produce very short segments with likely poor quality.
- Blockwise quality is less thoroughly tested upstream; smaller CFG scales may work better.
- The latent-prefix encoder uses `speaker_patch_size=4`, so block sizes and continuation lengths should be divisible by 4 for clean patching (non-divisible lengths will be zero-padded).

### Performance Tradeoff

Blockwise trades **total throughput** for **lower time-to-first-audio (TTFB)** and streaming capability.

Each block runs its own full diffusion loop (all `num_steps` Euler steps). The weight-loading bandwidth per forward pass (~3GB at float16) is the dominant cost, not the attention computation. For `k` blocks totaling the same output length as standard:

| Mode | Forward Passes | Latent Encoder Runs | Approx. Cost vs Standard |
|---|---|---|---|
| Standard (640 frames) | ~80 (40 steps × dual CFG) | 0 | 1.0× |
| Blockwise [320, 320] | ~160 | 2 | ~2× |
| Blockwise [128, 128, 64] | ~240 | 3 | ~2.5–3× |

**The benefit:** First audio is available after only the first block's diffusion completes. For `block_sizes=[64, 256, 320]`, the first ~3s of audio is ready in ~1/10th the total generation time.

Users should choose block sizes based on their latency vs. throughput priorities. For batch/offline generation, standard mode is always faster. For interactive/streaming use cases, blockwise with a small first block is preferred.

---

## 2. Architecture Changes

### 2.1 New Modules

Three new components are needed in `MlxEchoDiT`, all derived from the existing speaker encoder architecture:

#### Latent-Prefix Encoder (`latent_encoder`)

Identical architecture to `speaker_encoder`:
- **Input projection:** `Linear(speaker_patch_size × latent_size, speaker_model_size)` = `Linear(320, 1280)`
- **14 transformer blocks:** Causal self-attention, same as speaker encoder blocks
- **Activation scaling:** `÷ 6.0` after input projection (same trick as speaker encoder)
- **RMSNorm output:** `latent_norm` with shape `(speaker_model_size,)` = `(1280,)`

The latent-prefix encoder processes **previously generated clean latents** (not noised). During blockwise sampling, after each block is fully denoised, the clean result is fed through this encoder to produce context for the next block.

**Weight initialization (upstream training):** Initialized from speaker encoder weights. In our case, the fine-tuned weights are already in the HF checkpoint — we just need to stop pruning them.

#### Per-Layer Latent KV Projections (`wk_latent`, `wv_latent`)

In each of the 24 `JointAttention` blocks:

```
wk_latent: Linear(speaker_model_size, model_size) = Linear(1280, 2048), bias=False
wv_latent: Linear(speaker_model_size, model_size) = Linear(1280, 2048), bias=False
```

These project the latent-prefix encoder output into K/V for cross-attention, identical in shape and role to the existing `wk_speaker`/`wv_speaker` projections.

#### Latent Norm (`latent_norm`)

```
latent_norm: RMSNorm(speaker_model_size) = RMSNorm(1280)
```

Applied to latent encoder output before KV projection, matching the existing `speaker_norm` pattern.

### 2.2 Weight Key Mapping

Upstream PyTorch checkpoint keys for blockwise modules:

| Upstream Key Pattern | MLX Key Pattern | Shape |
|---|---|---|
| `latent_encoder.in_proj.{weight,bias}` | `latent_encoder.in_proj.{weight,bias}` | `(1280, 320)` / `(1280,)` |
| `latent_encoder.blocks.{i}.attention.{wq,wk,wv,wo,gate}.weight` | Same | `(1280, 1280)` |
| `latent_encoder.blocks.{i}.attention.{q_norm,k_norm}.weight` | Same | `(10, 128)` |
| `latent_encoder.blocks.{i}.{attention_norm,mlp_norm}.weight` | Same | `(1280,)` |
| `latent_encoder.blocks.{i}.mlp.{w1,w3}.weight` | Same | `(3328, 1280)` |
| `latent_encoder.blocks.{i}.mlp.w2.weight` | Same | `(1280, 3328)` |
| `latent_norm.weight` | `latent_norm.weight` | `(1280,)` |
| `blocks.{i}.attention.wk_latent.weight` | Same | `(2048, 1280)` |
| `blocks.{i}.attention.wv_latent.weight` | Same | `(2048, 1280)` |

Total new parameters: ~294M (latent encoder) + ~1.2M (latent norm) + ~118M (24 × wk/wv_latent) ≈ **413M additional parameters** (~788 MB at float16).

### 2.3 Modified Joint Attention

The decoder's `_joint_attention` method must be extended to accept an optional 4th KV source:

**Current KV concatenation order:**
```
keys   = [self_k, text_k, speaker_k]
values = [self_v, text_v, speaker_v]
```

**New KV concatenation order (blockwise):**
```
keys   = [self_k, latent_k, text_k, speaker_k]
values = [self_v, latent_v, text_v, speaker_v]
```

This matches the upstream `JointAttention.forward()` which concatenates in order: `[self, latent, text, speaker]`.

#### Latent Prefix Masking

The latent KV cache requires **position-based causal masking** — each denoiser position can only attend to latent-prefix positions from *strictly prior* blocks:

```python
# Upstream logic:
latent_positions = arange(latent_kv_length) * speaker_patch_size
latent_mask = (latent_positions[None, :] < start_pos).expand(batch, latent_kv_length)
```

Where `start_pos` is the starting latent frame index of the current block being generated. This ensures:
- Block 0 (start_pos=0): Cannot attend to any latent prefix (mask is all False)
- Block 1 (start_pos=128): Can attend to latent positions where `position * 4 < 128`, i.e., the first 32 patched positions
- Block 2 (start_pos=256): Can attend to latent positions where `position * 4 < 256`, i.e., the first 64 patched positions

If no latent KV cache is provided (standard non-blockwise mode), the latent component is simply omitted — zero-length tensors concatenated into the key/value streams.

#### RoPE for Latent Keys — Custom Implementation Required

Latent prefix keys receive **RoPE with patch-dilated positions** (non-contiguous: `[0, 4, 8, 12, ...]`):

```python
# Upstream (PyTorch):
positions = arange(seq_len) * speaker_patch_size  # [0, 4, 8, 12, ...]
freqs_cis = precompute_freqs_cis(head_dim, max_pos)
freqs_latent = freqs_cis[positions]  # Gather at non-contiguous indices
xk = apply_rotary_half(xk, freqs_latent)
```

**⚠️ `mx.fast.rope` cannot do this.** It only supports contiguous position ranges (via `offset`). We need a custom RoPE helper for non-contiguous positions:

```python
def _apply_rotary_at_positions(x: Any, positions: Any, mx: Any, theta: float = 10000.0) -> Any:
    """Apply RoPE at arbitrary (non-contiguous) position indices.
    
    x: (B, H, T, D)
    positions: (T,) integer array of position indices
    """
    D = int(x.shape[-1])
    half_d = D // 2
    
    # Build freq table for the specific positions
    freqs = 1.0 / (theta ** (mx.arange(0, half_d, dtype=mx.float32) / half_d))
    # positions: (T,) × freqs: (D/2,) → angles: (T, D/2)
    angles = mx.expand_dims(positions.astype(mx.float32), -1) * mx.expand_dims(freqs, 0)
    cos_vals = mx.cos(angles)  # (T, D/2)
    sin_vals = mx.sin(angles)  # (T, D/2)
    
    # Reshape for broadcast: (1, 1, T, D/2)
    cos_vals = mx.reshape(cos_vals, (1, 1, cos_vals.shape[0], half_d))
    sin_vals = mx.reshape(sin_vals, (1, 1, sin_vals.shape[0], half_d))
    
    # Split x into pairs and rotate
    x1 = x[..., :half_d]
    x2 = x[..., half_d:]
    return mx.concatenate([x1 * cos_vals - x2 * sin_vals, x1 * sin_vals + x2 * cos_vals], axis=-1)


def _apply_half_rotary_at_positions(x: Any, positions: Any, mx: Any) -> Any:
    """Half-rotary: apply RoPE only to first half of heads at given positions."""
    h = int(x.shape[1])
    h_half = h // 2
    if h_half == 0:
        return x
    x_rot = _apply_rotary_at_positions(x[:, :h_half, :, :], positions, mx)
    return mx.concatenate([x_rot, x[:, h_half:, :, :]], axis=1)
```

Both the `head_dim` of the main DiT (2048/16 = 128) and the speaker/latent encoder (1280/10 = 128) happen to be 128, so the same `theta=10000.0` frequency computation applies. But the RoPE is computed using the **DiT's `head_dim`**, since the latent keys are projected into the DiT's attention space via `wk_latent`.

#### Decoder RoPE Offset (`start_pos`)

Decoder self-attention RoPE is offset by `start_pos` in blockwise mode:

```python
# In _joint_attention, with start_pos:
q = _apply_half_rotary(q, mx, offset=start_pos)
k_self = _apply_half_rotary(k_self, mx, offset=start_pos)
```

This is already implemented in `model.py` by threading `start_pos` through
`forward()` -> `_joint_attention()` -> `_apply_half_rotary()`.

### 2.4 New Method: `get_kv_cache_latent`

```python
def get_kv_cache_latent(
    self,
    prefix_latent: Any,  # (B, T, 80) — full prefix buffer (may include zeros for future blocks)
) -> list[tuple[Any, Any]]:
    """Encode prefix latents and project to per-layer KV caches.
    
    The full prefix buffer is encoded, including zero-padded slots for
    not-yet-generated blocks. Masking inside JointAttention (via start_pos)
    controls which positions the decoder actually attends to.
    """
    # 1. Run through latent_encoder (14 causal transformer layers)
    #    Input: (B, T, 80) → patched: (B, T/4, 320) → encoded: (B, T/4, 1280)
    latent_state = self._encode_latent(prefix_latent)
    
    # 2. Apply latent_norm
    latent_state = rms_norm(latent_state, self.t("latent_norm.weight"), ...)
    
    # 3. Compute dilated RoPE positions (non-contiguous)
    seq_len = latent_state.shape[1]
    positions = mx.arange(seq_len) * speaker_patch_size  # [0, 4, 8, 12, ...]
    
    # 4. Project through each block's wk_latent / wv_latent with RoPE
    kv = []
    for i in range(num_layers):
        k = wk_latent[i](latent_state)  # (B, T/4, 2048)
        v = wv_latent[i](latent_state)
        k = reshape_heads(k)  # (B, H, T/4, D)  where H=16, D=128
        v = reshape_heads(v)
        k = k_norm(k)
        k = _apply_half_rotary_at_positions(k, positions, mx)  # Custom non-contiguous RoPE
        kv.append((k, v))
    
    return kv
```

**Important:** The encoder always processes the **full prefix buffer** — including zero-padded slots for blocks that haven't been generated yet. This matches the upstream exactly. The zeros get encoded into KV entries, but the position-based mask in `JointAttention` prevents the decoder from attending to positions beyond `start_pos`. Do NOT optimize by encoding only the valid prefix — it would change the causal encoder's hidden states for earlier positions.

### 2.5 Modified `forward` Method

The `forward` method gains two new optional parameters:

```python
def forward(
    self,
    latents,
    timesteps,
    *,
    kv_text,
    kv_speaker,
    text_mask,
    speaker_mask,
    # NEW:
    start_pos: int = 0,                              # RoPE offset for current block
    kv_latent: list[tuple[Any, Any]] | None = None,  # Latent prefix KV cache
) -> Any:
```

Changes inside `forward`:
1. **RoPE offset:** Use `start_pos` for decoder self-attention RoPE computation
2. **Joint attention mask:** Include latent prefix in mask construction, with position-based causal masking using `start_pos`
3. **KV concatenation:** Add latent KVs between self and text KVs in the concatenation order

**Backward compatibility:** When `kv_latent is None` and `start_pos == 0`, behavior is identical to current non-blockwise mode. No breaking changes.

---

## 3. Weight Conversion Changes

### 3.1 Remove Blockwise Pruning

The current conversion (`conversion.py`) prunes blockwise keys by default:

```python
PRUNE_PREFIXES = ("latent_encoder.", "latent_norm")
PRUNE_CONTAINS = (".wk_latent", ".wv_latent")
```

**Changes needed:**

1. Add a `--include-blockwise` / `--no-prune-blockwise` flag to `convert` CLI (default: still prune for backward compatibility).
2. When blockwise is included, convert all latent_encoder/latent_norm/wk_latent/wv_latent keys with the same dtype and format as existing weights.
3. Update `weight_map.json` to include blockwise key mappings.

### 3.2 Config Update

Add blockwise capability flag to `config.json`:

```json
{
  "blockwise_capable": true,
  "latent_encoder_num_layers": 14,
  "latent_encoder_model_size": 1280
}
```

The loader can use this to determine whether blockwise modules are available in the checkpoint.

### 3.3 Memory Impact

| Mode | DiT Size (f16) | Additional Blockwise (f16) | Total |
|---|---|---|---|
| Standard (pruned) | 4,541 MB | — | 4,541 MB |
| With blockwise | 4,541 MB | +788 MB | **5,329 MB** |
| 8-bit with blockwise | 2,289 MB | +394 MB | **2,683 MB** |

On 16GB Apple Silicon, blockwise at float16 is feasible. On 8GB machines, 8-bit quantization is recommended.

---

## 4. Sampler Changes

### 4.1 New Sampling Function

Add `sample_blockwise_euler_cfg` to `sampler.py`:

```python
@dataclass(frozen=True)
class BlockwiseSamplerConfig:
    """Configuration for blockwise Euler sampling."""
    block_sizes: list[int]                    # e.g., [128, 128, 64]
    num_steps: int = 32
    cfg_scale_text: float = 3.0
    cfg_scale_speaker: float = 5.0           # Note: upstream default is 5.0, not 8.0
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    truncation_factor: float | None = 0.8
    init_scale: float = 0.999
    speaker_kv_scale: float | None = None
    speaker_kv_max_layers: int | None = None
    speaker_kv_min_t: float | None = None
```

### 4.2 Blockwise Sampling Algorithm

```
Input:
  - text_ids, text_mask
  - speaker_latents, speaker_mask
  - block_sizes: list[int]              # e.g., [128, 128, 64] → total 320 frames
  - continuation_latent: (B, T, 80) | None  # previously generated audio as latent prefix
  - seed: int

Algorithm:

1. Pre-compute text KV cache (once)
2. Pre-compute speaker KV cache (once)
3. Initialize `prefix_latent = zeros(B, continuation_length + sum(block_sizes), 80)`
4. If continuation_latent is provided:
     a. Copy it into `prefix_latent[:, :continuation_length, :]`
     b. Set start_pos = continuation_latent.shape[1]
   Else:
     a. start_pos = 0

5. For each block_size in block_sizes:
     a. If speaker_kv_scale: re-apply scaling to speaker KV cache
        (see Speaker KV Scale Lifecycle below)
     b. Compute latent KV cache from FULL prefix_latent buffer
        - Encode entire buffer (including zero-padded future slots) through latent_encoder
        - Project to per-layer K/V with dilated RoPE (non-contiguous positions)
        - Batch 3× for CFG: kv_latent_full = [kv, kv, kv] along batch dim
        - Also extract kv_latent_cond = kv[:batch_size] for non-CFG steps
     c. Sample x_t = randn(B, block_size, 80) * truncation_factor
     d. For each Euler step i = 0..num_steps-1:
          i.   t, t_next = schedule[i], schedule[i+1]
          ii.  If CFG active (cfg_min_t <= t <= cfg_max_t):
                 - Forward pass with batch=3, passing:
                   * kv_text_full, kv_speaker_full (text/speaker masked per CFG pattern)
                   * kv_latent_full (identical for all 3 — latent context is NOT masked per CFG)
                   * start_pos for RoPE offset + latent masking
                 - Apply independent dual CFG formula
               Else:
                 - Single forward pass with kv_latent_cond, start_pos
          iii. If speaker_kv_scale and t crosses speaker_kv_min_t:
                 restore unscaled speaker KV snapshot
          iv.  x_t = x_t + v_pred * (t_next - t)
          v.   mx.eval(x_t)
     e. Store denoised block: prefix_latent[:, start_pos:start_pos+block_size] = x_t
     f. Invoke on_block_complete callback if provided
     g. start_pos += block_size

6. Return prefix_latent (including continuation prefix if provided)
```

#### Speaker KV Scale Lifecycle Across Blocks

When `speaker_kv_scale` is set, the implementation applies a per-block
scale-up and mid-diffusion restore from an unscaled snapshot:

```
At block start: derive scaled KV from unscaled snapshot
Mid-block:      restore exact unscaled KV snapshot
Next block:     derive scaled KV again from the same snapshot
...
```

This avoids cumulative floating-point drift from repeated multiply/divide cycles
across many blocks.

### 4.3 Key Differences from Standard Sampling

| Aspect | Standard | Blockwise |
|---|---|---|
| Noise shape | `(1, sequence_length, 80)` | `(1, block_size, 80)` per block |
| Forward passes per step | 3 (CFG) or 1 | Same, plus latent KV in attention |
| Latent KV cache | None | Re-computed each block from full prefix buffer (incl. zero-padded future) |
| RoPE offset | Always 0 | `start_pos` accumulates across blocks |
| Default `cfg_scale_speaker` | 8.0 | 5.0 (upstream recommendation for blockwise) |
| Per-block re-seeding | N/A | Use single RNG, advance sequentially |
| `on_step` callback semantics | `step=1..num_steps` | `step=1..num_steps` per block (resets each block) |

### 4.4 Continuation Support

For audio continuations, the user provides existing audio that gets encoded into latents:

```
existing_audio → DAC encode → PCA encode → continuation_latent (B, T, 80)
```

The continuation latent is prepended to the generation buffer and used as-is (not denoised). The model generates new audio that seamlessly continues from it.

**⚠️ CRITICAL: The text prompt MUST include the full text of the continuation prefix.** If the continuation audio says "Hello, world" and you want to generate "How are you?", the prompt must be the full concatenation:

```
"[S1] Hello, world. How are you?"
```

**NOT** just `"[S1] How are you?"`. The model aligns text to latent positions — omitting the prefix text will produce garbled, misaligned audio. This is the #1 foot-gun with continuation mode.

If the original text is unknown, use [WhisperD](https://huggingface.co/jordand/whisper-d-v1a) to transcribe the continuation audio in-distribution.

---

## 5. Pipeline Changes

### 5.1 New Method: `generate_blockwise`

```python
def generate_blockwise(
    self,
    *,
    text: str,
    block_sizes: list[int],
    speaker_latents: Any | None = None,
    speaker_audio: Any | None = None,
    speaker_mask: Any | None = None,
    continuation_audio: Any | None = None,
    continuation_latents: Any | None = None,
    seed: int | None = 0,
    num_steps: int = 32,
    cfg_scale_text: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    truncation_factor: float | None = 0.8,
    speaker_kv_scale: float | None = None,
    speaker_kv_max_layers: int | None = None,
    speaker_kv_min_t: float | None = None,
    trim_latents: bool = True,
    trim_mode: str = "latent",
    return_latents: bool = False,
    on_block_complete: Callable[[int, int, Any], None] | None = None,
    decode_intermediate_blocks: bool = True,
    progress_callback: Callable[[int, int, float, bool], None] | None = None,
) -> Any:
```

**Parameters:**
- `block_sizes`: List of latent frame counts per block. Sum must be ≤ `config.max_latent_length` (640 default).
- `continuation_audio`: Existing audio to continue from (will be encoded to latents).
- `continuation_latents`: Pre-encoded continuation latents (alternative to `continuation_audio`).
- `on_block_complete(block_idx, total_blocks, data)`: Callback invoked after each block is denoised.
  - `data` is decoded block audio when `decode_intermediate_blocks=True` (default)
  - `data` is raw block latents when `decode_intermediate_blocks=False`
- `progress_callback(step, num_steps, t, cfg_active)`: receives per-block step progress (`step=1..num_steps` each block).

**Validation:**
- `sum(block_sizes) <= config.max_latent_length`
- If continuation provided: `continuation_length + sum(block_sizes) <= config.max_latent_length`
- Each block size should be > 0
- Warn if any block_size is not divisible by `speaker_patch_size` (4)

### 5.2 New Method: `encode_continuation`

```python
def encode_continuation(
    self,
    *,
    audio: Any | None = None,
    latents: Any | None = None,
) -> tuple[Any, int]:
    """Encode audio or validate latents for use as a continuation prefix.
    
    Returns (continuation_latents, actual_length) where actual_length is
    the number of valid latent frames (trimmed to patch_size divisibility).
    """
```

### 5.3 Streaming Callback Pattern

```python
def on_block_done(block_idx: int, total: int, block_audio: mx.array):
    """Called after each block is fully denoised."""
    # Play or write the decoded chunk immediately
    play_audio(block_audio)

model.generate_blockwise(
    text="[S1] This is a long passage that will be generated in three blocks...",
    block_sizes=[128, 128, 128],
    speaker_audio=ref_audio,
    on_block_complete=on_block_done,
)
```

`on_block_complete` receives decoded block audio by default (`decode_intermediate_blocks=True`).

---

## 6. CLI Changes

### 6.1 New Subcommand Flags

Add blockwise options to the `generate` subcommand:

```bash
echo-tts-mlx generate \
  --text "[S1] Hello, this is blockwise generation." \
  --speaker ref.wav \
  --output out.wav \
  --weights weights/converted-blockwise/ \
  --blockwise 128,128,64 \              # Comma-separated block sizes
  --continuation existing.wav \          # Optional: continue from existing audio
  --cfg-speaker 5.0 \                   # Lower default for blockwise
  --steps 40 \
  --verbose
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--blockwise` | `str` | None | Comma-separated block sizes (e.g., `128,128,64`). Enables blockwise mode. |
| `--continuation` | `Path` | None | Audio file to continue from. Requires `--blockwise`. |

**Behavior:**
- When `--blockwise` is provided, uses `generate_blockwise()` instead of `generate()`.
- When `--blockwise` is absent, standard whole-sequence generation (unchanged).
- Validates: sum of block sizes ≤ max_latent_length.
- Validates: each block size ≥ 4 (speaker_patch_size). Warns if < 32.
- With `--verbose`: shows per-block completion and per-step diffusion progress.
- When `--continuation` is used, prints a reminder that `--text` must include the full continuation transcript.

### 6.2 Updated `convert` Command

```bash
echo-tts-mlx convert \
  --dit weights/upstream/dit_model.safetensors \
  --dac weights/upstream/dac_model.safetensors \
  --pca weights/upstream/pca_state.safetensors \
  --output weights/converted-blockwise/ \
  --include-blockwise                   # NEW: don't prune blockwise modules
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--include-blockwise` | `flag` | False | Include latent_encoder/wk_latent/wv_latent in converted weights |

---

## 7. Config Changes

### 7.1 No ModelConfig Struct Changes Required

The latent encoder reuses the speaker encoder's hyperparameters (`speaker_model_size`, `speaker_num_layers`, `speaker_num_heads`, `speaker_intermediate_size`, `speaker_patch_size`). No new config fields are architecturally necessary.

### 7.2 Runtime Detection

The model detects blockwise capability by checking for the presence of `latent_encoder.in_proj.weight` in the loaded checkpoint. If absent, blockwise methods raise a clear error:

```python
if not self._has_blockwise_modules:
    raise RuntimeError(
        "Blockwise generation requires weights converted with --include-blockwise. "
        "Re-run: echo-tts-mlx convert --include-blockwise ..."
    )
```

---

## 8. Quantization Compatibility

Blockwise modules follow the same quantization rules as existing modules:

| Module | Quantization Rule |
|---|---|
| `latent_encoder.blocks.{i}.attention.{wq,wk,wv,wo,gate}` | Same as speaker encoder (8-bit) |
| `latent_encoder.blocks.{i}.mlp.{w1,w2,w3}` | Same as speaker encoder (8-bit) |
| `latent_encoder.in_proj` | Skip (input projection) |
| `latent_norm` | Skip (norm) |
| `blocks.{i}.attention.wk_latent` | Same as `wk_text`/`wk_speaker` (follows DiT block rule) |
| `blocks.{i}.attention.wv_latent` | Same as `wv_text`/`wv_speaker` (follows DiT block rule) |
| `latent_encoder.blocks.{i}.{attention_norm,mlp_norm}` | Skip (norm) |
| `latent_encoder.blocks.{i}.attention.{q_norm,k_norm}` | Skip (norm) |

The existing `_quantize_predicate` function already handles these patterns via prefix/suffix rules. The latent encoder modules match `text_encoder.`/`speaker_encoder.` conventions, so the predicate should work without modification — but this must be verified in tests.

---

## 9. Test Plan

### 9.1 Unit Tests

| Test | What It Verifies |
|---|---|
| `test_latent_encoder_forward` | Latent encoder produces correct output shape `(B, T/4, 1280)` from `(B, T, 80)` input |
| `test_latent_encoder_causal` | Latent encoder uses causal attention (future positions masked) |
| `test_get_kv_cache_latent` | KV cache shape: 24 layers × `(B, H, T/4, D)`, dtype float32 |
| `test_kv_cache_latent_rope` | Latent KV keys have dilated RoPE positions at `i * speaker_patch_size` |
| `test_joint_attention_with_latent` | 4-way KV concatenation: `[self, latent, text, speaker]` in correct order |
| `test_latent_mask_block0` | Block 0 (start_pos=0): latent mask is all-False (no prefix to attend to) |
| `test_latent_mask_block1` | Block 1 (start_pos=128): latent mask allows positions where `pos * 4 < 128` |
| `test_rope_offset` | Decoder self-attention RoPE uses `start_pos` offset correctly |
| `test_forward_backward_compat` | `forward()` with `kv_latent=None, start_pos=0` matches non-blockwise output exactly |
| `test_rotary_at_positions_vs_contiguous` | `_apply_rotary_at_positions(x, [0,1,2,...])` matches `mx.fast.rope(x, offset=0)` |
| `test_rotary_at_positions_dilated` | `_apply_rotary_at_positions(x, [0,4,8,...])` produces different output than contiguous |

### 9.2 Integration Tests

| Test | What It Verifies |
|---|---|
| `test_blockwise_single_block` | Blockwise with `block_sizes=[640]` ≈ standard generation (same noise, close output) |
| `test_blockwise_multi_block_shapes` | `block_sizes=[128, 128, 64]`: output shape is `(1, 320, 80)` |
| `test_blockwise_continuation` | Continuation latent is preserved in output prefix |
| `test_blockwise_continuation_length` | `continuation_length + sum(block_sizes) <= 640` enforced |
| `test_blockwise_determinism` | Same seed + block_sizes → identical output across runs |
| `test_conversion_include_blockwise` | `--include-blockwise` preserves all latent_encoder/wk_latent/wv_latent keys |
| `test_conversion_prune_blockwise` | Default conversion still prunes blockwise keys (backward compat) |
| `test_blockwise_quantized_8bit` | Blockwise works with 8-bit quantized weights |

### 9.3 End-to-End Tests

| Test | What It Verifies |
|---|---|
| `test_e2e_blockwise_generates_audio` | Full pipeline: text + speaker → blockwise → decode → valid WAV |
| `test_e2e_blockwise_vs_standard_quality` | MCD between blockwise `[640]` and standard `640` < 10.0 dB |
| `test_e2e_continuation_seamless` | No audible click/pop at block boundaries in continuation |
| `test_e2e_streaming_callback` | `on_block_complete` fires for each block with valid audio |
| `test_e2e_cli_blockwise` | CLI `--blockwise 128,128,64` produces valid output file |
| `test_e2e_parity_upstream_2block` | **Parity with PyTorch:** 2-block generation with fixed noise, MCD < 5.0 dB vs upstream output |

### 9.4 Fixture Generation

Extend `scripts/generate_fixtures.py` to produce blockwise reference data from the upstream PyTorch implementation:

- `blockwise_prefix_latent.npy`: Full prefix buffer after 2-block generation `[128, 128]`
- `blockwise_latent_kv_layer0.npy`: Latent KV cache for first layer (block 1, after block 0 is filled)
- `blockwise_block0_output.npy`: Denoised latents for first block (128 frames)
- `blockwise_block1_output.npy`: Denoised latents for second block (128 frames)
- `blockwise_noise_blocks.npy`: Pre-generated noise for both blocks (for deterministic parity)
- `continuation_latent.npy`: Pre-encoded continuation latent from reference audio

These fixtures enable the critical `test_e2e_parity_upstream_2block` test without requiring PyTorch at test time.

---

## 10. Implementation Plan

### Step 1: Weight Conversion Update

1. Add `--include-blockwise` flag to `convert` CLI
2. When enabled, skip pruning of `latent_encoder.*`, `latent_norm*`, `*.wk_latent`, `*.wv_latent`
3. Update `weight_map.json` generation to include blockwise keys
4. Add conversion test: verify all expected keys present with correct shapes
5. **Gating test:** Load converted blockwise weights, verify key count matches upstream checkpoint (minus buffer keys)

### Step 2: Model Architecture Extension

1. Add `latent_encoder` to `_build_module_tree()`:
   - Mirror `speaker_encoder` structure exactly (in_proj + 14 blocks + norms)
   - Add `latent_norm` weight node
2. Add `wk_latent` and `wv_latent` Linear modules to each of the 24 decoder blocks
3. Add `_encode_latent()` method — identical to `_encode_speaker()` but using latent encoder weights
4. Add `get_kv_cache_latent()` method with dilated RoPE
5. Modify `_joint_attention()`:
   - Accept optional `kv_latent` parameter
   - Accept `start_pos` for RoPE offset and latent masking
   - Construct 4-way KV concatenation when latent KV is present
   - Build latent position mask: `positions * patch_size < start_pos`
6. Modify `forward()` to accept and pass through `start_pos` and `kv_latent`
7. Add `_has_blockwise_modules` property for runtime detection
8. **Gating test:** All existing tests pass unchanged (backward compatibility). New `forward()` with default args produces identical output.

### Step 3: Sampler Extension

1. Add `BlockwiseSamplerConfig` dataclass
2. Implement `sample_blockwise_euler_cfg` function following algorithm in Section 4.2
3. Handle speaker KV scale/reversal per block (upstream applies scale at start of each block)
4. Re-compute latent KV cache at the start of each block from the full accumulated prefix
5. **Gating test:** Blockwise with `block_sizes=[640]` produces output that is close to standard generation (not identical due to latent prefix masking, but structurally similar)

### Step 4: Pipeline Integration

1. Add `encode_continuation()` method
2. Add `generate_blockwise()` method
3. Wire up `on_block_complete` streaming callback
4. Add validation: block size sum, continuation length, patch divisibility warnings
5. **Gating test:** Full pipeline produces valid audio for basic blockwise case

### Step 5: CLI Integration

1. Add `--blockwise` and `--continuation` flags to `generate` subcommand
2. Add `--include-blockwise` flag to `convert` subcommand
3. Parse comma-separated block sizes, validate
4. Route to `generate_blockwise()` when `--blockwise` is set
5. Update `--verbose` output to show per-block progress
6. **Gating test:** CLI round-trip: generate blockwise WAV, verify valid audio file

### Step 6: Documentation & Polish

1. Update `ARCHITECTURE.md` with blockwise pipeline diagram
2. Update `README.md` with blockwise usage examples
3. Add blockwise section to `GUIDE.md`
4. Run benchmark suite with blockwise variants
5. Update `MEMORY.md` with blockwise project status

---

## 11. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Latent KV cache precision drift across blocks | Medium | Audio artifacts at block boundaries | Float32 KV caches (already done for text/speaker) |
| RoPE offset miscalculation | Medium | Garbled audio | Fixture-based parity test against upstream PyTorch |
| Block boundary artifacts (clicks/pops) | Medium | Audible quality regression | S1-DAC is causal, so no inherent boundary issue; verify with listening tests |
| Memory pressure from latent encoder (additional 788MB) | Low | OOM on 8GB devices | Recommend 8-bit quantization; latent encoder quantizes well |
| Blockwise + quantization interaction | Low | Quality regression | Test 8-bit blockwise against float16 blockwise; same MCD thresholds |
| Breaking backward compatibility | Low | Existing non-blockwise usage affected | Default `kv_latent=None, start_pos=0` preserves exact behavior |
| Latent prefix re-encoding overhead per block | Medium | Slower than expected | Cache previous blocks' KV if latent encoder supports incremental KV (upstream notes this is possible since the encoder is causal) |

### Future Optimization: Incremental Latent KV Cache

The latent encoder is **causal**, which means we can cache its KV state across blocks instead of re-encoding the full prefix each time. The upstream blog notes:

> "technically (as the encoder is causal) only need to process the most recently generated latents if we have the encoder kv cache of earlier generated latents"

This optimization is deferred to a follow-up PR to keep the initial implementation simple and correct.

---

## Appendix: Upstream Reference Summary

### `inference_blockwise.py` Key Observations

1. **Latent prefix grows:** `prefix_latent` accumulates across blocks — initialized as `zeros(B, total_length, 80)`, filled in sequentially.
2. **Full prefix re-encoded each block:** `model.get_kv_cache_latent(full_prefix_latent)` is called fresh for each block. No incremental caching in upstream reference.
3. **Speaker KV scale applied per block:** If `speaker_kv_scale` is set, it's multiplied at the start of each block (not just once).
4. **Continuation latent prepended:** When continuing, `continuation_latent` is prepended to `prefix_latent`, and `start_pos` begins at `continuation_latent.shape[1]`.
5. **Single RNG:** One `torch.Generator` seeded once, producing noise for all blocks sequentially. Blocks get different noise naturally from the sequential `.randn()` calls.
6. **CFG batching:** Same 3-way batching as standard inference. All three batch elements share the same latent KV cache.
7. **No latent trimming per block:** Trimming is applied only to the final concatenated output.

### Upstream Default Parameters for Blockwise

```python
block_sizes = [128, 128, 64]    # ~15 seconds total
num_steps = 40
cfg_scale_text = 3.0
cfg_scale_speaker = 5.0         # Lower than standard 8.0
cfg_min_t = 0.5
cfg_max_t = 1.0
truncation_factor = 0.8
```

Note the lower `cfg_scale_speaker` (5.0 vs 8.0) — the upstream examples use a gentler speaker scale for blockwise, suggesting the latent prefix provides additional speaker context that reduces the need for aggressive KV-based speaker forcing.

---

## Appendix B: Amendment Log

**Review 1 (v1.0 → v1.1, 2026-03-03):** Self-review before implementation. 9 findings.
1. 🔴 `mx.fast.rope` cannot handle non-contiguous positions — added custom `_apply_rotary_at_positions()` helper with full pseudocode (Section 2.3)
2. 🔴 Decoder self-attention RoPE needs `offset=start_pos` — added signature changes for `_apply_rotary`, `_apply_half_rotary`, `_joint_attention` (Section 2.3)
3. 🔴 Full prefix buffer (including zero-padded future blocks) is always encoded — explicit warning not to optimize away (Section 2.4)
4. 🟠 CFG batching for latent KV — added explicit 3× batching pattern and kv_latent_cond extraction (Section 4.2)
5. 🟠 Speaker KV scale lifecycle across blocks — documented accumulation risk, per-block scale/reverse pattern (Section 4.2)
6. 🟠 Performance tradeoff quantified — blockwise is ~2–3× total compute for same output, benefit is TTFB (Section 1)
7. 🟠 Minimum block size specified — hard floor at 4 frames, recommended ≥ 32 (Section 1)
8. 🟡 Added upstream parity test `test_e2e_parity_upstream_2block` with MCD < 5.0 dB gate (Section 9.3)
9. 🟡 Continuation text warning strengthened to bold box + CLI reminder (Sections 4.4, 6.1)

**Review 2 (v1.1 → v1.2, 2026-03-05):** Post-implementation accuracy pass.
1. ✅ Added implementation review stamp and aligned constraints to `config.max_latent_length` (default 640)
2. ✅ Updated decoder RoPE section from planned changes to implemented behavior
3. ✅ Updated speaker KV lifecycle to snapshot-based scale/restore (no cumulative drift)
4. ✅ Added current `generate_blockwise(..., decode_intermediate_blocks=True)` callback contract
5. ✅ Documented blockwise `on_step` semantics (`step=1..num_steps` per block)

**Review 3 (v1.2 → v1.3, 2026-03-05):** Post-implementation code review. 16 findings across 3 severity levels, all resolved. Source review: `logs/BLOCKWISE_BENCHMARKS_REVIEW.md`. Fixes implemented in commits `aa024e6` (Batch 1), `d3eeff3` (Batch 2), `5c78f81` (Batch 3).

*Batch 1 — Correctness:*
1. 🔴 **C1 — Sampler step counter:** `sample_blockwise_euler_cfg` was emitting global step count via `on_step`. Changed to per-block steps (1..num_steps), matching standard sampler semantics. Benchmark `_on_progress` updated to track `current_block_idx` explicitly.
2. 🔴 **C2 — Custom RoPE:** `_apply_rotary_at_positions` implements RoPE manually because `mx.fast.rope` only accepts scalar offsets (can't do non-contiguous positions like `[0,4,8,12]`). Added docstrings documenting why, plus parity tests against `mx.fast.rope` for contiguous cases.
3. 🔴 **S2 — Speaker KV drift:** Repeated in-place multiply/divide across blocks accumulated floating-point error. Introduced immutable `kv_speaker_cond_unscaled` snapshot; scale and restore now derive from snapshot each block.
4. 🔴 **C3+C4 — Latent encoder masking + block 0 visibility:** Added upstream reference comments, `_encode_latent` docstring, latent visibility explanation. Added tests for encoder output shape, zero-padded regions, and visibility masks for block 0 (with/without continuation) and block 1.

*Batch 2 — Benchmark Accuracy:*
5. 🟠 **S1 — Block decode overhead:** Added `decode_intermediate_blocks` parameter to `generate_blockwise`. Benchmarks pass `False` to exclude per-block DAC decode from timing. CLI keeps `True` default for streaming.
6. 🟠 **S6 — Tier 3 eval barriers:** Added `mx.eval()` in `on_block_complete` callback. Progress callback documented as approximate (no reliable tensor to eval mid-step from outside sampler).
7. 🟠 **S5 — Compare script coverage:** Added `_print_blockwise_all` (generic iteration over all `bench_blockwise_*` keys) and `_print_tier3_blockwise` comparison.
8. 🟡 **S3 — Monkey-patching documentation:** Added block comment explaining why, how, and limitations of monkey-patching `get_kv_cache_latent` / `encode_continuation` for timing.
9. 🟡 **S4 — StandardCase docstring:** Added field docstring on `block_sizes` explaining it's only set by `build_blockwise_cases()`.

*Batch 3 — Documentation:*
10. 🟡 **M1 — BLOCKWISE_SPEC.md accuracy:** This file — reviewed against implementation, corrected constraints, callback contracts, sampler semantics.
11. 🟡 **M3 — ARCHITECTURE.md + GUIDE.md:** Aligned blockwise limit wording with `max_latent_length`.
12. 🟡 **M4 — README.md:** Verified, no correction needed.
13. 🟡 **M6 — CLI --filter help:** Added descriptive help text with examples.
14. 🟡 **M5 — Synthetic audio docstring:** Documented that continuation benchmarks use synthetic input.
15. 🟡 **M2 — BENCHMARKS.md:** Added blockwise placeholder section.

*Tests added (13 total):*

| Test | File |
|---|---|
| `test_blockwise_on_step_emits_per_block_steps` | `tests/test_sampler.py` |
| `test_custom_rope_matches_fast_rope` | `tests/test_blockwise.py` |
| `test_custom_rope_with_offset_matches_fast_rope` | `tests/test_blockwise.py` |
| `test_custom_rope_non_contiguous_positions` | `tests/test_blockwise.py` |
| `test_speaker_kv_no_drift_after_many_blocks` | `tests/test_blockwise.py` |
| `test_encode_latent_output_shape` | `tests/test_blockwise.py` |
| `test_encode_latent_zero_padded_region` | `tests/test_blockwise.py` |
| `test_latent_visibility_block0_no_continuation` | `tests/test_blockwise.py` |
| `test_latent_visibility_block0_with_continuation` | `tests/test_blockwise.py` |
| `test_latent_visibility_block1` | `tests/test_blockwise.py` |
| `test_generate_blockwise_no_decode_passes_latents` | `tests/test_blockwise.py` |
| `test_compare_finds_all_blockwise_keys` | `tests/test_benchmarks_compare.py` |
| `test_compare_tier3_blockwise` | `tests/test_benchmarks_compare.py` |

---

_Specification version: 1.3_
_Date: 2026-03-05_
_Author: Larry 🦞_
