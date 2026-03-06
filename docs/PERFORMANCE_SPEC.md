# Performance Optimization Spec

> **Goal:** Reduce end-to-end generation wall time by 3–5× on Apple Silicon through software optimizations alone (no hardware changes, no retraining).
>
> **Baseline (Mac mini M4, 8-bit, 32 steps):**
> - Max-length generation (seq=640): **66.9s** (RT factor 0.27×)
> - Medium clone (seq=300): **31.9s** (RT factor 0.43×)
> - Short uncond (seq=100): **10.1s** (RT factor 0.09×)
>
> **Target:** Max-length under 20s, medium under 10s, short under 4s.

---

## Optimization 1: Fused Flash Attention

**Impact estimate:** 1.5–2× speedup on DiT diffusion (80–90% of wall time)

### Current Implementation

`model.py` lines 611–622 and 681–705 implement attention manually:

```python
# Self-attention (encoder blocks)
scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) / math.sqrt(int(q.shape[-1]))
# ... masking ...
attn = mx.softmax(scores, axis=-1)
y = mx.matmul(attn, v)

# Joint attention (DiT blocks) — same pattern with concatenated KV
scores = mx.matmul(q, mx.transpose(k_full, (0, 1, 3, 2))) / math.sqrt(self.head_dim)
# ... mask slicing per span ...
attn = mx.softmax(scores, axis=-1)
y = mx.matmul(attn, v_full)
```

This materializes the full `(B, H, T_q, T_kv)` attention matrix in memory. For seq=640 with 32 heads and 3× CFG batch, that's `3 × 32 × 640 × (640 + T_text + T_speaker)` floats — significant memory bandwidth pressure.

### Target Implementation

Replace with `mx.fast.scaled_dot_product_attention`:

```python
y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
```

This is a fused Metal kernel that:
- Never materializes the full attention matrix
- Tiles the computation in SRAM
- Fuses the scale → mask → softmax → matmul pipeline
- Reduces memory bandwidth by O(T) factor

### Challenges

1. **Joint attention masking:** `_joint_attention` slices the score matrix into spans (self, text, speaker) and applies different masks per span. `mx.fast.scaled_dot_product_attention` accepts a single mask tensor. Need to pre-build a combined `(B, 1, T_q, T_kv)` additive mask that encodes all spans. **Important:** With 3× CFG batching, the masks differ per batch element — text is zeroed in batch[1], speaker is zeroed in batch[2]. The mask must be `(B, 1, T_q, T_kv)`, not `(1, 1, T_q, T_kv)`.

2. **Encoder self-attention:** `_self_attention` has optional causal masking and key masking. Can pass `mask="causal"` string for the pure-causal case; combined causal+key requires a pre-built additive mask.

3. **Gate mechanism:** Post-attention `y = y * sigmoid(gate(x))` is outside the fused kernel. This is fine — it's a cheap element-wise op.

4. **RoPE interaction:** RoPE is applied to Q and K before attention. The fused SDPA doesn't include RoPE — keep the existing RoPE application, feed rotated Q/K into SDPA.

5. **Mask caching:** The joint attention mask is identical across all 24 DiT blocks within a diffusion step (same spans, same text/speaker masks). Build the mask **once per step** in the `forward` method and pass it to each `forward_step` call. Currently each block recomputes the mask implicitly via score slicing.

### Implementation Plan

```python
def _self_attention(self, x, *, prefix, causal, key_mask, query_mask, half_rotary):
    # ... existing Q/K/V projection, reshape, norm, RoPE ...
    
    # Build mask
    if causal and key_mask is not None:
        mask = self._build_combined_mask(t, key_mask, causal=True)
    elif causal:
        mask = "causal"  # mx.fast.sdpa accepts this string
    elif key_mask is not None:
        mask = self._build_key_mask(t, key_mask)
    else:
        mask = None
    
    scale = 1.0 / math.sqrt(self.head_dim)
    y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    y = _merge_heads(y, mx)
    # ... gate, output projection ...

def _joint_attention(self, x, *, layer_idx, kv_text, kv_speaker, text_mask, speaker_mask):
    # ... existing Q/K/V projection, reshape, norm, RoPE, KV concat ...
    
    # Pre-build combined mask: (B, 1, T_q, T_kv_total)
    mask = self._build_joint_mask(t, spans, text_mask, speaker_mask)
    
    scale = 1.0 / math.sqrt(self.head_dim)
    y = mx.fast.scaled_dot_product_attention(q, k_full, v_full, scale=scale, mask=mask)
    y = _merge_heads(y, mx)
    # ... gate, output projection ...
```

### Mask Builder

```python
def _build_joint_mask(self, b, t_q, spans, text_mask, speaker_mask):
    """Build additive attention mask for joint attention.
    
    Returns: (B, 1, T_q, T_kv) float16 tensor where 0.0 = attend, -inf = ignore.
    
    With 3× CFG batch, masks differ per batch element:
      batch[0] = conditioned (text + speaker active)
      batch[1] = uncond text (text zeroed)
      batch[2] = uncond speaker (speaker zeroed)
    The text_mask and speaker_mask are already (B, T) with this CFG
    replication applied upstream by the sampler.
    """
    mx = self.mx
    parts = []
    for name, length in spans:
        if name == "self":
            parts.append(mx.zeros((b, 1, t_q, length), dtype=mx.float16))
        elif name == "text" and text_mask is not None:
            # text_mask: (B, T_text) → (B, 1, 1, T_text) → broadcast to (B, 1, T_q, T_text)
            inv = (1.0 - text_mask.astype(mx.float16)) * NEG_INF
            part = mx.reshape(inv, (b, 1, 1, length))
            parts.append(mx.broadcast_to(part, (b, 1, t_q, length)))
        elif name == "speaker" and speaker_mask is not None:
            inv = (1.0 - speaker_mask.astype(mx.float16)) * NEG_INF
            part = mx.reshape(inv, (b, 1, 1, length))
            parts.append(mx.broadcast_to(part, (b, 1, t_q, length)))
        else:
            parts.append(mx.zeros((b, 1, t_q, length), dtype=mx.float16))
    return mx.concatenate(parts, axis=-1)
```

> **Note:** This mask is invariant across blocks within a step. Build it once in `forward()` and pass it to all 24 `forward_step()` calls.

### Parity Gate

- `quantize=none` with Flash Attention must produce `allclose(atol=1e-5, rtol=1e-4)` output vs manual attention (not bitwise — fused kernels have different accumulation order)
- Run all 144 existing tests
- Tier 1 `bench_dit_forward_single` and `bench_dit_forward_cfg` must show improvement

---

## Optimization 2: `mx.fast.rms_norm` and `mx.fast.rope`

**Impact estimate:** 5–15% speedup (norm and RoPE are per-layer overhead)

### Current Implementation

`_rms_norm`, `_rms_norm_head`, `_build_rope`, `_apply_rotary`, `_apply_half_rotary` are all manual Python/MLX ops:

```python
def _rms_norm(x, weight, mx, eps):
    orig_dtype = x.dtype
    x = x.astype(mx.float32)
    norm = 1.0 / mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x * norm).astype(orig_dtype) * weight
```

### Target Implementation

```python
def _rms_norm(x, weight, mx, eps):
    return mx.fast.rms_norm(x, weight, eps)

# RoPE: mx.fast.rope handles cos/sin generation + application in one fused call
def _apply_rope(x, dims, offset=0, traditional=False):
    return mx.fast.rope(x, dims=dims, offset=offset, traditional=traditional)
```

### Challenges

1. **Half-rotary RoPE is head-split, not dim-split:** `_apply_half_rotary` rotates the **first half of heads** (axis=1), leaving the second half unrotated. This is NOT the same as `mx.fast.rope(dims=head_dim//2)`, which would rotate half the dimensions of ALL heads. The correct approach:
   ```python
   # Split on head axis, apply full rope to first half, concat
   h_half = num_heads // 2
   q_rot = mx.fast.rope(q[:, :h_half, :, :], dims=head_dim,
                         traditional=True, base=10000.0, scale=1.0, offset=0)
   q = mx.concatenate([q_rot, q[:, h_half:, :, :]], axis=1)
   ```
   This still requires a split+concat, but the rotation itself uses the fused kernel.

2. **RoPE style is traditional:** Current code uses `x[..., ::2]` / `x[..., 1::2]` (consecutive pair rotation), which corresponds to `traditional=True` in `mx.fast.rope`.

3. **RoPE frequency recomputation:** `_build_rope` currently recomputes cos/sin from numpy on every attention call. `mx.fast.rope` eliminates this entirely — it computes frequencies internally from `base` and `dims`. This removes redundant numpy→MLX conversions.

4. **Per-head RMS norm:** `_rms_norm_head` operates on `(B, H, T, D)` with per-head weights `(H, D)`. `mx.fast.rms_norm` expects `(*, D)` with weight `(D,)` — incompatible shapes. Options:
   - Reshape to `(B*H, T, D)` with `H` separate norm calls (overhead may exceed benefit)
   - Keep manual implementation (it's not a bottleneck — called 2× per encoder block, encoders are <5% of wall time)
   - **Recommendation:** Keep `_rms_norm_head` manual.

5. **float32 upcast:** `mx.fast.rms_norm` docs state "softmax is performed in float32" for SDPA but don't explicitly state rms_norm precision. Need to verify experimentally that `mx.fast.rms_norm` on float16 input matches our manual float32-upcast implementation to within tolerance.

### Implementation Plan

- Replace `_rms_norm` with `mx.fast.rms_norm` (drop-in for the `(B, T, D)` case used in DiT blocks)
- Replace `_build_rope` + `_apply_rotary` with `mx.fast.rope(traditional=True, base=10000.0)` for full-rotary case (encoder self-attention)
- For `_apply_half_rotary`, use head-split + `mx.fast.rope` + concat (as shown above)
- Keep `_rms_norm_head` manual (per-head norm in encoders, not a bottleneck)

### Parity Gate

- `allclose(atol=1e-5, rtol=1e-4)` vs current implementation
- All 144 tests pass

---

## Optimization 3: `mx.compile` Graph Fusion

**Impact estimate:** 10–30% speedup from reduced kernel launch overhead

### What to Compile

The DiT forward pass launches hundreds of small Metal kernels per step. `mx.compile` traces the computation graph and fuses compatible ops into fewer, larger kernels.

**Primary targets:**

1. **`forward_step`** — The per-block DiT computation (AdaLN → attention → AdaLN → MLP). This is the innermost hot loop, called 24 times per diffusion step.

2. **`_swiglu_paths`** — Three linear ops + SiLU + multiply + linear. Currently 7+ kernel launches; should fuse to 1–2.

3. **`_lowrank_adaln`** — SiLU → down-project → up-project → add. 4+ kernel launches per call, called 2× per block.

### Implementation

```python
@mx.compile
def _compiled_dit_block(self, x, t_emb, cond, *, layer_idx, kv_text, kv_speaker, text_mask, speaker_mask):
    """Single DiT block: AdaLN → joint attention → AdaLN → SwiGLU MLP."""
    # ... existing forward_step body ...
```

### Challenges

1. **Dynamic shapes:** `mx.compile` works best with fixed shapes. The first call traces and compiles; subsequent calls with the same shapes reuse the compiled graph. Different sequence lengths trigger recompilation. During generation, sequence length is fixed (set by `--seq-length`), so all 32 steps × 24 blocks reuse the same trace. Only different runs with different `seq_length` trigger recompilation.

2. **Control flow:** `mx.compile` doesn't support Python control flow that depends on tensor values. Our code has `if cfg_active:` which depends on a Python float — this is fine (resolved before tracing). But any tensor-dependent branching inside the compiled region will fail.

3. **Module tree access:** Compiled functions capture `self.tree` references during tracing. Since weights don't change during inference, this is safe. The `_resolve_path` calls happen during tracing and are baked into the compiled graph.

4. **Quantized linear layers:** When quantized, `nn.QuantizedLinear.__call__` internally dequantizes + matmuls. `mx.compile` should trace through this since it's all MLX ops, but verify that:
   - The compiled graph correctly handles the quantized weight format
   - No performance regression vs uncompiled quantized path (dequantization may not fuse well)

5. **SwiGLU float32 upcast:** `_swiglu` currently upcasts to float32 to avoid fp16 overflow in the 5888-dim intermediate space. Any compiled version must preserve this upcast. Verify that `mx.compile` doesn't optimize away the dtype conversions.

### Implementation Plan

- Start with `@mx.compile` on `_swiglu_paths` (simplest, no control flow, fixed shapes)
- Profile, verify speedup for both `quantize=none` and `quantize=8bit`
- Extend to `forward_step` if shapes are stable — **but exclude `_joint_attention`** initially (complex mask logic)
- Do NOT compile the outer sampler loop (has Python control flow)
- If quantized compile shows regression, gate compilation on `self.quantize_mode == "none"`

### Parity Gate

- `allclose(atol=1e-5, rtol=1e-4)` — compilation may reorder floating-point ops
- All 144 tests pass

---

## Optimization 4: Reduced Diffusion Steps

**Impact estimate:** 2× speedup (16 steps) or 4× (8 steps)

### Current Behavior

Default: 32 steps. Tier 2 benchmarks confirmed linear scaling (exponent 1.006) — halving steps halves diffusion time exactly.

### Implementation

Add recommended presets to the CLI and pipeline:

```python
QUALITY_PRESETS = {
    "quality": {"num_steps": 32, "truncation_factor": 0.8},   # Current default
    "balanced": {"num_steps": 16, "truncation_factor": 0.7},  # 2× faster
    "fast": {"num_steps": 8, "truncation_factor": 0.6},       # 4× faster
    "draft": {"num_steps": 4, "truncation_factor": 0.5},      # 8× faster, low quality
}
```

CLI:
```bash
echo-tts-mlx generate --text "Hello" --preset balanced --output out.wav
```

### Quality Impact

Step reduction is a quality/speed trade-off. No retraining needed — Euler sampling gracefully degrades with fewer steps. Expected behavior:
- **16 steps:** Slightly reduced detail in prosody, otherwise very close to 32-step
- **8 steps:** Noticeable smoothing, still intelligible and speaker-preserving
- **4 steps:** Useful for quick previews only

### CLI Changes

```bash
# New --preset flag (sugar for --steps + --truncation-factor)
echo-tts-mlx generate --text "Hello" --preset balanced --output out.wav

# Explicit steps still works (--preset overrides if both given)
echo-tts-mlx generate --text "Hello" --steps 16 --output out.wav
```

The preset dict lives in `pipeline.py` and is resolved in `cli.py` before calling `generate()`. The benchmark runner already supports `--steps` — no changes needed there.

### Validation

- Generate at each preset, compute MCD vs 32-step baseline
- Tier 3 cases at each preset
- Document quality/speed trade-off table:

| Preset | Steps | Est. Time (max, 8-bit) | Quality |
|---|---|---|---|
| quality | 32 | 66.9s (baseline) | Best |
| balanced | 16 | ~33s | Minor prosody loss |
| fast | 8 | ~17s | Noticeable smoothing |
| draft | 4 | ~8s | Preview only |

> **Note:** These time estimates assume no other optimizations. With Flash Attention + mx.fast ops, "balanced" would be ~8–10s.

---

## Optimization 5: Streaming DAC Decode

**Impact estimate:** Eliminates decode latency from critical path (8–15% of wall time)

### Current Behavior

DAC decode runs after all diffusion steps complete. For max-length (seq=640), DAC decode takes ~1.7s.

### Target

Overlap DAC decode with the final diffusion steps using pipeline parallelism:

```
Step 30: [DiT forward ~~~~~~~~]
Step 31: [DiT forward ~~~~~~~~]  [DAC decode chunk 0-160 ~~~]
Step 32: [DiT forward ~~~~~~~~]  [DAC decode chunk 160-320 ~]  [DAC decode chunk 0-160 → audio out]
Done:                             [DAC decode chunk 320-640 ~]  [remaining chunks → audio out]
```

### Challenges

1. **DAC expects full latent sequence:** The Fish S1-DAC operates on the complete `(1, 80, T)` tensor. Chunked decode may produce discontinuities at chunk boundaries without overlap-add.

2. **MLX lazy evaluation:** MLX's lazy eval model means we'd need explicit `mx.eval()` calls to force compute ordering. The current `eval_step` callback in the sampler could be extended.

3. **Complexity vs payoff:** 1.7s savings on a 67s pipeline is ~2.5%. May not be worth the implementation complexity.

### Recommendation

**Defer.** The payoff is small relative to Optimizations 1–4. Revisit if the other optimizations bring total time down enough that DAC decode becomes a larger percentage.

---

## Implementation Order

| Priority | Optimization | Est. Speedup | Effort | Dependencies |
|---|---|---|---|---|
| **P0** | Flash Attention | 1.5–2× | Medium (1–2 days) | None |
| **P1** | `mx.fast.rms_norm` + `rope` | 1.05–1.15× | Low (half day) | None |
| **P1** | Quality presets (step reduction) | 2× at "balanced" | Low (half day) | None |
| **P2** | `mx.compile` graph fusion | 1.1–1.3× | Medium (1 day) | P0 (compile the optimized code) |
| **P3** | Streaming DAC decode | 1.02–1.03× | High (2 days) | Defer |

**Combined estimate (P0 + P1 + P2 at 16 steps):**
- Multiplicative: 1.75× × 1.1× × 2× × 1.2× = **~4.6×**
- Max-length: 66.9s → **~14.5s** (RT factor ~1.3×)
- Medium: 31.9s → **~6.9s** (RT factor ~2.0×)
- Short: 10.1s → **~2.2s** (RT factor ~0.4×)

> **Caveat:** Speedups are not perfectly multiplicative. Flash Attention and `mx.compile` both reduce memory bandwidth pressure — once Flash Attention eliminates the attention bottleneck, `mx.compile` may see diminished returns on the attention path (though it still helps MLP/AdaLN). Conservative estimate: **3–4× combined** rather than 4.6×. Step reduction is the only optimization that compounds independently (it's purely algorithmic).

---

## Pre-Optimization Baseline

Before any performance-optimization code changes, capture reference outputs for regression testing:

```bash
# Save deterministic latent output at each precision for parity checks
python -c "
import mlx.core as mx
from echo_tts_mlx.pipeline import EchoTTS
import numpy as np

for mode in ['none', '8bit']:
    model = EchoTTS.from_pretrained('weights/converted', dtype='float16', quantize=mode)
    latents = model.generate_latents(text='[S1] Hello world.', num_steps=4, seed=42, sequence_length=16)
    mx.eval(latents)
    np.save(f'logs/baseline_latents_{mode}.npy', np.array(latents))
    print(f'{mode}: saved {np.array(latents).shape}')
"
```

These `.npy` files serve as the regression reference for parity gates.

## Benchmark Plan

Each optimization gets its own benchmark comparison:

```bash
# Baseline (current, run once before any changes)
python -m benchmarks.run_benchmarks --tier 1 --quantize 8bit --output logs/bench_perf_baseline.json
python -m benchmarks.run_benchmarks --tier 3 --quantize 8bit --output logs/bench_perf_baseline_tier3.json

# After Flash Attention (Opt 1)
python -m benchmarks.run_benchmarks --tier 1 --quantize 8bit --output logs/bench_opt1_flash.json
python -m benchmarks.compare logs/bench_perf_baseline.json logs/bench_opt1_flash.json

# After mx.fast.rms_norm + rope (Opt 2)
python -m benchmarks.run_benchmarks --tier 1 --quantize 8bit --output logs/bench_opt2_fast_ops.json
python -m benchmarks.compare logs/bench_opt1_flash.json logs/bench_opt2_fast_ops.json

# After mx.compile (Opt 3)
python -m benchmarks.run_benchmarks --tier 1 --quantize 8bit --output logs/bench_opt3_compile.json

# Full end-to-end with presets (Opt 4)
python -m benchmarks.run_benchmarks --tier 3 --quantize 8bit --output logs/bench_final_32step.json
python -m benchmarks.run_benchmarks --tier 3 --quantize 8bit --steps 16 --output logs/bench_final_16step.json
python -m benchmarks.compare logs/bench_perf_baseline_tier3.json logs/bench_final_32step.json
```

---

## Parity & Quality Gates

| Gate | Criterion |
|---|---|
| Numerical parity (no-quantize) | `allclose(atol=1e-5, rtol=1e-4)` vs pre-optimization baseline |
| Numerical parity (8-bit) | `allclose(atol=1e-4, rtol=1e-3)` vs pre-optimization baseline |
| Test suite | All 144+ tests pass |
| Tier 1 regression | No component slower than baseline (within noise margin) |
| Quality (presets) | MCD table published for each preset vs 32-step |

---

## Non-Goals

- **Retraining / distillation:** Requires GPU cluster time and training data pipeline. Out of scope.
- **Custom Metal kernels:** Writing raw Metal shaders is high-effort and MLX's fused ops cover the main cases.
- **Neural Engine (ANE):** MLX doesn't support ANE dispatch. The GPU path is the only option.
- **Multi-device / distributed:** Single-device only.
