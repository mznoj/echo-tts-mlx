# Blockwise Generation — Benchmark Spec

> **Goal:** Quantify blockwise generation performance across block configurations, measure TTFB improvements vs standard mode, and establish regression baselines for blockwise-specific overhead.
>
> **Prerequisite:** Converted weights with `--include-blockwise` (adds ~788 MB at f16, ~394 MB at 8-bit).

---

## 1. Overview

Blockwise generation trades total throughput for lower time-to-first-audio (TTFB). Each block runs its own full diffusion loop, so `k` blocks ≈ `k×` the diffusion cost of standard mode for the same total output length. The benchmarks below quantify this tradeoff precisely and track:

1. **TTFB** — time from invocation to first decoded audio block
2. **Total wall time** — end-to-end including all blocks + decode
3. **Per-block breakdown** — diffusion + latent KV recomputation per block
4. **Overhead vs standard** — blockwise-specific costs (latent encoder, KV recomputation, RoPE offset)
5. **Block configuration scaling** — how block count and size affect throughput and TTFB
6. **Peak memory** — additional GPU memory pressure from blockwise modules and KV caches
7. **Standard mode regression** — verify loading blockwise weights doesn't slow down non-blockwise generation

All benchmarks require blockwise-capable weights. If weights lack blockwise modules, the benchmark runner emits explicit skipped entries (`{"skipped": "no blockwise weights"}`) for each blockwise benchmark key rather than omitting them. This makes it obvious in JSON output and the compare script that blockwise was expected but couldn't run — silent key omission is harder to debug.

---

## 2. New Benchmarks

### 2.1 Tier 1 — Component Microbenchmarks

Add to the existing Tier 1 suite in `run_benchmarks.py`:

| Benchmark | What It Measures |
|---|---|
| `bench_latent_encode` | Latent encoder forward pass: `(1, T, 80)` → `(1, T/4, 1280)`. Measures the 14-layer causal transformer. |
| `bench_kv_cache_latent` | Full `get_kv_cache_latent()`: latent encode + norm + per-layer K/V projection with dilated RoPE. |
| `bench_dit_forward_cfg_blockwise` | Single DiT CFG forward pass (3× batch) with latent KV injected + `start_pos` offset. Measures the 4-way attention overhead vs standard 3-way. |

**Fixture setup:**
```python
# Latent encode input: simulate 128 frames of previously generated latent
prefix_latent = _make_latents((1, 128, int(config.latent_size)), seed=20)

# KV cache latent: same input, full pipeline
kv_latent = model.get_kv_cache_latent(prefix_latent)

# CFG forward with latent KV: block 1 at start_pos=128
latents_cfg = _make_latents((3, 128, int(config.latent_size)), seed=21)
kv_latent_full = _repeat_kv(pipeline, kv_latent, repeats=3)
```

**Expected results (estimates, M4 16GB, f16):**

| Benchmark | Estimated Latency |
|---|---|
| `bench_latent_encode` | 30–50 ms (similar to speaker encode — same architecture) |
| `bench_kv_cache_latent` | 40–70 ms (encode + 24 layer projections + custom RoPE) |
| `bench_dit_forward_cfg_blockwise` | 850–950 ms (3-way CFG: ~806 ms baseline + latent KV concat overhead) |

### 2.2 Tier 2 — Blockwise Pipeline Benchmarks

New benchmark group: `bench_blockwise_*`. Uses the same `_benchmark_measure` / warmup / cooldown infrastructure as existing Tier 2.

#### 2.2.1 `bench_blockwise_breakdown`

End-to-end blockwise generation with per-block timing breakdown.

**Configurations (run each):**

| Config ID | Block Sizes | Total Frames | Steps | Speaker |
|---|---|---|---|---|
| `bw_single` | `[320]` | 320 | 32 | 10s synthetic |
| `bw_2block` | `[160, 160]` | 320 | 32 | 10s synthetic |
| `bw_3block` | `[128, 128, 64]` | 320 | 32 | 10s synthetic |
| `bw_streaming` | `[64, 128, 128]` | 320 | 32 | 10s synthetic |
| `bw_max_2block` | `[320, 320]` | 640 | 32 | 10s synthetic |

**Metrics collected per config:**

| Metric | Description |
|---|---|
| `wall_time_s` | Total end-to-end time (text/speaker encode + all blocks + final decode) |
| `ttfb_s` | Time to first audio: text/speaker encode + block 0 diffusion + block 0 decode |
| `diffusion_total_s` | Sum of all blocks' diffusion time |
| `latent_kv_total_s` | Sum of all `get_kv_cache_latent()` calls across blocks |
| `decode_s` | Final DAC decode time |
| `per_block_s[]` | Array of per-block wall times (diffusion + latent KV for that block) |
| `audio_duration_s` | Output audio length in seconds |
| `realtime_factor` | `audio_duration_s / wall_time_s` |
| `overhead_vs_standard` | `wall_time_s / standard_wall_time_s` for same total frames and steps |

**Implementation approach:**

Use `on_block_complete` callback + `progress_callback` to capture per-block timing. Wrap `generate_blockwise()` with timing instrumentation:

```python
def _blockwise_breakdown(runtime, *, block_sizes, num_steps, use_speaker):
    mx = runtime.mx
    pipeline = runtime.pipeline
    block_times = []
    block_start = [None]
    ttfb_diffusion = [float('nan')]
    ttfb_audio = [float('nan')]
    t_origin = [None]
    current_block = [0]
    block0_last_step_done = [False]
    
    def on_progress(step, total_steps, t, cfg_active):
        """Fires after each Euler step. Capture diffusion-complete for block 0."""
        if current_block[0] == 0 and step == total_steps and not block0_last_step_done[0]:
            ttfb_diffusion[0] = time.perf_counter() - t_origin[0]
            block0_last_step_done[0] = True
    
    def on_block(block_idx, total, block_audio):
        """Fires after block is denoised AND decoded (pipeline decodes before callback)."""
        now = time.perf_counter()
        block_times.append(now - block_start[0])
        
        if block_idx == 0:
            # block_audio is already decoded — this IS the audio-ready timestamp
            ttfb_audio[0] = now - t_origin[0]
        
        current_block[0] = block_idx + 1
        block_start[0] = time.perf_counter()
    
    t_origin[0] = time.perf_counter()
    # ... text/speaker conditioning setup (counted in TTFB) ...
    
    block_start[0] = time.perf_counter()
    t_gen = time.perf_counter()
    audio = pipeline.generate_blockwise(
        text=TEXT,
        block_sizes=block_sizes,
        speaker_audio=speaker_audio,
        num_steps=num_steps,
        seed=SEED,
        on_block_complete=on_block,
        progress_callback=on_progress,
    )
    mx.eval(audio)
    generation_s = time.perf_counter() - t_gen
    conditioning_s = t_gen - t_origin[0]
    
    return {
        'wall_time_s': conditioning_s + generation_s,
        'ttfb_diffusion_s': ttfb_diffusion[0],
        'ttfb_audio_s': ttfb_audio[0],
        'per_block_s': block_times,
        'conditioning_s': conditioning_s,
        ...
    }
```

**Note on timing accuracy:** Since `on_block_complete` receives already-decoded audio, no extra decode pass is performed inside the callback. The `wall_time_s` metric is clean — callback overhead is limited to the `time.perf_counter()` calls and list appends (negligible). The `progress_callback` for `ttfb_diffusion_s` fires at the boundary between diffusion-complete and decode-start, giving a precise split.

#### 2.2.2 `bench_blockwise_vs_standard`

Direct A/B comparison: same total output length, same steps, same seed. Standard mode vs blockwise with various block counts.

**Sweep:**

| Total Frames | Standard | 1 Block | 2 Blocks | 3 Blocks | 4 Blocks |
|---|---|---|---|---|---|
| 320 | `[320]` std | `[320]` bw | `[160, 160]` | `[128, 96, 96]` | `[80, 80, 80, 80]` |
| 640 | `[640]` std | `[640]` bw | `[320, 320]` | `[256, 192, 192]` | `[160, 160, 160, 160]` |

**Metrics:**

| Metric | Description |
|---|---|
| `standard_wall_s` | Standard mode total time |
| `blockwise_wall_s` | Blockwise mode total time |
| `overhead_ratio` | `blockwise_wall_s / standard_wall_s` |
| `standard_ttfb_s` | Standard TTFB (= full generation, no streaming) |
| `blockwise_ttfb_s` | Blockwise TTFB (first block only) |
| `ttfb_speedup` | `standard_ttfb_s / blockwise_ttfb_s` |
| `peak_memory_mb` | Peak GPU memory during generation (via `SyncAdapter.get_memory_metrics()`) |

This benchmark directly validates the theoretical ~k× overhead and TTFB improvement claims from the blockwise spec.

**Theoretical overhead validation:** The naive theoretical model is "k blocks ≈ k× total cost," but this ignores fixed costs (conditioning and decode happen once regardless of block count). The adjusted formula uses measured values from the standard run:

```
diffusion_per_block = standard_diffusion_s / 1  # standard is 1-block equivalent
expected_wall_s = (num_blocks * diffusion_per_block) + conditioning_s + decode_s
expected_overhead = expected_wall_s / standard_wall_s
```

Where `conditioning_s` and `decode_s` come from the standard run's breakdown. This produces a tighter expected ratio (e.g., for 3 blocks with conditioning=0.1s and decode=1.2s out of standard_wall=19.5s, the adjusted expected is ~2.87× vs naive 3.0×).

Log a warning in the benchmark output if the measured overhead deviates by more than 20% from the **adjusted** prediction — this would indicate unexpected overhead sources (e.g., latent KV recomputation cost being larger than expected).

#### 2.2.6 `bench_blockwise_standard_regression`

**Purpose:** Verify that loading blockwise-capable weights does NOT regress standard (non-blockwise) generation performance.

Run standard `generate()` (not `generate_blockwise()`) with blockwise weights loaded. Compare against the same generation with standard (pruned) weights.

| Config | Weights | Mode | Frames | Steps |
|---|---|---|---|---|
| `std_pruned` | (derived — see below) | standard | 320 | 32 |
| `std_blockwise_weights` | (from `--weights`) | standard | 320 | 32 |

**Weight directory resolution:** Derive the counterpart directory from `--weights` rather than hardcoding paths:
- If `--weights` points to blockwise weights (has blockwise modules): look for a sibling `converted/` directory at the same level (e.g., `--weights weights/converted-blockwise/` → try `weights/converted/`).
- Optionally accept `--weights-standard <path>` to specify the pruned weights explicitly.
- If the counterpart directory doesn't exist or isn't found: **skip** with `{"skipped": "pruned weights not found at <derived_path>"}`. Don't fail the benchmark run.

**Gate:** `std_blockwise_weights.wall_time_s` must be within 5% of `std_pruned.wall_time_s`. The extra blockwise modules should not participate in standard forward passes (they're only activated when `kv_latent` is provided).

> **Note:** This requires loading two different weight sets in one benchmark run. Implementation: load pruned weights first, benchmark, then load blockwise weights, benchmark again. The model load cost difference is already captured by the existing `bench_model_load`.

#### 2.2.3 `bench_blockwise_scale_blocks`

Fixed total output (320 frames, 32 steps), sweep block count from 1 to 5.

All block sizes are divisible by `speaker_patch_size` (4) to avoid zero-padding artifacts.

| Blocks | Block Sizes |
|---|---|
| 1 | `[320]` |
| 2 | `[160, 160]` |
| 3 | `[108, 108, 104]` |
| 4 | `[80, 80, 80, 80]` |
| 5 | `[64, 64, 64, 64, 64]` |

**Output:** Wall time and TTFB at each block count. Fit power-law exponent for wall time vs block count (expected ~1.0 = linear scaling).

#### 2.2.4 `bench_blockwise_scale_first_block`

Fixed total output (320 frames, 32 steps, 2 blocks), sweep first block size to measure TTFB sensitivity.

| First Block | Second Block | Total | Expected TTFB |
|---|---|---|---|
| 32 | 288 | 320 | Fastest TTFB, ~1.5s audio |
| 64 | 256 | 320 | ~3s audio |
| 128 | 192 | 320 | ~6s audio |
| 160 | 160 | 320 | Equal split |
| 256 | 64 | 320 | Slowest TTFB |

**Output:** TTFB vs first block size. Validates the streaming latency curve.

#### 2.2.5 `bench_blockwise_continuation`

Measure overhead of continuation (pre-existing audio as prefix).

| Config | Continuation Frames | Block Sizes | Total New |
|---|---|---|---|
| `cont_none` | 0 | `[160, 160]` | 320 |
| `cont_short` | 64 | `[128, 128]` | 256 |
| `cont_medium` | 256 | `[192, 192]` | 384 |

**Metrics per config:**

| Metric | Description |
|---|---|
| `wall_time_s` | Total end-to-end |
| `continuation_encode_s` | Time to encode continuation audio → latents (DAC encode + PCA encode). Zero for `cont_none`. |
| `ttfb_diffusion_s` | Conditioning + continuation encoding + first block diffusion |
| `ttfb_audio_s` | Above + first block decode |
| `latent_kv_total_s` | Sum of `get_kv_cache_latent()` — larger prefix = more latent frames to encode each block |

**Continuation encoding pipeline:** `audio → DAC encode (zq) → PCA encode → latent frames`. This is measured via `pipeline.encode_continuation(audio=...)`. The cost scales with continuation audio length (~0.75s per 10s of audio, dominated by DAC encode).

### 2.3 Tier 3 — Cross-Implementation Standard Cases (Blockwise Variants)

Extend `cross_impl_protocol.py` with blockwise variants of the standard cases:

All block sizes must be divisible by `speaker_patch_size` (4). Where the base case `seq_length` isn't evenly divisible into patch-aligned blocks, the last block absorbs the remainder (and is itself kept divisible by 4 by rounding the total down).

| Case ID | Base Case | seq_length | Block Sizes | Steps | Speaker |
|---|---|---|---|---|---|
| `case_a_bw` | case_a | 100 | `[100]` | 32 | None |
| `case_b_bw` | case_b | 150 | `[76, 76]` | 32 | 5s |
| `case_c_bw` | case_c | 300 | `[128, 128, 44]` | 32 | 10s |
| `case_d_bw` | case_d | 640 | `[160, 160, 160, 160]` | 32 | 10s |
| `case_d_bw_stream` | case_d | 640 | `[64, 192, 192, 192]` | 32 | 10s |

> **Note on case_b_bw:** Original seq_length 150 split as `[76, 76]` = 152 frames. The extra 2 frames (beyond the text-aligned length) will be trimmed by `trim_latents=True` during decode, matching the standard case_b behavior. Alternative: `[76, 74]` = 150 exact, but 74 is not divisible by 4 — prefer slightly overshooting and trimming.

These use the same text, speaker, and seed as the standard cases, enabling direct wall time and quality comparison.

**Quality gates (same as standard Tier 3):**
- Determinism: same seed → `allclose(atol=1e-5, rtol=1e-4)`
- Non-silence: peak > 0.01
- Duration: > 0.5s

**Additional blockwise-specific gate:**
- `overhead_ratio` reported (no hard gate — this is informational since blockwise is expected to be slower)

---

## 3. Implementation Plan

### 3.1 Benchmark Runner Changes (`run_benchmarks.py`)

1. **Weight detection:** At runtime init, check `pipeline.model.has_blockwise_modules`. Set `runtime.blockwise_capable = True/False`.

2. **Tier 1 additions:**
   - Add `bench_latent_encode`, `bench_kv_cache_latent`, `bench_dit_forward_cfg_blockwise` to the `benches` OrderedDict.
   - Gate on `runtime.blockwise_capable` — emit `{"skipped": "no blockwise weights"}` for each benchmark key if unavailable (see §1 on skip policy).

3. **Tier 2 additions:**
   - Add new benchmark functions: `_run_blockwise_breakdown`, `_run_blockwise_vs_standard`, `_run_blockwise_scale_blocks`, `_run_blockwise_scale_first_block`, `_run_blockwise_continuation`.
   - Register in `_run_tier2` benchmarks OrderedDict.
   - Gate on `runtime.blockwise_capable`.

4. **Tier 3 additions:**
   - Add a **separate** `build_blockwise_cases()` function in `cross_impl_protocol.py` (do NOT modify `build_standard_cases()` — that would break the existing protocol for all implementations and the `validate_cross_impl_report()` validator).
   - Add `StandardCase.block_sizes: list[int] | None = None` field for blockwise cases.
   - Extend `AbstractBenchmarkRunner` with `run_case_blockwise()` method (optional, default raises `NotImplementedError`). Implementations that don't support blockwise skip these cases gracefully.
   - In `_MlxTier3Runner`, implement `run_case_blockwise()` calling `pipeline.generate_blockwise()`.
   - Blockwise Tier 3 results nest under `tier3_blockwise` (not `tier3`) to keep standard report validation intact. Keep `validate_cross_impl_report()` completely unchanged — if blockwise validation becomes needed later, add a separate `validate_blockwise_report()` function.

### 3.2 CLI Changes

```bash
# Run all tiers including blockwise
python benchmarks/run_benchmarks.py --weights weights/converted-blockwise/

# Run only blockwise benchmarks
python benchmarks/run_benchmarks.py --tier 2 --filter blockwise --weights weights/converted-blockwise/

# Run Tier 1 blockwise components
python benchmarks/run_benchmarks.py --tier 1 --filter latent --weights weights/converted-blockwise/

# Compare standard vs blockwise weights
python benchmarks/compare.py standard_results.json blockwise_results.json
```

No new CLI flags needed — blockwise benchmarks are auto-detected based on weight capability and can be filtered with `--filter blockwise`.

### 3.3 Compare Script Changes (`compare.py`)

Add blockwise-aware comparison:
- When both reports contain `bench_blockwise_vs_standard`, print the overhead ratio and TTFB speedup delta.
- Flag regressions where blockwise overhead increases by >10% between runs.

### 3.4 Output Schema

Blockwise results nest under the existing tier keys:

```json
{
  "tier1": {
    "bench_latent_encode": { "median_ms": 42.1, "std_ms": 1.2, "runs": 5, "warmup": 2 },
    "bench_kv_cache_latent": { "median_ms": 58.3, "std_ms": 2.1, "runs": 5, "warmup": 2 },
    "bench_dit_forward_cfg_blockwise": { "median_ms": 891.0, "std_ms": 15.3, "runs": 5, "warmup": 2 }
  },
  "tier2": {
    "bench_blockwise_breakdown": {
      "bw_single": {
        "wall_time_s": 22.1,
        "ttfb_diffusion_s": 21.8,
        "ttfb_audio_s": 22.1,
        "per_block_s": [21.8],
        "latent_kv_total_s": 0.06,
        "decode_s": 1.2,
        "peak_memory_mb": 5842.0,
        "audio_duration_s": 15.0,
        "realtime_factor": 0.68
      },
      "bw_3block": {
        "wall_time_s": 54.3,
        "ttfb_diffusion_s": 18.6,
        "ttfb_audio_s": 18.9,
        "per_block_s": [18.6, 18.2, 16.1],
        "latent_kv_total_s": 0.18,
        "decode_s": 1.2,
        "peak_memory_mb": 5850.0,
        "audio_duration_s": 15.0,
        "realtime_factor": 0.28
      }
    },
    "bench_blockwise_vs_standard": {
      "320_frames": {
        "standard_wall_s": 19.5,
        "standard_peak_memory_mb": 5100.0,
        "configs": {
          "1_block": { "wall_s": 19.8, "overhead_ratio": 1.02, "ttfb_audio_s": 19.8, "ttfb_speedup": 0.98, "peak_memory_mb": 5842.0 },
          "2_blocks": { "wall_s": 38.1, "overhead_ratio": 1.95, "ttfb_audio_s": 19.5, "ttfb_speedup": 1.0, "peak_memory_mb": 5845.0 },
          "3_blocks": { "wall_s": 53.2, "overhead_ratio": 2.73, "ttfb_audio_s": 12.5, "ttfb_speedup": 1.56, "peak_memory_mb": 5848.0 }
        },
        "theoretical_check": {
          "conditioning_s": 0.1,
          "decode_s": 1.2,
          "diffusion_per_block_s": 18.2,
          "2_blocks": { "expected_ratio": 1.94, "actual_ratio": 1.95, "deviation_pct": 0.5 },
          "3_blocks": { "expected_ratio": 2.87, "actual_ratio": 2.73, "deviation_pct": -4.9 }
        }
      }
    },
    "bench_blockwise_scale_blocks": {
      "points": [
        { "num_blocks": 1, "wall_time_s": 19.8, "ttfb_audio_s": 19.8 },
        { "num_blocks": 2, "wall_time_s": 38.1, "ttfb_audio_s": 19.5 }
      ],
      "wall_scaling_exponent": 0.97
    },
    "bench_blockwise_scale_first_block": {
      "points": [
        { "first_block_size": 32, "ttfb_audio_s": 4.2 },
        { "first_block_size": 64, "ttfb_audio_s": 7.8 }
      ]
    },
    "bench_blockwise_continuation": { "..." : "..." },
    "bench_blockwise_standard_regression": {
      "std_pruned_wall_s": 19.5,
      "std_blockwise_weights_wall_s": 19.6,
      "regression_pct": 0.5,
      "gate_passed": true
    }
  },
  "tier3_blockwise": {
    "case_a_bw": { "wall_time_s": 12.5, "realtime_factor": 0.07, "quality_ok": true, "status": "PASS" },
    "case_d_bw_stream": { "wall_time_s": 62.8, "ttfb_audio_s": 8.1, "realtime_factor": 0.30, "quality_ok": true, "status": "PASS" }
  }
}
```

**When blockwise modules are unavailable**, the output contains explicit skipped entries:

```json
{
  "tier1": {
    "bench_latent_encode": { "skipped": "no blockwise weights" },
    "bench_kv_cache_latent": { "skipped": "no blockwise weights" },
    "bench_dit_forward_cfg_blockwise": { "skipped": "no blockwise weights" }
  },
  "tier2": {
    "bench_blockwise_breakdown": { "skipped": "no blockwise weights" },
    "bench_blockwise_vs_standard": { "skipped": "no blockwise weights" }
  },
  "tier3_blockwise": { "skipped": "no blockwise weights" }
}
```

> **Note:** Blockwise Tier 3 results go in `tier3_blockwise` (not `tier3`) to preserve backward compatibility with `validate_cross_impl_report()` which expects exactly the standard case IDs. `validate_cross_impl_report()` is not modified — a separate `validate_blockwise_report()` can be added later if cross-implementation blockwise comparison becomes needed.

---

## 4. BENCHMARKS.md Additions

After running, update `docs/BENCHMARKS.md` with a new section:

```markdown
## Blockwise Generation

### Tier 1: Blockwise Component Microbenchmarks

| Component | f16 | 8-bit |
|---|---|---|
| latent_encode | TBD | TBD |
| kv_cache_latent | TBD | TBD |
| dit_forward_cfg_blockwise | TBD | TBD |

### Tier 2: Blockwise vs Standard

Total frames = 320, 32 steps, 10s speaker reference.

| Config | Wall Time | TTFB | Overhead vs Std | TTFB Speedup |
|---|---|---|---|---|
| Standard (baseline) | TBD | N/A | 1.0× | — |
| 1 block [320] | TBD | TBD | TBD | TBD |
| 2 blocks [160, 160] | TBD | TBD | TBD | TBD |
| 3 blocks [128, 128, 64] | TBD | TBD | TBD | TBD |

### Tier 2: TTFB vs First Block Size

2 blocks, total 320 frames, 32 steps.

| First Block | TTFB | First Audio Duration |
|---|---|---|
| 32 frames | TBD | ~1.5s |
| 64 frames | TBD | ~3.0s |
| 128 frames | TBD | ~6.0s |
| 256 frames | TBD | ~12.0s |

### Tier 3: Cross-Implementation (Blockwise)

| Case | Description | Wall Time | RTF | TTFB |
|---|---|---|---|---|
| A (bw) | Short, 1 block | TBD | TBD | TBD |
| B (bw) | Medium, 2 blocks | TBD | TBD | TBD |
| C (bw) | Long, 3 blocks | TBD | TBD | TBD |
| D (bw) | Max, 4 blocks | TBD | TBD | TBD |
| D (stream) | Max, streaming first block | TBD | TBD | TBD |
```

---

## 5. Methodology Notes

### Warmup & Measurement

Same protocol as existing benchmarks:
- **Tier 1:** 2 warmup + 5 measured runs, median reported
- **Tier 2:** Configurable warmup + 3 measured runs, 3s cooldown between configs
- **Tier 3:** 1 run for wall time, 2 runs for determinism check

### TTFB Measurement

We report **two** TTFB metrics to capture different use cases:

| Metric | Definition | Use Case |
|---|---|---|
| `ttfb_diffusion_s` | `conditioning_time + first_block_diffusion_time` | Latent-level TTFB: when first block latents are ready for decode |
| `ttfb_audio_s` | `conditioning_time + first_block_diffusion_time + first_block_decode_time` | True audio TTFB: when first playable audio samples are available |

**Why both?** The current `generate_blockwise()` (line 593) already decodes each block and passes **decoded audio** (not raw latents) to `on_block_complete`. This means the callback fires after both diffusion and DAC decode for that block are complete. We use this to our advantage:

**Measurement approach — no extra decode pass needed:**
1. `ttfb_diffusion_s`: Captured via `progress_callback` at the final step of block 0's diffusion loop. The `progress_callback(step, total_steps, t, cfg_active)` fires after each Euler step — when `step == num_steps` for the first block, that's the diffusion-complete timestamp.
2. `ttfb_audio_s`: Captured at the first `on_block_complete` invocation. Since the pipeline already decoded block 0's latents before calling the callback, this timestamp naturally includes diffusion + decode.

**Important:** Both callbacks must ensure `mx.eval()` has been called on the relevant tensors before reading `time.perf_counter()`. The pipeline's internal `eval_step` callback handles this for diffusion steps. For `on_block_complete`, the decoded audio is already evaluated by the pipeline before the callback fires.

For standard (non-blockwise) mode, both TTFBs equal the full wall time (no streaming possible). This makes the TTFB speedup metric meaningful: `standard_ttfb / blockwise_ttfb` shows how much faster the user hears first audio.

### Blockwise Weight Loading

Blockwise weights add ~788 MB (f16) or ~394 MB (8-bit). The `bench_model_load` Tier 1 benchmark already measures load time. When blockwise weights are present, load time will naturally be ~15-20% longer. This is captured automatically — no special handling needed.

### Peak Memory Tracking

Every Tier 2 blockwise benchmark calls `runtime.sync.reset_peak()` before and `runtime.sync.get_memory_metrics()` after generation. This captures `peak_memory_mb` and `active_memory_mb` — critical for understanding the memory cost of blockwise modules.

**Expected additional memory vs standard:**
- Blockwise weights: +788 MB (f16) / +394 MB (8-bit) — static, loaded at startup
- Latent KV cache per block: 24 layers × 2 (K+V) × `(B, H, T/4, D)` at float32. For T=320, H=16, D=128: ~15 MB. With 3× CFG batching: ~45 MB. This is recomputed per block (not accumulated).
- Prefix latent buffer: `(1, total_frames, 80)` at float32. For 640 frames: ~0.2 MB. Negligible.

### Latent KV Recomputation

A key cost in blockwise: the latent encoder re-processes the **full prefix buffer** (including zero-padded future blocks) at the start of each block. This means:
- Block 0: encode 320 frames of zeros → cheap (but still full encoder pass)
- Block 1: encode 320 frames (128 real + 192 zeros) → same cost
- Block 2: encode 320 frames (256 real + 64 zeros) → same cost

The cost per `get_kv_cache_latent()` call is constant regardless of how much of the prefix is filled. The benchmark captures this as `latent_kv_total_s` (sum across all blocks) and per-block in the breakdown.

### Step Count Interaction

All blockwise benchmarks default to 32 steps, matching the standard benchmark baseline. The `--steps` CLI flag applies to blockwise benchmarks too.

Step count is particularly impactful for blockwise because the cost multiplier is `num_blocks × num_steps × forward_pass_cost`. At 16 steps ("balanced" preset), a 3-block config would be ~3× the cost of standard 16-step — still faster absolute wall time than 3-block at 32 steps. A future follow-up could add a `bench_blockwise_scale_steps` sweep (block count × step count matrix), but this is deferred to keep the initial suite manageable.

### Thermal Throttling

Blockwise benchmarks run significantly more forward passes than standard benchmarks (k× per config). The existing 3s cooldown between benchmarks applies between configs within each benchmark function, but for the `bench_blockwise_breakdown` suite (5 configs × 3 runs each), total GPU-hot time is substantial.

**Mitigation:** Run benchmark configs in order of increasing computational cost (single block → 5 blocks). If later configs show unexpectedly high times, thermal throttling is the likely cause. The compare script should flag this pattern.

### Quality Validation

Blockwise benchmarks apply the same quality gates as standard:
- **Determinism:** Same seed, same block_sizes → identical output
- **Non-silence:** Peak amplitude > 0.01
- **Duration:** Output > 0.5s

No MCD comparison gate between blockwise and standard for the same text — they produce different audio (blockwise has different noise per block, different RoPE offsets). The Tier 3 blockwise cases only compare determinism against themselves.

---

## 6. Quantization Matrix

Run blockwise benchmarks at each supported quantization level:

| Quantization | Weight Size (with blockwise) | Run Priority |
|---|---|---|
| f16 | ~5,329 MB | P0 (baseline) |
| 8-bit | ~2,683 MB | P0 (production default) |
| mxfp4 | ~1,450 MB (est.) | P1 |
| mixed | ~2,100 MB (est.) | P1 |

The benchmark runner already accepts `--quantize`. Blockwise benchmarks inherit this — no special handling.

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Blockwise weights not converted | Benchmarks can't run | Auto-detect and skip with clear message |
| Latent KV recomputation dominates cost for many blocks | Overhead higher than theoretical k× | Track `latent_kv_total_s` separately; consider future incremental KV caching optimization |
| MLX runtime crash in blockwise path (untested codepath) | Benchmark failure | Run Tier 1 microbenchmarks first as smoke test before Tier 2/3 |
| `on_block_complete` callback timing accuracy | TTFB measurement noise | Use `time.perf_counter()` consistently; warmup eliminates JIT effects |

---

## Appendix: Amendment Log

**Review 1 (v1.0 → v1.1, 2026-03-03):** Self-review before implementation. 9 findings.

1. 🔴 **Tier 3 block sizes not divisible by 4:** `case_b_bw` had `[80, 70]` — 70 is not divisible by `speaker_patch_size` (4), causing zero-padding and non-deterministic behavior. Fixed to `[76, 76]` with a note about trim_latents handling the 2-frame overshoot.
2. 🔴 **TTFB definition ambiguous:** Original spec conflated "latents ready" with "audio ready." Added dual TTFB metrics (`ttfb_diffusion_s` and `ttfb_audio_s`) with explicit measurement approach including `mx.eval()` sync points.
3. 🔴 **Missing `mx.eval()` in callback timing:** Original pseudocode didn't call `mx.eval()` inside `on_block_complete`, so MLX lazy evaluation would defer computation and give wrong timing. Fixed with explicit eval barriers.
4. 🔴 **Tier 3 protocol backward compatibility:** Original plan added blockwise cases to `build_standard_cases()`, which would break `validate_cross_impl_report()` for all implementations. Changed to separate `build_blockwise_cases()` function and `tier3_blockwise` output key.
5. 🟠 **Missing standard-mode regression check:** Added `bench_blockwise_standard_regression` (§2.2.6) — verifies loading blockwise weights doesn't slow down standard generation path. 5% gate.
6. 🟠 **Missing peak memory tracking:** Added `peak_memory_mb` to all Tier 2 blockwise benchmarks and a methodology section on expected memory overhead (weights + KV cache + prefix buffer).
7. 🟠 **Missing theoretical overhead validation:** Added explicit check in `bench_blockwise_vs_standard` that compares measured overhead ratio against theoretical k× prediction, with 20% deviation warning.
8. 🟡 **Continuation encoding cost not detailed:** Expanded `bench_blockwise_continuation` metrics to include `continuation_encode_s` and describe the DAC+PCA encode pipeline.
9. 🟡 **Thermal throttling risk for long benchmark suites:** Added methodology note about running configs in increasing cost order and flagging suspicious timing patterns.

**Review 2 (v1.1 → v1.2, 2026-03-03):** Pre-implementation Q&A with implementer. 5 decisions.

1. 🟠 **`bench_blockwise_standard_regression` weight dirs (Q1):** Don't hardcode paths. Derive counterpart from `--weights` (look for sibling `converted/` dir). Accept optional `--weights-standard` override. Skip gracefully if counterpart not found.
2. 🔴 **`on_block_complete` passes decoded audio, not raw latents (Q2):** The spec's pseudocode assumed raw latents and added an extra decode pass in the callback — wrong per actual `generate_blockwise():593` implementation. Fixed: use `progress_callback` at block 0's final step for `ttfb_diffusion_s`, and `on_block_complete` (which already receives decoded audio) for `ttfb_audio_s`. No extra decode, no wall-time inflation.
3. 🟠 **Skipped entries when blockwise unavailable (Q3):** Emit explicit `{"skipped": "no blockwise weights"}` for every blockwise benchmark key. Added skip schema example to §3.4.
4. 🟠 **Theoretical overhead formula adjusted for fixed costs (Q4):** Naive `num_blocks` ratio ignores conditioning and decode (fixed costs). Updated formula uses measured `conditioning_s` and `decode_s` from the standard run to compute adjusted expected ratio. Updated schema example accordingly.
5. 🟡 **Tier 3 validator unchanged (Q5):** Confirmed: `validate_cross_impl_report()` stays as-is. Blockwise results under `tier3_blockwise`. Separate `validate_blockwise_report()` only if needed later.

---

_Specification version: 1.2_
_Date: 2026-03-03_
_Author: Larry 🦞_
