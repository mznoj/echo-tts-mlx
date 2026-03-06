# Benchmark Results

Current benchmark summary for Echo-TTS MLX on Mac mini (M4, 16GB).

## Environment

- Date: March 6, 2026
- Platform: macOS 15.7.3, Apple M4, 16GB unified memory
- Python: 3.14.2
- MLX: 0.31.0
- dtype: float16

## Validation Snapshot

- Test suite: 222 passed, 0 failed
- Coverage: 88% (2,450 statements, 290 uncovered)

## Tier 1: Component Microbenchmarks

Median latencies.

| Component | f16 | 8-bit | mxfp4 | mixed |
|---|---|---|---|---|
| model_load | 25,207ms | 30,392ms | 31,122ms | 34,172ms |
| text_encode | 37.3ms | 20.2ms | 21.0ms | 20.2ms |
| speaker_encode | 38.4ms | 22.0ms | 21.3ms | 21.1ms |
| kv_cache_text | 14.9ms | 9.1ms | 11.6ms | 13.8ms |
| kv_cache_speaker | 15.1ms | 12.3ms | 8.0ms | 8.0ms |
| dit_forward (single) | 377.6ms | 266.0ms | 254.9ms | 258.6ms |
| dit_forward (CFG) | 806.2ms | 710.2ms | 698.3ms | 696.5ms |
| pca_encode | 0.4ms | 0.5ms | 0.4ms | 0.4ms |
| pca_decode | 0.4ms | 0.4ms | 0.3ms | 0.3ms |
| dac_encode | 752.0ms | 725.6ms | 727.6ms | 722.5ms |
| dac_decode | 1,665ms | 1,680ms | 1,656ms | 1,677ms |
| latent_encode | 35.9ms | — | — | — |
| kv_cache_latent | 50.3ms | — | — | — |
| dit_forward (CFG blockwise) | 517.9ms | — | — | — |

## Tier 2: Diffusion Scaling

### Steps scaling (seq_length=200)

| Steps | f16 | 8-bit | mxfp4 | mixed |
|---|---|---|---|---|
| 8 | 4.7s | 4.1s | 3.8s | 4.0s |
| 16 | 9.4s | 8.3s | 7.8s | 7.9s |
| 32 | 18.6s | 16.2s | 15.7s | 15.8s |
| 64 | 37.5s | 32.9s | 31.6s | 31.7s |

### Sequence-length scaling (32 steps)

| Seq Length | f16 | 8-bit | mxfp4 | mixed |
|---|---|---|---|---|
| 100 | 11.3s | 8.9s | 8.6s | 8.7s |
| 200 | 18.8s | 16.5s | 15.9s | 15.8s |
| 400 | 33.9s | 32.3s | 31.7s | 31.8s |
| 640 | 51.8s | 51.6s | 50.7s | 50.1s |

## Tier 3: End-to-End Generation

| Case | Description | f16 | 8-bit | mxfp4 | mixed |
|---|---|---|---|---|---|
| A | Short unconditioned | 12.1s (RTF 0.08) | 9.8s (RTF 0.09) | 9.9s (RTF 0.09) | 9.3s (RTF 0.10) |
| B | Medium unconditioned | 19.5s (RTF 0.28) | 16.1s (RTF 0.34) | 16.3s (RTF 0.36) | 15.7s (RTF 0.44) |
| C | Long unconditioned | 31.4s (RTF 0.44) | 30.1s (RTF 0.45) | 29.3s (RTF 0.47) | 29.4s (RTF 0.47) |
| D | Long cloned | 61.3s (RTF 0.30) | 61.5s (RTF 0.30) | 60.2s (RTF 0.30) | 60.4s (RTF 0.30) |

All Tier 3 quality gates passed.

## Blockwise Generation

All blockwise benchmarks run on f16 with 32 steps and 10s speaker reference unless noted. Blockwise weights required (`--include-blockwise` during conversion).

### Tier 1: Blockwise Component Microbenchmarks

| Component | Median (ms) |
|---|---|
| Latent Encode | 35.9 |
| KV Cache (latent) | 50.3 |
| DiT Forward (CFG blockwise, 2×) | 517.9 |

Blockwise CFG forward is 34.9% faster than standard CFG (517.9ms vs 794.8ms) — blockwise runs 2 concurrent passes instead of 3.

### Tier 2: Blockwise vs Standard (320 frames)

| Config | Wall Time (s) | Overhead | TTFB Audio (s) | TTFB Speedup |
|---|---|---|---|---|
| Standard (baseline) | 31.7 | 1.0× | 31.7 (full) | — |
| 1 block [320] | 32.8 | 1.04× | 30.0 | 1.05× |
| 2 blocks [160, 160] | 36.6 | 1.15× | 18.2 | 1.74× |
| 3 blocks [128, 96, 96] | 39.7 | 1.25× | 15.3 | 2.07× |
| 4 blocks [80, 80, 80, 80] | 44.4 | 1.40× | 12.4 | 2.56× |

### Tier 2: TTFB vs First Block Size (2 blocks, 320 frames)

| First Block | TTFB Audio (s) | Approx First Audio |
|---|---|---|
| 32 frames | 10.8 | ~1.5s |
| 64 frames | 11.6 | ~3.0s |
| 128 frames | 15.1 | ~6.0s |
| 160 frames | 18.8 | ~7.5s |
| 256 frames | 24.6 | ~12.0s |

### Tier 2: Block Count Scaling (320 frames)

| Blocks | Wall Time (s) | TTFB Audio (s) |
|---|---|---|
| 1 | 31.8 | 29.2 |
| 2 | 36.6 | 18.2 |
| 3 | 41.9 | 15.5 |
| 4 | 44.9 | 13.0 |
| 5 | 47.9 | 11.2 |

Wall scaling exponent: 0.26 (sub-linear — overhead grows slower than block count).

### Tier 2: Continuation

| Config | Continuation | Wall Time (s) | Cont. Encode (s) | TTFB Audio (s) |
|---|---|---|---|---|
| No continuation | 0 frames | 36.7 | 0.0 | 18.2 |
| Short continuation | 64 frames | 33.0 | 2.3 | 17.7 |
| Medium continuation | 256 frames | 60.3 | — | — |

### Tier 2: Standard Regression

Loading blockwise weights does NOT regress standard generation: 32.4s (pruned) vs 32.4s (blockwise weights). Regression: −0.3% (within 5% gate). ✅

### Tier 3: Cross-Implementation (Blockwise)

| Case | Description | Wall Time (s) | RTF | Status |
|---|---|---|---|---|
| case_a_bw | Short, 1 block [100] | 25.5 | 0.036 | ✅ PASS |
| case_b_bw | Medium, 2 blocks [76, 76] | 24.3 | 0.272 | ✅ PASS |
| case_c_bw | Long, 3 blocks [128, 128, 44] | 41.5 | 0.336 | ✅ PASS |
| case_d_bw | Max, 4 blocks [160×4] | 84.6 | 0.219 | ✅ PASS |
| case_d_bw_stream | Max, streaming [64, 192×3] | 93.3 | 0.203 | ✅ PASS |

All blockwise cases pass quality gates (determinism, non-silence, duration).

## Mode Guidance

| Mode | Recommended Use |
|---|---|
| f16 | Quality baseline |
| 8-bit | Stable production default |
| mxfp4 | Fastest for short/medium diffusion |
| mixed | Best blend of speed and quality retention |

---

## Methodology

### Tiers

- **Tier 1 — Component microbenchmarks:** Isolate individual pipeline stages (text encode, DiT forward, DAC decode, etc). 2 warmup runs + 5 measured runs, median reported.
- **Tier 2 — Pipeline scaling:** End-to-end generation with step and sequence-length sweeps. 3 measured runs, 3s cooldown between benchmarks to avoid thermal throttling.
- **Tier 3 — Cross-implementation standard:** Fixed test cases (A–D) with standardized text, speaker refs, and parameters for comparing across backends (MLX, CUDA, MPS).

### MLX Timing

MLX uses lazy evaluation — operations build a computation graph but don't execute until `mx.eval()`. All timing measurements include an `mx.eval()` barrier to capture actual GPU execution time.

### Quality Gates

Every benchmark producing audio verifies: deterministic output (same seed = identical bytes), non-silence (peak > 0.01), and minimum duration (> 0.5s).

### Running Benchmarks

```bash
python benchmarks/run_benchmarks.py                    # All tiers
python benchmarks/run_benchmarks.py --tier 1           # Microbenchmarks only
python benchmarks/run_benchmarks.py --tier 3           # Cross-impl standard
python benchmarks/run_benchmarks.py --output results.json
python benchmarks/compare.py baseline.json current.json  # Compare two runs
```

## Raw Data

- Raw benchmark JSON files are in `logs/` (gitignored)
- Historical snapshots and reports are archived in `logs/`
