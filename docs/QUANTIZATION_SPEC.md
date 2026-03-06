# Experimental Improvements Spec

>
> **Scope:** Two workstreams addressing experimental items from the v1.0.0 README:
> 1. Tail pitch (F0) investigation & fix
> 2. Improved 4-bit quantization via MXFP4 and mixed-precision policies
>
> Each workstream is independent and can be tackled in either order.
>
> **Baseline:** echo-tts-mlx v1.0.0 (commit `f9b3846`)

---

## Workstream 1: Tail Pitch (F0) Instability Fix

### Problem

Generated audio exhibits F0 (fundamental frequency) instability in the final ~2 seconds — an unnatural pitch drift or wobble near the end of utterances. The artifact is also present in the upstream PyTorch implementation, confirming it's a model/sampling issue rather than an MLX port bug.

The upstream author notes that "scaling the initial noise by a factor of 0.8 or 0.9 reduces artifacts" and calls this "totally unprincipled and suggestive that something is off." Our codebase uses `truncation_factor=0.8` as the default, which partially mitigates but doesn't eliminate the issue.

**This workstream cannot fully solve the problem** — that would require retraining the model. Instead, it improves post-processing to trim before the instability becomes audible, and tunes truncation parameters with data rather than guesswork.

### Root Cause Analysis

The tail pitch instability likely stems from:

1. **Fixed sequence length vs variable content length:** The DiT generates a fixed 640-frame latent sequence, but actual speech content typically occupies fewer frames. The transition region between content and padding is under-constrained during training.

2. **Truncation factor as a bandaid:** Scaling initial noise by 0.8 compresses the output distribution. This incidentally stabilizes the tail but is a crude fix that may subtly degrade overall quality.

3. **Coarse trimming:** The current `find_flattening_point()` in `sampler.py` uses a simple sliding-window std/mean threshold to detect where latents go to zero. It trims at this boundary, but F0 instability occurs *before* the boundary — in the last frames of actual content, before the latents fully flatten.

### Implementation

#### Diagnostic Tooling

Script: `scripts/tail_pitch_analysis.py`

**Test matrix (all at seed=42, 8-bit quantized):**

| Case | Text | Seq Length | Speaker |
|---|---|---|---|
| short_uncond | "Hello, this is a test of Echo TTS." | 100 | None |
| medium_uncond | "The quick brown fox..." (longer text) | 200 | None |
| long_uncond | Multi-sentence paragraph | 400 | None |
| max_uncond | Multi-sentence paragraph | 640 | None |
| short_cloned | Short sentence | 150 | Matt's ref audio |
| medium_cloned | Medium text | 300 | Matt's ref audio |
| max_cloned | Multi-sentence paragraph | 640 | Matt's ref audio |

Repeat each case 3× with different seeds (42, 123, 7) = **21 samples total**.

For each sample:
1. Generate audio with current trimming (baseline)
2. Extract F0 contour using `librosa.pyin` (fmin=50, fmax=500 Hz)
3. Compute F0 variance in 2-second sliding windows (hop = 0.5s)
4. Compute **tail-to-body ratio**: F0 variance in last 2s ÷ median F0 variance in body
5. Record onset frame where F0 variance first exceeds 2× body median
6. Save per-sample: F0 contour plot, raw F0 data, onset frame, tail ratio

**Output:** `logs/tail_pitch_analysis/report.json` with per-case statistics + PNG plots.

**Performance note:** `librosa.pyin` on a 30s clip takes ~1–2s. This is diagnostic-only — not in the generation hot path.

#### Improved Tail Trimming

The improved trimmer lives in a new function in `sampler.py` alongside the existing `find_flattening_point`. It takes both latents and decoded audio and uses three signals:

```python
def find_content_boundary(
    latents: np.ndarray,     # (T, 80)  — latent frames
    audio: np.ndarray,       # (samples,) — decoded waveform
    sample_rate: int = 44100,
    ae_downsample_factor: int = 2048,
    *,
    # Signal 1: existing latent flattening
    latent_window: int = 20,
    latent_std_threshold: float = 0.05,
    latent_mean_threshold: float = 0.1,
    # Signal 2: audio energy drop
    energy_threshold_db: float = -40.0,
    energy_hop_samples: int = 2048,
    # Signal 3: F0 instability onset
    f0_enabled: bool = True,
    f0_variance_window_s: float = 2.0,
    f0_variance_ratio_threshold: float = 2.0,
) -> int:
    """Find content boundary using up to three signals.
    
    Returns the latent frame index where content ends, choosing the
    most conservative (earliest) boundary across enabled signals:
    
    1. Latent energy flattening (existing approach, always enabled)
    2. Audio RMS energy drop below threshold
    3. F0 variance spike exceeding body median × ratio threshold
    
    The F0 signal is optional (requires librosa) and off by default at
    runtime. Enabled via --trim-mode=f0 CLI flag or trim_mode='f0'.
    """
```

**Integration into `pipeline.py`:** The `decode_latents()` method currently:
1. PCA decode latents → z_q
2. DAC decode z_q → audio
3. Find flattening point on **latents only** → trim audio

Updated flow:
1. PCA decode latents → z_q
2. DAC decode z_q → audio (full sequence)
3. Find content boundary using **latents + audio** → trim audio

The caller chooses trim mode:
- `trim_mode="latent"` — current behavior (default, zero overhead)
- `trim_mode="energy"` — latent + audio energy (fast, no new deps)
- `trim_mode="f0"` — latent + audio energy + F0 (requires librosa, adds ~1–2s)

**CLI:**
```bash
echo-tts-mlx generate --text "Hello" --trim-mode energy --output out.wav
echo-tts-mlx generate --text "Hello" --trim-mode f0 --output out.wav
```

**Default stays `latent`** to avoid adding runtime cost. Users experiencing tail artifacts can opt into `energy` or `f0`.

#### Data-Driven Truncation Factor

The current fixed `truncation_factor=0.8` is a guess. Use diagnostic data from 8.1a to find optimal values.

**Method:**
1. For each sequence length in the test matrix, sweep truncation_factor from 0.7 to 1.0 in steps of 0.02
2. Measure: tail-to-body F0 variance ratio, overall MCD vs a "clean" reference (body-only)
3. Find the Pareto-optimal truncation per sequence length: lowest tail ratio while keeping MCD < 1 dB above minimum

Store results in a lookup table:
```python
# Values populated from diagnostic sweep — these are PLACEHOLDERS
ADAPTIVE_TRUNCATION: dict[int, float] = {
    # seq_length: truncation_factor (to be filled by 8.1a sweep)
}
```

**CLI:**
```bash
# Explicit (unchanged)
echo-tts-mlx generate --text "Hello" --truncation-factor 0.8

# Auto: interpolate from data-driven table
echo-tts-mlx generate --text "Hello" --truncation-factor auto

# Presets use auto by default
echo-tts-mlx generate --text "Hello" --preset balanced
```

The sweep script writes the table to `logs/truncation_sweep.json`. A developer reviews and hardcodes the final values into `pipeline.py`. No runtime sweep — just a one-time calibration.

### Files Changed

| File | Change |
|---|---|
| `scripts/tail_pitch_analysis.py` | **New** — diagnostic + truncation sweep script |
| `src/echo_tts_mlx/sampler.py` | Add `find_content_boundary()` |
| `src/echo_tts_mlx/pipeline.py` | Add `trim_mode` param to `decode_latents()` and `generate()`; add `ADAPTIVE_TRUNCATION` table; update preset defaults |
| `src/echo_tts_mlx/cli.py` | Add `--trim-mode` and `--truncation-factor auto` |
| `tests/` | New tests for `find_content_boundary`, adaptive truncation interpolation |
| `docs/QUANTIZATION.md` or new `docs/AUDIO_QUALITY.md` | Document trim modes and truncation behavior |

### Validation

| Gate | Criterion |
|---|---|
| F0 tail ratio (f0 mode) | Tail 2s F0 variance < 2× body median for ≥ 80% of test matrix samples |
| F0 tail ratio (energy mode) | Measurable improvement over latent-only mode across test matrix |
| MCD regression | Body-only MCD (excluding trimmed tail) within 0.5 dB of baseline |
| Audio duration | Trimmed audio ≥ 90% of baseline trimmed length (not over-trimming) |
| Test suite | All 146+ existing tests pass + new tests for boundary detection |
| Backward compat | `trim_mode="latent"` (default) produces identical output to v1.0.0 |

**How we determine "improved":** Run the diagnostic script before and after, compare:
1. Tail-to-body F0 variance ratio distributions (histogram)
2. Count of samples with audible tail artifacts (manual listen to all 21 samples)
3. MCD of trimmed body region (shouldn't regress)

This gives quantitative evidence, not just subjective opinion.

### Effort Estimate

- 8.1a (diagnostic script + initial sweep): 1 day
- 8.1b (improved trimming + CLI integration): 1 day
- 8.1c (truncation sweep + table): Half day (piggybacks on 8.1a infrastructure)
- Tests + validation runs: Half day
- **Total: ~3 days**

---

## Workstream 2: Improved 4-Bit Quantization

### Problem

Current 4-bit quantization uses affine integer mode with `group_size=64` and produces catastrophic quality degradation: MCD 21–85 dB vs float16 baseline (compared to 2.9–6.5 dB for 8-bit). The existing mixed policy (DiT → 4-bit, encoders → 8-bit) isn't enough because the DiT blocks themselves are too sensitive at affine 4-bit precision.

### Key Discovery: `nn.quantize` Supports `mode` Parameter

Verified on MLX 0.31.0: `mlx.nn.quantize()` accepts `mode="mxfp4"` and passes it through to `mx.core.quantize`. The `class_predicate` can return a **dict of per-module params** (bits, group_size, mode) instead of just `True/False`. This enables mixed-precision quantization in a single pass:

```python
nn.quantize(model, mode="mxfp4", group_size=32, bits=4,
            class_predicate=lambda path, mod: 
                {"bits": 8, "group_size": 64, "mode": "affine"} 
                if path in sensitive_layers 
                else isinstance(mod, nn.Linear))
```

### MXFP4 Uniform Quantization

Add `mxfp4` as a new quantize mode alongside `none`, `8bit`, and `4bit`.

**MXFP4 format:**
- E2M1 floating-point values (not integer)
- Shared E8M0 scale per group of 32 elements
- Fixed `group_size=32` (required by the MX specification)
- No bias term (unlike affine)

**Why it may help:** Floating-point quantization preserves more dynamic range than affine integer. The E2M1 format can represent {0, 0.5, 1.0, 1.5} × sign with an 8-bit shared exponent, which may better capture the weight distributions in DiT attention layers where affine 4-bit fails.

**Implementation in `model.py`:**

```python
VALID_QUANTIZE_MODES = {"none", "8bit", "4bit", "mxfp4", "mixed"}

def apply_quantization(self, *, mode: str, group_size: int = 64) -> None:
    mode = _normalize_quantize_mode(mode)
    # ... existing none/8bit/4bit paths ...
    
    elif mode == "mxfp4":
        # Uniform MXFP4 for all quantizable layers
        def pred(path: str, module: Any) -> bool:
            keep = self._quantize_predicate(path, module)
            if keep:
                selected[path] = 4
            return keep
        self.nn.quantize(self.tree, group_size=32, bits=4, mode="mxfp4",
                         class_predicate=pred)
```

**CLI:**
```bash
echo-tts-mlx generate --text "Hello" --quantize mxfp4 --output out.wav
```

**Expected memory:** Similar to current affine 4-bit (~3,165 MB, 48% below float16). Quality is the variable — if MXFP4 keeps MCD under 10 dB, it's a major win.

### Per-Layer Sensitivity Sweep

Systematically identify which DiT blocks/components tolerate 4-bit quantization.

Script: `scripts/layer_sensitivity_sweep.py`

**Method:**
For each of the 24 DiT blocks, and for attention vs MLP separately:
1. Load fresh float16 model from checkpoint
2. Quantize **only** the target component to MXFP4 (everything else stays float16)
3. Generate 3 test cases at seed=42 (short uncond, medium uncond, short cloned)
4. Compute MCD vs float16 baseline
5. Record MCD per (block_index, component_type)

**Total generations:** 24 blocks × 2 components × 3 cases = **144 generations**.
At ~10s each (short, 8-step): ~24 minutes total runtime. Practical on Mac mini.

**No deepcopy needed:** Reload from checkpoint each iteration. Model load is ~48s cold but weights are cached in unified memory after first load, so subsequent loads are ~2–5s.

**Output:** `logs/layer_sensitivity.json` + heatmap visualization:
```json
{
  "blocks.0.attention": {"mcd_short": 1.2, "mcd_medium": 1.5, "mcd_cloned": 1.8},
  "blocks.0.mlp": {"mcd_short": 0.8, "mcd_medium": 0.9, "mcd_cloned": 1.1},
  ...
  "blocks.23.attention": {"mcd_short": 15.3, "mcd_medium": 22.1, "mcd_cloned": 18.7}
}
```

**Expected pattern (hypothesis):**
- MLP layers more tolerant than attention (higher redundancy in FFN weights)
- Early blocks (0–7) more tolerant than late blocks (16–23)
- Late attention layers most sensitive (fine-grained F0/prosody control)

### Mixed-Precision Policy

Use sensitivity data from 8.2b to define a per-layer quantization policy.

```python
def apply_quantization(self, *, mode: str, group_size: int = 64) -> None:
    # ...
    elif mode == "mixed":
        # Per-module policy derived from sensitivity sweep
        def pred(path: str, module: Any) -> bool | dict:
            if not self._quantize_predicate(path, module):
                return False
            
            if path in SENSITIVE_MODULES:
                selected[path] = 8
                return {"bits": 8, "group_size": 64, "mode": "affine"}
            
            if path.startswith("text_encoder.") or path.startswith("speaker_encoder."):
                selected[path] = 8
                return {"bits": 8, "group_size": 64, "mode": "affine"}
            
            # Tolerant DiT layers → MXFP4
            selected[path] = 4
            return {"bits": 4, "group_size": 32, "mode": "mxfp4"}
        
        self.nn.quantize(self.tree, class_predicate=pred)
```

Where `SENSITIVE_MODULES` is populated from the sweep results — the set of module paths where MXFP4 produces MCD > some threshold (e.g., > 5 dB).

**This is the key innovation:** Instead of uniform 4-bit everywhere or uniform 8-bit everywhere, use the cheapest precision each layer can tolerate. Modules that handle it get MXFP4 (maximum compression), modules that don't get affine 8-bit (maximum quality).

### Save/Load for New Modes

**Quantize config format extension:**

```json
{
  "mode": "mixed",
  "per_module": true,
  "modules": {
    "blocks.0.attention.wq": {"bits": 4, "group_size": 32, "mode": "mxfp4"},
    "blocks.0.mlp.w1": {"bits": 4, "group_size": 32, "mode": "mxfp4"},
    "blocks.20.attention.wq": {"bits": 8, "group_size": 64, "mode": "affine"},
    "text_encoder.blocks.0.attention.wq": {"bits": 8, "group_size": 64, "mode": "affine"}
  }
}
```

**Backward compatibility:** 
- Existing configs (no `per_module` key) load with current uniform logic
- New configs with `per_module: true` use the per-module map
- `from_pretrained` detects config version and branches accordingly

**`save_quantized` update:** The existing path uses `nn.Module.save_weights()` which serializes whatever quantized state the modules hold. This should work for MXFP4 since MLX `QuantizedLinear` stores the quantized weight, scales, and mode internally. Verify during implementation.

### Files Changed

| File | Change |
|---|---|
| `scripts/layer_sensitivity_sweep.py` | **New** — per-layer MXFP4 sensitivity sweep |
| `src/echo_tts_mlx/model.py` | Extend `VALID_QUANTIZE_MODES`, add `mxfp4` and `mixed` paths to `apply_quantization()`, update config load/save for per-module format |
| `src/echo_tts_mlx/pipeline.py` | Pass `mode` through `from_pretrained`, update `save_quantized` |
| `src/echo_tts_mlx/cli.py` | Add `mxfp4` and `mixed` to `--quantize` choices |
| `scripts/quantize_quality_validation.py` | Add MXFP4 and mixed modes to validation matrix |
| `tests/` | New tests for MXFP4 quantize/dequantize, mixed policy, config save/load |
| `docs/QUANTIZATION.md` | Add MXFP4 and mixed mode documentation |

### Validation

All quality measurements use `scripts/quantize_quality_validation.py` methodology: same seed, same text prompts, MCD computed against float16 32-step baseline.

| Gate | Criterion |
|---|---|
| MXFP4 uniform MCD | < 15 dB (improvement over affine 4-bit's 21–85 dB) |
| Mixed policy MCD | < 10 dB |
| Speaker similarity | > 0.85 (8-bit baseline: 0.890) |
| Memory (MXFP4 uniform) | ~48% reduction vs float16 (similar to current affine 4-bit) |
| Memory (mixed) | > 38% reduction vs float16 (between 8-bit's 34% and MXFP4's 48%) |
| Speed | No regression vs affine 8-bit |
| Determinism | Reproducible with fixed seed |
| Save/load round-trip | Load quantized checkpoint → identical output to runtime quantization |
| Backward compat | Existing `8bit` and `4bit` modes unchanged |
| Test suite | All 146+ existing tests pass + new quant mode tests |

### Risk: MXFP4 May Not Help Enough

The E2M1 format has only 4 representable magnitudes (0, 0.5, 1.0, 1.5). If DiT weights have distributions that need finer resolution, MXFP4 may not significantly improve over affine 4-bit. The smaller group_size (32 vs 64) helps, but may not compensate.

**Mitigation:** The sensitivity sweep runs early and gives us data before committing to the full mixed-policy implementation. If MXFP4 uniform is still catastrophic (MCD > 40 dB), pivot to:
- Affine 4-bit with `group_size=32` (smaller groups, same format — may help)
- MLP-only 4-bit (keep all attention at 8-bit)
- Accept that 4-bit is infeasible for this model and document findings

The sweep data is valuable regardless — it characterizes the model's quantization sensitivity.

### Effort Estimate

- 8.2a (MXFP4 uniform mode): 1 day
- 8.2b (sensitivity sweep script + run): 1 day
- 8.2c (mixed policy implementation): 1 day
- 8.2d (save/load + config format): Half day
- Quality validation + benchmarks: Half day
- Documentation: Half day
- **Total: ~4.5 days**

---

## Implementation Order

```
1. Diagnostic tooling (tail pitch)
    ↓
2. Improved trimming
    ↓
3. Truncation sweep
    ↓  (Workstream 1 complete)
4. MXFP4 uniform mode
    ↓
5. Sensitivity sweep  ←── decision gate: is MXFP4 viable?
    ↓
6. Mixed-precision policy
    ↓
7. Save/load + docs
    ↓  (Workstream 2 complete)
```

Workstream 1 first because it fixes a user-audible quality issue with low risk. Workstream 2 has a research risk (MXFP4 may not help) so the sweep is placed early to fail fast.

---

## New Dependencies

| Package | Required By | Runtime? | Notes |
|---|---|---|---|
| `librosa` | Workstream 1 (F0 analysis), existing quality validation | **Optional** — only for `trim_mode="f0"` and scripts | Already installed in dev venv, not in pyproject.toml |
| `matplotlib` | Diagnostic scripts (plots) | **Dev only** | Not installed — add to `[dev]` extras |

**Action:** Add `librosa` and `matplotlib` to `pyproject.toml` under `[project.optional-dependencies].dev`. Keep them out of core deps — `trim_mode="f0"` does a lazy import with a helpful error if librosa is missing.

---

## Summary

| Workstream | Impact | Effort | Risk |
|---|---|---|---|
| **1: Tail Pitch** | Audible quality improvement via better trimming + data-driven truncation | ~3 days | Low — post-processing only, no model changes, default behavior unchanged |
| **2: 4-bit Quant** | New quantization tier with usable quality at ~48% memory savings | ~4.5 days | Medium — MXFP4 effectiveness on DiT is unproven; sweep fails fast if no-go |

**Total: ~7.5 days**

---

## Parity & Quality Gates (All Workstreams)

| Gate | Criterion |
|---|---|
| Backward compat (no-quantize) | `trim_mode="latent"` + `truncation_factor=0.8` → bitwise identical to v1.0.0 |
| Backward compat (8-bit) | Existing `--quantize 8bit` → bitwise identical to v1.0.0 |
| Test suite | All 146+ existing tests pass |
| New modes | Quality table published with MCD + speaker similarity + memory |

---

*Created: 2026-03-02*  
*Revised: 2026-03-02 (dropped Workstream 3, added implementation details, clarified validation)*  
*Baseline: echo-tts-mlx v1.0.0 (commit `f9b3846`)*
