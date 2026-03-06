"""Tests for blockwise generation — unit tests and integration tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import types
import warnings

import numpy as np
import pytest

from echo_tts_mlx.pipeline import _normalize_block_sizes
from echo_tts_mlx.sampler import BlockwiseSamplerConfig, sample_blockwise_euler_cfg


# ── Guards ────────────────────────────────────────────────────────────────────

WEIGHTS_DIR = Path("weights/converted")
BLOCKWISE_WEIGHTS_DIR = Path("weights/converted-blockwise")
HAS_BLOCKWISE = Path("weights/converted-blockwise/dit_weights.safetensors").exists()
HAS_CONVERTED = (
    (WEIGHTS_DIR / "config.json").exists()
    and (WEIGHTS_DIR / "dit_weights.safetensors").exists()
    and (WEIGHTS_DIR / "dac_weights.safetensors").exists()
    and (WEIGHTS_DIR / "pca_state.safetensors").exists()
)


def _mlx_runtime_available() -> bool:
    if importlib.util.find_spec("mlx") is None:
        return False
    proc = subprocess.run(
        [sys.executable, "-c", "import mlx.core as mx; _ = mx.array([0], dtype=mx.float16)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


HAS_MLX = _mlx_runtime_available()

needs_blockwise = pytest.mark.skipif(not HAS_BLOCKWISE, reason="blockwise weights not present")
needs_pruned = pytest.mark.skipif(not HAS_CONVERTED, reason="converted (pruned) weights not present")
needs_mlx = pytest.mark.skipif(not HAS_MLX, reason="mlx not installed or runtime unavailable")


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ══════════════════════════════════════════════════════════════════════════════


# ── Blockwise module detection ────────────────────────────────────────────────


@needs_blockwise
@needs_mlx
def test_detect_blockwise_modules():
    from echo_tts_mlx.model import MlxEchoDiT

    model = MlxEchoDiT.from_pretrained(BLOCKWISE_WEIGHTS_DIR, dtype="float16")
    assert model.has_blockwise_modules is True


@needs_pruned
@needs_mlx
def test_detect_blockwise_modules_pruned():
    from echo_tts_mlx.model import MlxEchoDiT

    model = MlxEchoDiT.from_pretrained(WEIGHTS_DIR, dtype="float16")
    assert model.has_blockwise_modules is False


# ── Latent encoder forward shape ──────────────────────────────────────────────


@needs_blockwise
@needs_mlx
def test_latent_encoder_forward_shape():
    import mlx.core as mx

    from echo_tts_mlx.model import MlxEchoDiT

    model = MlxEchoDiT.from_pretrained(BLOCKWISE_WEIGHTS_DIR, dtype="float16")
    prefix_latent = mx.zeros((1, 16, 80), dtype=mx.float32)
    kv = model.get_kv_cache_latent(prefix_latent)

    # 24 layers of KV pairs
    assert len(kv) == 24
    for k, v in kv:
        # Input 16 frames / patch_size 4 = 4 patched positions
        assert tuple(k.shape) == (1, 16, 4, 128), f"k shape mismatch: {tuple(k.shape)}"
        assert tuple(v.shape) == (1, 16, 4, 128), f"v shape mismatch: {tuple(v.shape)}"
        assert k.dtype == mx.float32
        assert v.dtype == mx.float32


# ── RoPE offset changes output ────────────────────────────────────────────────


@needs_blockwise
@needs_mlx
def test_rope_offset_changes_output():
    """start_pos=0 vs start_pos=64 should give different output from forward."""
    import mlx.core as mx

    from echo_tts_mlx.model import MlxEchoDiT

    model = MlxEchoDiT.from_pretrained(BLOCKWISE_WEIGHTS_DIR, dtype="float16")

    # Prepare minimal inputs
    text_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    text_mask = mx.ones((1, 3), dtype=mx.bool_)
    kv_text, text_k_mask = model.get_kv_cache_text(text_ids, text_mask)

    speaker_lat = mx.zeros((1, 4, 80), dtype=mx.float32)
    speaker_mask = mx.ones((1, 4), dtype=mx.bool_)
    kv_speaker, speaker_k_mask = model.get_kv_cache_speaker(speaker_lat, speaker_mask)

    latents = mx.ones((1, 8, 80), dtype=mx.float32) * 0.1
    timesteps = mx.array([0.5], dtype=mx.float32)

    out0 = model.forward(
        latents, timesteps,
        kv_text=kv_text, kv_speaker=kv_speaker,
        text_mask=text_k_mask, speaker_mask=speaker_k_mask,
        start_pos=0,
    )
    mx.eval(out0)

    out64 = model.forward(
        latents, timesteps,
        kv_text=kv_text, kv_speaker=kv_speaker,
        text_mask=text_k_mask, speaker_mask=speaker_k_mask,
        start_pos=64,
    )
    mx.eval(out64)

    diff = float(mx.max(mx.abs(out0 - out64)))
    assert diff > 1e-3, f"Outputs should differ with different start_pos, but max diff = {diff}"


# ── Forward backward compatibility ────────────────────────────────────────────


@needs_blockwise
@needs_mlx
def test_forward_backward_compat():
    """forward() with kv_latent=None, start_pos=0 matches call without those args."""
    import mlx.core as mx

    from echo_tts_mlx.model import MlxEchoDiT

    model = MlxEchoDiT.from_pretrained(BLOCKWISE_WEIGHTS_DIR, dtype="float16")

    text_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    text_mask = mx.ones((1, 3), dtype=mx.bool_)
    kv_text, text_k_mask = model.get_kv_cache_text(text_ids, text_mask)

    speaker_lat = mx.zeros((1, 4, 80), dtype=mx.float32)
    speaker_mask = mx.ones((1, 4), dtype=mx.bool_)
    kv_speaker, speaker_k_mask = model.get_kv_cache_speaker(speaker_lat, speaker_mask)

    latents = mx.ones((1, 8, 80), dtype=mx.float32) * 0.1
    timesteps = mx.array([0.5], dtype=mx.float32)

    # Call with explicit defaults
    out_explicit = model.forward(
        latents, timesteps,
        kv_text=kv_text, kv_speaker=kv_speaker,
        text_mask=text_k_mask, speaker_mask=speaker_k_mask,
        kv_latent=None, start_pos=0,
    )
    mx.eval(out_explicit)

    # Call without blockwise args (same defaults)
    out_default = model.forward(
        latents, timesteps,
        kv_text=kv_text, kv_speaker=kv_speaker,
        text_mask=text_k_mask, speaker_mask=speaker_k_mask,
    )
    mx.eval(out_default)

    np.testing.assert_allclose(
        np.asarray(out_explicit), np.asarray(out_default),
        atol=1e-5, rtol=1e-5,
        err_msg="forward() with explicit kv_latent=None, start_pos=0 should match defaults",
    )


# ── Rotary at positions vs contiguous ─────────────────────────────────────────


@needs_mlx
def test_custom_rope_matches_fast_rope():
    """Custom arbitrary-position RoPE should match mx.fast.rope for contiguous positions."""
    import mlx.core as mx

    from echo_tts_mlx.model import _apply_rotary_at_positions

    mx.random.seed(99)
    x = mx.random.normal((1, 4, 8, 128), dtype=mx.float32)
    mx.eval(x)

    # mx.fast.rope with traditional=False (split-half, same as custom impl)
    out_fast = mx.fast.rope(x, dims=128, traditional=False, base=10000.0, scale=1.0, offset=0)
    mx.eval(out_fast)

    # Custom with contiguous positions
    positions = mx.arange(8, dtype=mx.int32)
    out_custom = _apply_rotary_at_positions(x, positions, mx)
    mx.eval(out_custom)

    np.testing.assert_allclose(np.asarray(out_fast), np.asarray(out_custom), atol=1e-4, rtol=1e-4)


@needs_mlx
def test_custom_rope_with_offset_matches_fast_rope():
    import mlx.core as mx

    from echo_tts_mlx.model import _apply_rotary_at_positions

    mx.random.seed(7)
    x = mx.random.normal((1, 4, 8, 128), dtype=mx.float32)
    mx.eval(x)
    offset = 32

    out_fast = mx.fast.rope(x, dims=128, traditional=False, base=10000.0, scale=1.0, offset=offset)
    mx.eval(out_fast)

    positions = mx.arange(8, dtype=mx.int32) + offset
    out_custom = _apply_rotary_at_positions(x, positions, mx)
    mx.eval(out_custom)

    np.testing.assert_allclose(np.asarray(out_fast), np.asarray(out_custom), atol=1e-4, rtol=1e-4)


@needs_mlx
def test_custom_rope_non_contiguous_positions():
    import mlx.core as mx

    from echo_tts_mlx.model import _apply_rotary_at_positions

    mx.random.seed(123)
    x = mx.random.normal((1, 4, 8, 128), dtype=mx.float32)
    mx.eval(x)

    positions = mx.arange(8, dtype=mx.int32) * 4
    out = _apply_rotary_at_positions(x, positions, mx)
    mx.eval(out)

    assert tuple(out.shape) == tuple(x.shape)
    assert not np.allclose(np.asarray(out), np.asarray(x), atol=1e-4, rtol=1e-4)


def test_speaker_kv_no_drift_after_many_blocks():
    rng = np.random.default_rng(0)
    k0 = rng.standard_normal((1, 16, 64, 128), dtype=np.float32)
    v0 = rng.standard_normal((1, 16, 64, 128), dtype=np.float32)
    scale = np.float32(1.5)

    # Old approach: mutate in-place by multiply/divide each block.
    k_old = k0.copy()
    v_old = v0.copy()
    for _ in range(20):
        k_old *= scale
        v_old *= scale
        k_old /= scale
        v_old /= scale
    old_drift = max(float(np.max(np.abs(k_old - k0))), float(np.max(np.abs(v_old - v0))))

    # New approach: derive per-block tensors from an immutable snapshot.
    k_snapshot = k0.copy()
    v_snapshot = v0.copy()
    for _ in range(20):
        _scaled_k = k_snapshot * scale
        _scaled_v = v_snapshot * scale
        _ = (_scaled_k, _scaled_v)
        restored_k = k_snapshot
        restored_v = v_snapshot
    new_drift = max(float(np.max(np.abs(restored_k - k0))), float(np.max(np.abs(restored_v - v0))))

    assert old_drift > 1e-7
    assert new_drift == 0.0


def test_generate_blockwise_no_decode_passes_latents(monkeypatch: pytest.MonkeyPatch):
    from echo_tts_mlx.pipeline import EchoTTS
    import echo_tts_mlx.pipeline as pipeline_mod

    class _DummyRandom:
        @staticmethod
        def seed(_seed: int | None) -> None:
            return None

        @staticmethod
        def normal(shape, dtype=np.float32):
            return np.zeros(shape, dtype=dtype)

    class _DummyMx:
        float32 = np.float32
        bool_ = np.bool_
        random = _DummyRandom()

        @staticmethod
        def zeros(shape, dtype=np.float32):
            return np.zeros(shape, dtype=dtype)

        @staticmethod
        def zeros_like(x):
            return np.zeros_like(x)

        @staticmethod
        def ones(shape, dtype=np.bool_):
            return np.ones(shape, dtype=dtype)

        @staticmethod
        def concatenate(items, axis=0):
            return np.concatenate(items, axis=axis)

        @staticmethod
        def array(values, dtype=None):
            return np.array(values, dtype=dtype)

        @staticmethod
        def eval(*_values):
            return None

    class _DummyModel:
        has_blockwise_modules = True

        @staticmethod
        def get_kv_cache_text(text_ids, text_mask):
            kv = [(np.zeros((1, 1, 1, 1), dtype=np.float32), np.zeros((1, 1, 1, 1), dtype=np.float32))]
            return kv, text_mask

        @staticmethod
        def get_kv_cache_speaker(speaker_latents, speaker_mask):
            kv = [(np.zeros((1, 1, 1, 1), dtype=np.float32), np.zeros((1, 1, 1, 1), dtype=np.float32))]
            return kv, speaker_mask

    dummy = types.SimpleNamespace()
    dummy.mx = _DummyMx()
    dummy.model = _DummyModel()
    dummy.config = types.SimpleNamespace(
        speaker_patch_size=4,
        latent_size=80,
        max_latent_length=640,
    )
    dummy._repeat_kv_cache = lambda kv, repeats: kv * int(repeats)
    dummy.prepare_text = lambda _text: (
        np.array([[1, 2, 3]], dtype=np.int32),
        np.ones((1, 3), dtype=np.bool_),
    )
    dummy.prepare_speaker_latents = lambda **_kwargs: (
        np.zeros((1, 4, 80), dtype=np.float32),
        np.ones((1, 4), dtype=np.bool_),
    )
    dummy.encode_continuation = lambda **_kwargs: (_ for _ in ()).throw(AssertionError("not expected"))

    decode_calls: list[tuple[tuple[int, ...], bool]] = []

    def _decode_latents(latents, *, trim_latents=True, **_kwargs):
        decode_calls.append((tuple(np.asarray(latents).shape), bool(trim_latents)))
        return np.zeros((1, 1, 16), dtype=np.float32)

    dummy.decode_latents = _decode_latents

    callback_payload_shapes: list[tuple[int, ...]] = []

    def _on_block_complete(_idx: int, _total: int, payload) -> None:
        callback_payload_shapes.append(tuple(np.asarray(payload).shape))

    def _fake_sampler(**kwargs):
        on_block_complete = kwargs["on_block_complete"]
        block_latents = np.zeros((1, 32, 80), dtype=np.float32)
        on_block_complete(0, 1, block_latents)
        return np.zeros((1, 32, 80), dtype=np.float32)

    monkeypatch.setattr(pipeline_mod, "sample_blockwise_euler_cfg", _fake_sampler)

    out = EchoTTS.generate_blockwise(
        dummy,
        text="[S1] test",
        block_sizes=[32],
        num_steps=2,
        decode_intermediate_blocks=False,
        on_block_complete=_on_block_complete,
    )

    assert tuple(np.asarray(out).shape) == (1, 1, 16)
    assert callback_payload_shapes == [(1, 32, 80)]
    assert decode_calls == [((1, 32, 80), True)]


@needs_blockwise
@needs_mlx
def test_encode_latent_output_shape():
    import mlx.core as mx

    from echo_tts_mlx.model import MlxEchoDiT

    model = MlxEchoDiT.from_pretrained(BLOCKWISE_WEIGHTS_DIR, dtype="float16")
    patch = int(model.config.speaker_patch_size)
    prefix_latents = mx.random.normal((1, 256, int(model.config.latent_size)), dtype=mx.float32)

    hidden, mask = model._encode_latent(prefix_latents)
    mx.eval(hidden, mask)

    assert tuple(hidden.shape) == (1, 256 // patch, int(model.config.speaker_model_size))
    assert tuple(mask.shape) == (1, 256 // patch)


@needs_blockwise
@needs_mlx
def test_encode_latent_zero_padded_region():
    import mlx.core as mx

    from echo_tts_mlx.model import MlxEchoDiT

    model = MlxEchoDiT.from_pretrained(BLOCKWISE_WEIGHTS_DIR, dtype="float16")

    real = np.random.default_rng(42).standard_normal((1, 128, int(model.config.latent_size))).astype(np.float32)
    zeros = np.zeros((1, 128, int(model.config.latent_size)), dtype=np.float32)
    prefix = mx.array(np.concatenate([real, zeros], axis=1), dtype=mx.float32)

    hidden, _ = model._encode_latent(prefix)
    mx.eval(hidden)
    hidden_np = np.asarray(hidden)

    half = hidden_np.shape[1] // 2
    real_region = hidden_np[:, :half, :]
    zero_region = hidden_np[:, half:, :]
    assert float(np.max(np.abs(zero_region))) > 1e-6
    assert not np.allclose(real_region, zero_region, atol=1e-5, rtol=1e-5)


def _latent_visible_mask(*, t_latent: int, start_pos: int, patch_size: int) -> np.ndarray:
    latent_positions = np.arange(t_latent, dtype=np.int32) * int(patch_size)
    return latent_positions < int(start_pos)


def test_latent_visibility_block0_no_continuation():
    visible = _latent_visible_mask(t_latent=64, start_pos=0, patch_size=4)
    assert visible.shape == (64,)
    assert int(visible.sum()) == 0


def test_latent_visibility_block0_with_continuation():
    visible = _latent_visible_mask(t_latent=64, start_pos=128, patch_size=4)
    assert int(visible.sum()) == 32
    assert np.all(visible[:32])
    assert not np.any(visible[32:])


def test_latent_visibility_block1():
    visible = _latent_visible_mask(t_latent=64, start_pos=256, patch_size=4)
    assert int(visible.sum()) == 64


# ── Encode continuation patch trim ───────────────────────────────────────────


@needs_blockwise
@needs_mlx
def test_encode_continuation_patch_trim():
    """Continuation latents (1,11,80) should trim to (1,8,80) for patch alignment."""
    from echo_tts_mlx.pipeline import EchoTTS

    pipeline = EchoTTS.from_pretrained(BLOCKWISE_WEIGHTS_DIR, dtype="float16")
    latents = np.zeros((1, 11, 80), dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cont, length = pipeline.encode_continuation(latents=latents)

    assert length == 8
    assert tuple(cont.shape) == (1, 8, 80)


# ── Block size validation ─────────────────────────────────────────────────────


def test_block_size_validation():
    """_normalize_block_sizes rejects <4, warns <32."""
    # Reject block size < 4 (patch_size=4)
    with pytest.raises(ValueError, match="must be >= 4"):
        _normalize_block_sizes([3, 32], patch_size=4)

    with pytest.raises(ValueError, match="must be >= 4"):
        _normalize_block_sizes([2], patch_size=4)

    # Warn on block sizes < 32
    with pytest.warns(UserWarning, match="Block sizes < 32"):
        _normalize_block_sizes([16, 128], patch_size=4)

    # Valid sizes pass without error
    sizes = _normalize_block_sizes([32, 64, 128], patch_size=4)
    assert sizes == [32, 64, 128]

    # Empty block sizes rejected
    with pytest.raises(ValueError, match="must not be empty"):
        _normalize_block_sizes([], patch_size=4)

    # Zero block size rejected
    with pytest.raises(ValueError, match="must be > 0"):
        _normalize_block_sizes([0, 32], patch_size=4)


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════


@needs_blockwise
@needs_mlx
def test_generate_blockwise_produces_audio():
    """Blockwise generation with block_sizes=[32,32], num_steps=2 produces non-silent audio."""
    from echo_tts_mlx.pipeline import EchoTTS

    pipeline = EchoTTS.from_pretrained(BLOCKWISE_WEIGHTS_DIR, dtype="float16")
    audio = pipeline.generate_blockwise(
        text="[S1] Hello, this is a blockwise test.",
        block_sizes=[32, 32],
        num_steps=2,
        seed=42,
        trim_latents=False,
    )

    audio_np = np.asarray(audio, dtype=np.float32).reshape(-1)
    assert audio_np.size > 0, "Output audio should not be empty"
    peak = float(np.max(np.abs(audio_np)))
    assert peak > 1e-4, f"Audio should be non-silent, but peak amplitude = {peak}"


@needs_blockwise
@needs_mlx
def test_blockwise_continuation_length_enforced():
    """620-frame continuation + [32] blocks should exceed 640 max and raise ValueError."""
    import mlx.core as mx

    from echo_tts_mlx.pipeline import EchoTTS

    pipeline = EchoTTS.from_pretrained(BLOCKWISE_WEIGHTS_DIR, dtype="float16")

    # 620 frames of continuation + 32 block = 652 > 640
    big_continuation = np.zeros((1, 620, 80), dtype=np.float32)
    with pytest.raises(ValueError, match="must be <= 640"):
        pipeline.generate_blockwise(
            text="[S1] Overflow test.",
            block_sizes=[32],
            continuation_latents=big_continuation,
            num_steps=2,
            seed=0,
        )


@needs_blockwise
@needs_mlx
def test_blockwise_determinism():
    """Same seed should produce identical output across two runs."""
    from echo_tts_mlx.pipeline import EchoTTS

    pipeline = EchoTTS.from_pretrained(BLOCKWISE_WEIGHTS_DIR, dtype="float16")

    kwargs = dict(
        text="[S1] Determinism check.",
        block_sizes=[32, 32],
        num_steps=2,
        seed=123,
        trim_latents=False,
        return_latents=True,
    )

    audio1, latents1 = pipeline.generate_blockwise(**kwargs)
    audio2, latents2 = pipeline.generate_blockwise(**kwargs)

    np.testing.assert_allclose(
        np.asarray(latents1), np.asarray(latents2),
        atol=0, rtol=0,
        err_msg="Blockwise generation should be deterministic with the same seed",
    )


@needs_blockwise
@needs_mlx
def test_blockwise_streaming_callback():
    """on_block_complete should fire twice for block_sizes=[32, 32]."""
    from echo_tts_mlx.pipeline import EchoTTS

    pipeline = EchoTTS.from_pretrained(BLOCKWISE_WEIGHTS_DIR, dtype="float16")

    callback_log: list[tuple[int, int]] = []

    def on_block(block_idx: int, total_blocks: int, block_audio) -> None:
        callback_log.append((block_idx, total_blocks))

    pipeline.generate_blockwise(
        text="[S1] Streaming callback test.",
        block_sizes=[32, 32],
        num_steps=2,
        seed=0,
        trim_latents=False,
        on_block_complete=on_block,
    )

    assert len(callback_log) == 2, f"Expected 2 callbacks, got {len(callback_log)}"
    assert callback_log[0] == (0, 2)
    assert callback_log[1] == (1, 2)


@needs_pruned
@needs_mlx
def test_blockwise_without_modules_raises():
    """Pruned weights should raise RuntimeError on generate_blockwise."""
    from echo_tts_mlx.pipeline import EchoTTS

    pipeline = EchoTTS.from_pretrained(WEIGHTS_DIR, dtype="float16")

    with pytest.raises(RuntimeError, match="Blockwise generation requires weights"):
        pipeline.generate_blockwise(
            text="[S1] Should fail.",
            block_sizes=[32, 32],
            num_steps=2,
            seed=0,
        )
