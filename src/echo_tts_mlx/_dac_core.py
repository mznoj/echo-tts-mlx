# Copyright 2025 Jordan Darefsky (original Echo-TTS)
# Copyright 2026 Matthew Znoj (MLX port)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from the original Echo-TTS autoencoder implementation:
# - Ported to MLX
# - Adapted for converted checkpoint format

"""MLX implementation for the Fish S1-DAC autoencoder.

Inference-only port mirroring the upstream PyTorch reference.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ._conversion_utils import to_mlx_state

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - handled at runtime in CLI
    mx = None


def _require_mlx() -> None:
    if mx is None:  # pragma: no cover
        raise RuntimeError("MLX is required for DAC inference.")


def _snake(x: "mx.array", alpha: "mx.array") -> "mx.array":
    return x + (1.0 / (alpha + 1e-9)) * mx.sin(alpha * x) ** 2


def _rms_norm(x: "mx.array", weight: "mx.array", eps: float = 1e-6) -> "mx.array":
    norm = 1.0 / mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * norm * weight


def _linear(x: "mx.array", weight: "mx.array", bias: Optional["mx.array"] = None) -> "mx.array":
    y = x @ mx.transpose(weight)
    if bias is not None:
        y = y + bias
    return y


def _silu(x: "mx.array") -> "mx.array":
    return x * (1.0 / (1.0 + mx.exp(-x)))


def _gelu(x: "mx.array") -> "mx.array":
    # Fast GELU approximation used in PyTorch (`approximate='tanh'`).
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3))))


def _swiglu(x: "mx.array", w1: "mx.array", w2: "mx.array", w3: "mx.array") -> "mx.array":
    return _linear(_silu(_linear(x, w1)) * _linear(x, w3), w2)


def _precompute_freqs(head_dim: int, length: int, theta: float, dtype: "mx.Dtype") -> tuple["mx.array", "mx.array"]:
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32)[: head_dim // 2] / head_dim))
    t = np.arange(length, dtype=np.float32)
    angles = np.outer(t, freqs)
    cos = mx.array(np.cos(angles), dtype=dtype)
    sin = mx.array(np.sin(angles), dtype=dtype)
    return cos, sin


def _apply_rotary(x: "mx.array", cos: "mx.array", sin: "mx.array") -> "mx.array":
    # x: (B, T, H, D)
    b, t, h, d = x.shape
    x_ = mx.reshape(x, (b, t, h, d // 2, 2))
    x_r = x_[..., 0]
    x_i = x_[..., 1]
    cos = mx.reshape(cos[:t], (1, t, 1, d // 2))
    sin = mx.reshape(sin[:t], (1, t, 1, d // 2))
    y_r = x_r * cos - x_i * sin
    y_i = x_r * sin + x_i * cos
    return mx.reshape(mx.stack([y_r, y_i], axis=-1), x.shape)


def _causal_mask(length: int, window_size: Optional[int], dtype: "mx.Dtype") -> "mx.array":
    mask = np.triu(np.ones((length, length), dtype=np.float32), k=1)
    if window_size is not None:
        window = np.triu(np.ones((length, length), dtype=np.float32), k=-window_size)
        mask = np.maximum(mask, 1.0 - window)
    # float mask with 0/1 to avoid bool + where API differences.
    return mx.array(mask, dtype=dtype)


class MlxFishS1DAC:
    """Inference-only Fish S1-DAC model backed by folded checkpoint tensors."""

    def __init__(self, np_state: dict[str, object]) -> None:
        _require_mlx()
        self.state = {k: v.astype(mx.float32) for k, v in to_mlx_state(np_state).items()}
        self.n_head = 16
        self.window_size = 128

    def t(self, key: str) -> "mx.array":
        if key not in self.state:
            raise KeyError(f"Missing checkpoint key: {key}")
        return self.state[key]

    def _conv1d(
        self,
        x: "mx.array",
        weight: "mx.array",
        bias: Optional["mx.array"] = None,
        *,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ) -> "mx.array":
        # x: (B, C_in, T), weight: (C_out, C_in/groups, K)
        # Use mx.conv1d which expects:
        #   input:  (B, T, C_in)  — channels-last
        #   weight: (C_out, K, C_in/groups) — channels-last
        # So we transpose in and out.
        x_cl = mx.transpose(x, (0, 2, 1))                      # (B, T, C_in)
        w_cl = mx.transpose(weight, (0, 2, 1))                  # (C_out, K, C_in/groups)
        y_cl = mx.conv1d(x_cl, w_cl, stride=stride, dilation=dilation, groups=groups)  # (B, T_out, C_out)
        y = mx.transpose(y_cl, (0, 2, 1))                       # (B, C_out, T_out)

        if bias is not None:
            y = y + mx.reshape(bias, (1, -1, 1))
        return y

    def _causal_conv1d(
        self,
        x: "mx.array",
        weight: "mx.array",
        bias: Optional["mx.array"] = None,
        *,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ) -> "mx.array":
        kernel_size = int(weight.shape[-1])
        effective_kernel = (kernel_size - 1) * dilation + 1
        # Match upstream CausalConvNet: pad_left = effective_kernel - stride
        pad_left = effective_kernel - stride
        length = int(x.shape[-1])
        n_frames = (length - effective_kernel + pad_left) / stride + 1
        import math
        ideal_length = (math.ceil(n_frames) - 1) * stride + (effective_kernel - pad_left)
        extra_right = max(int(ideal_length - length), 0)
        x = mx.pad(x, ((0, 0), (0, 0), (pad_left, extra_right)))
        return self._conv1d(x, weight, bias=bias, stride=stride, dilation=dilation, groups=groups)

    def _conv_transpose1d(
        self,
        x: "mx.array",
        weight: "mx.array",
        bias: Optional["mx.array"] = None,
        *,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
    ) -> "mx.array":
        # x: (B, C_in, T), weight: (C_in, C_out, K)
        # Use mx.conv_transpose1d which expects:
        #   input:  (B, T, C_in)  — channels-last
        #   weight: (C_out, K, C_in) — note: different axis order
        # PyTorch ConvTranspose1d weight: (C_in, C_out, K)
        # MLX conv_transpose1d weight: (C_out, K, C_in)
        x_cl = mx.transpose(x, (0, 2, 1))                       # (B, T, C_in)
        # Transpose weight from (C_in, C_out, K) to (C_out, K, C_in)
        w_cl = mx.transpose(weight, (1, 2, 0))                   # (C_out, K, C_in)
        y_cl = mx.conv_transpose1d(
            x_cl, w_cl, stride=stride, padding=padding,
            dilation=dilation, groups=1
        )  # (B, T_out, C_out)
        # Handle output_padding by appending zeros if needed
        if output_padding > 0:
            pad_shape = list(y_cl.shape)
            pad_shape[1] = output_padding
            y_cl = mx.concatenate([y_cl, mx.zeros(pad_shape, dtype=y_cl.dtype)], axis=1)
        y = mx.transpose(y_cl, (0, 2, 1))                        # (B, C_out, T_out)

        if bias is not None:
            y = y + mx.reshape(bias, (1, -1, 1))
        return y

    def _causal_conv_transpose1d(
        self,
        x: "mx.array",
        weight: "mx.array",
        bias: Optional["mx.array"] = None,
        *,
        stride: int,
        dilation: int = 1,
    ) -> "mx.array":
        # Match upstream CausalTransConvNet: plain conv_transpose then trim
        kernel_size = int(weight.shape[-1])
        y = self._conv_transpose1d(x, weight, bias=bias, stride=stride, dilation=dilation)
        # Trim for causality: remove (kernel_size - stride) samples
        pad = kernel_size - stride
        padding_right = math.ceil(pad)
        padding_left = pad - padding_right
        if padding_left > 0 or padding_right > 0:
            end = int(y.shape[-1]) - padding_right if padding_right > 0 else int(y.shape[-1])
            y = y[..., padding_left:end]
        return y

    def _transformer_attention(
        self,
        x: "mx.array",
        prefix: str,
        *,
        window_size: Optional[int],
    ) -> "mx.array":
        # x: (B, T, C)
        b, t, c = x.shape
        h = self.n_head
        d = c // h

        qkv = _linear(x, self.t(f"{prefix}.wqkv.weight"))
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = mx.reshape(q, (b, t, h, d))
        k = mx.reshape(k, (b, t, h, d))
        v = mx.reshape(v, (b, t, h, d))

        cos, sin = _precompute_freqs(d, int(t), theta=10000.0, dtype=x.dtype)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) / math.sqrt(d)
        mask = _causal_mask(int(t), window_size=window_size, dtype=scores.dtype)
        scores = scores + mx.reshape(mask * (-1e9), (1, 1, t, t))

        attn = mx.softmax(scores, axis=-1)
        y = mx.matmul(attn, v)
        y = mx.transpose(y, (0, 2, 1, 3))
        y = mx.reshape(y, (b, t, c))
        return _linear(y, self.t(f"{prefix}.wo.weight"))

    def _transformer_core(
        self,
        x: "mx.array",
        prefix: str,
        *,
        n_layers: int,
        window_size: Optional[int],
    ) -> "mx.array":
        # x: (B, T, C)
        for i in range(n_layers):
            lprefix = f"{prefix}.layers.{i}"

            h = _rms_norm(x, self.t(f"{lprefix}.attention_norm.weight"))
            h = self._transformer_attention(h, f"{lprefix}.attention", window_size=window_size)
            h = h * self.t(f"{lprefix}.attention_layer_scale.gamma")
            x = x + h

            h = _rms_norm(x, self.t(f"{lprefix}.ffn_norm.weight"))
            h = _swiglu(
                h,
                self.t(f"{lprefix}.feed_forward.w1.weight"),
                self.t(f"{lprefix}.feed_forward.w2.weight"),
                self.t(f"{lprefix}.feed_forward.w3.weight"),
            )
            h = h * self.t(f"{lprefix}.ffn_layer_scale.gamma")
            x = x + h

        return _rms_norm(x, self.t(f"{prefix}.norm.weight"))

    def _window_transformer(
        self,
        x: "mx.array",
        prefix: str,
        *,
        n_layers: int,
        window_size: int,
    ) -> "mx.array":
        # x: (B, C, T)
        x = mx.transpose(x, (0, 2, 1))
        t = int(x.shape[1])
        chunks = []
        for start in range(0, t, window_size):
            seg = x[:, start : start + window_size, :]
            chunks.append(self._transformer_core(seg, prefix, n_layers=n_layers, window_size=window_size))
        x = mx.concatenate(chunks, axis=1)
        return mx.transpose(x, (0, 2, 1))

    def _residual_unit(self, x: "mx.array", prefix: str, *, dilation: int) -> "mx.array":
        y = _snake(x, self.t(f"{prefix}.block.0.alpha"))
        y = self._causal_conv1d(
            y,
            self.t(f"{prefix}.block.1.conv.weight"),
            self.t(f"{prefix}.block.1.conv.bias"),
            dilation=dilation,
        )
        y = _snake(y, self.t(f"{prefix}.block.2.alpha"))
        y = self._causal_conv1d(
            y,
            self.t(f"{prefix}.block.3.conv.weight"),
            self.t(f"{prefix}.block.3.conv.bias"),
            dilation=1,
        )
        return x + y

    def _encoder_block(self, x: "mx.array", prefix: str, *, stride: int, n_t_layers: int) -> "mx.array":
        x = self._residual_unit(x, f"{prefix}.block.0", dilation=1)
        x = self._residual_unit(x, f"{prefix}.block.1", dilation=3)
        x = self._residual_unit(x, f"{prefix}.block.2", dilation=9)
        x = _snake(x, self.t(f"{prefix}.block.3.alpha"))
        x = self._causal_conv1d(
            x,
            self.t(f"{prefix}.block.4.conv.weight"),
            self.t(f"{prefix}.block.4.conv.bias"),
            stride=stride,
        )
        if n_t_layers > 0:
            x = self._transformer_core(
                mx.transpose(x, (0, 2, 1)),
                f"{prefix}.block.5",
                n_layers=n_t_layers,
                window_size=None,
            )
            x = mx.transpose(x, (0, 2, 1))
        return x

    def _decoder_block(self, x: "mx.array", prefix: str, *, stride: int, n_t_layers: int) -> "mx.array":
        x = _snake(x, self.t(f"{prefix}.block.0.alpha"))
        x = self._causal_conv_transpose1d(
            x,
            self.t(f"{prefix}.block.1.conv.weight"),
            self.t(f"{prefix}.block.1.conv.bias"),
            stride=stride,
        )
        x = self._residual_unit(x, f"{prefix}.block.2", dilation=1)
        x = self._residual_unit(x, f"{prefix}.block.3", dilation=3)
        x = self._residual_unit(x, f"{prefix}.block.4", dilation=9)
        if n_t_layers > 0:
            x = self._transformer_core(
                mx.transpose(x, (0, 2, 1)),
                f"{prefix}.block.5",
                n_layers=n_t_layers,
                window_size=None,
            )
            x = mx.transpose(x, (0, 2, 1))
        return x

    def _convnext_block(self, x: "mx.array", prefix: str) -> "mx.array":
        residual = x
        dw_w = self.t(f"{prefix}.dwconv.conv.weight")
        dw_b = self.t(f"{prefix}.dwconv.conv.bias")
        x = self._causal_conv1d(x, dw_w, dw_b, groups=int(dw_w.shape[0]))

        x = mx.transpose(x, (0, 2, 1))
        mu = mx.mean(x, axis=-1, keepdims=True)
        var = mx.mean((x - mu) ** 2, axis=-1, keepdims=True)
        x = (x - mu) / mx.sqrt(var + 1e-6)
        x = x * self.t(f"{prefix}.norm.weight") + self.t(f"{prefix}.norm.bias")
        x = _linear(x, self.t(f"{prefix}.pwconv1.weight"), self.t(f"{prefix}.pwconv1.bias"))
        x = _gelu(x)
        x = _linear(x, self.t(f"{prefix}.pwconv2.weight"), self.t(f"{prefix}.pwconv2.bias"))
        x = mx.transpose(x, (0, 2, 1))

        gamma = mx.reshape(self.t(f"{prefix}.gamma"), (1, -1, 1))
        return residual + gamma * x

    def _vector_quantize(self, z: "mx.array", prefix: str) -> tuple["mx.array", "mx.array", "mx.array"]:
        in_w = self.t(f"{prefix}.in_proj.weight")
        in_b = self.t(f"{prefix}.in_proj.bias")
        out_w = self.t(f"{prefix}.out_proj.weight")
        out_b = self.t(f"{prefix}.out_proj.bias")
        codebook = self.t(f"{prefix}.codebook.weight")

        z_e = self._conv1d(z, in_w, in_b)

        b, d, t = z_e.shape
        flat = mx.reshape(mx.transpose(z_e, (0, 2, 1)), (-1, d))

        distances = (
            mx.sum(flat * flat, axis=1, keepdims=True)
            - 2.0 * mx.matmul(flat, mx.transpose(codebook))
            + mx.sum(codebook * codebook, axis=1)
        )
        indices = mx.reshape(mx.argmin(distances, axis=1), (b, t))

        emb = mx.take(codebook, indices, axis=0)
        z_q = mx.transpose(emb, (0, 2, 1))
        z_q = self._conv1d(z_q, out_w, out_b)
        return z_q, indices, z_e

    def _residual_vq(self, z: "mx.array", n_quantizers: Optional[int]) -> tuple["mx.array", "mx.array", "mx.array"]:
        total_q = 10
        if n_quantizers is None:
            n_quantizers = total_q
        if n_quantizers < 1 or n_quantizers > total_q:
            raise ValueError(f"n_quantizers must be in [1, {total_q}], got {n_quantizers}")

        z_q_sem, idx_sem, lat_sem = self._vector_quantize(
            z,
            "quantizer.semantic_quantizer.quantizers.0",
        )

        codes = [mx.expand_dims(idx_sem, axis=0)]
        latents = [lat_sem]

        residual = z - z_q_sem
        z_q = z_q_sem

        for i in range(n_quantizers - 1):
            z_q_i, idx_i, lat_i = self._vector_quantize(residual, f"quantizer.quantizer.quantizers.{i}")
            residual = residual - z_q_i
            z_q = z_q + z_q_i
            codes.append(mx.expand_dims(idx_i, axis=0))
            latents.append(lat_i)

        codes_stacked = mx.concatenate(codes, axis=0)
        latents_stacked = mx.concatenate(latents, axis=1)
        return z_q, codes_stacked, latents_stacked

    def encode_zq(self, audio: "mx.array", n_quantizers: Optional[int] = None) -> tuple["mx.array", "mx.array", "mx.array"]:
        if audio.ndim != 3 or int(audio.shape[1]) != 1:
            raise ValueError(f"audio must have shape (B, 1, samples), got {tuple(audio.shape)}")

        x = audio.astype(mx.float32)

        x = self._causal_conv1d(x, self.t("encoder.block.0.conv.weight"), self.t("encoder.block.0.conv.bias"))
        x = self._encoder_block(x, "encoder.block.1", stride=2, n_t_layers=0)
        x = self._encoder_block(x, "encoder.block.2", stride=4, n_t_layers=0)
        x = self._encoder_block(x, "encoder.block.3", stride=8, n_t_layers=0)
        x = self._encoder_block(x, "encoder.block.4", stride=8, n_t_layers=4)
        x = _snake(x, self.t("encoder.block.5.alpha"))
        x = self._causal_conv1d(x, self.t("encoder.block.6.conv.weight"), self.t("encoder.block.6.conv.bias"))

        x = self._causal_conv1d(
            x,
            self.t("quantizer.downsample.0.0.conv.weight"),
            self.t("quantizer.downsample.0.0.conv.bias"),
            stride=2,
        )
        x = self._convnext_block(x, "quantizer.downsample.0.1")
        x = self._causal_conv1d(
            x,
            self.t("quantizer.downsample.1.0.conv.weight"),
            self.t("quantizer.downsample.1.0.conv.bias"),
            stride=2,
        )
        x = self._convnext_block(x, "quantizer.downsample.1.1")
        x = self._window_transformer(x, "quantizer.pre_module", n_layers=8, window_size=self.window_size)

        return self._residual_vq(x, n_quantizers=n_quantizers)

    def decode_zq(self, z_q: "mx.array") -> "mx.array":
        if z_q.ndim != 3 or int(z_q.shape[1]) != 1024:
            raise ValueError(f"z_q must have shape (B, 1024, T), got {tuple(z_q.shape)}")

        x = z_q.astype(mx.float32)

        x = self._window_transformer(x, "quantizer.post_module", n_layers=8, window_size=self.window_size)
        x = self._causal_conv_transpose1d(
            x,
            self.t("quantizer.upsample.0.0.conv.weight"),
            self.t("quantizer.upsample.0.0.conv.bias"),
            stride=2,
        )
        x = self._convnext_block(x, "quantizer.upsample.0.1")
        x = self._causal_conv_transpose1d(
            x,
            self.t("quantizer.upsample.1.0.conv.weight"),
            self.t("quantizer.upsample.1.0.conv.bias"),
            stride=2,
        )
        x = self._convnext_block(x, "quantizer.upsample.1.1")

        x = self._causal_conv1d(x, self.t("decoder.model.0.conv.weight"), self.t("decoder.model.0.conv.bias"))
        x = self._decoder_block(x, "decoder.model.1", stride=8, n_t_layers=0)
        x = self._decoder_block(x, "decoder.model.2", stride=8, n_t_layers=0)
        x = self._decoder_block(x, "decoder.model.3", stride=4, n_t_layers=0)
        x = self._decoder_block(x, "decoder.model.4", stride=2, n_t_layers=0)
        x = _snake(x, self.t("decoder.model.5.alpha"))
        x = self._causal_conv1d(x, self.t("decoder.model.6.conv.weight"), self.t("decoder.model.6.conv.bias"))
        return mx.tanh(x)
