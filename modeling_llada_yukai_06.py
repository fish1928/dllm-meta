# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)
from dataclasses import fields
from typing import List, Optional, Tuple, Union
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTN_AVAILABLE = True
except:
    FLEX_ATTN_AVAILABLE = False
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel
from transformers.cache_utils import Cache

from configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)

from tools_debug import jprint

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")

__all__ = [
    "LayerNormBase",
    "LayerNorm",
    "RMSLayerNorm",
    "GemmaRMSLayerNorm",
    "RotaryEmbedding",
    "Activation",
    "GELU",
    "ReLU",
    "SwiGLU",
    "LLaDABlock",
    "LLaDASequentialBlock",
    "LLaDAModel",
    "LLaDAOutput",
    "LLaDAGenerateOutput",
]


log = logging.getLogger(__name__)





@torch.compile()
def scaled_dot_product_attention(q, k, v, mask=None, attn_mask=None, dropout_p=0.0, is_causal=False):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
# end

class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights(
    config: ModelConfig,
    module: Union[nn.Linear, nn.Embedding],
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    """
    Initialize weights of a linear or embedding module.

    :param config: The model config.
    :param module: The linear or embedding submodule to initialize.
    :param d: The effective input dimensionality of the weights. This could be smaller than the actual dimensions
        for fused layers.
    :param layer_id: When set, the standard deviation for the "mitchell" method will be adjusted by
        ``1 / sqrt(2 * (layer_id + 1))``.
    """
    d = d if d is not None else config.d_model
    if config.init_fn == InitFnType.normal:
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff_value = config.init_cutoff_factor * std
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.mitchell:
        std = std_factor / math.sqrt(d)
        if layer_id is not None:
            std = std / math.sqrt(2 * (layer_id + 1))
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    elif config.init_fn == InitFnType.kaiming_normal:
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif config.init_fn == InitFnType.fan_in:
        std = std_factor / math.sqrt(d)
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.full_megatron:
        if type_of_module is None:
            raise RuntimeError(f"When using the {InitFnType.full_megatron} init, every module must have a type.")

        cutoff_factor = config.init_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        if type_of_module == ModuleType.in_module:
            # for att_proj (same as QKV), ff_proj
            std = config.init_std
        elif type_of_module == ModuleType.out_module:
            # for attn_out, ff_out
            std = config.init_std / math.sqrt(2.0 * config.n_layers)
        elif type_of_module == ModuleType.emb:
            # positional embeddings (wpe)
            # token embeddings (wte)
            std = config.init_std
        elif type_of_module == ModuleType.final_out:
            # final output (ff_out)
            std = config.d_model**-0.5
        else:
            raise RuntimeError(f"Unknown module type '{type_of_module}'")
        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
    else:
        raise NotImplementedError(config.init_fn)

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if config.init_fn == InitFnType.normal and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.n_layers))


def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    """
    Modify ``x`` in place to replace ``float("-inf")`` with the minimum value of the dtype when ``check_neg_inf``
    is ``True`` and to replace ``float("inf")`` with the maximum value of the dtype when ``check_pos_inf`` is ``True``.
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)


def activation_checkpoint_function(cfg: ModelConfig):
    preserve_rng_state = (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and (cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def _non_meta_init_device(config: ModelConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        elif config.layer_norm_type == LayerNormType.gemma_rms:
            return GemmaRMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-05,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            )
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x


class GemmaRMSLayerNorm(LayerNormBase):
    """
    Gemma RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return x * (1 + self.weight) + self.bias
            else:
                return x * (1 + self.weight)
        else:
            return x


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.rope_theta = config.rope_theta
        self.get_rotary_embedding(config.max_sequence_length, _non_meta_init_device(config))

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    # jinyu rotary
    def forward(self, q: torch.Tensor, k: torch.Tensor,
                block_end_index: Optional[torch.Tensor] = None,
                idx_q: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k
        # end

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)

            # Use index_select on the sequence dimension (dim=2)
            pos_sin_slice = pos_sin.index_select(2, idx_q)
            pos_cos_slice = pos_cos.index_select(2, idx_q)

            q_ = self.apply_rotary_pos_emb(pos_sin_slice, pos_cos_slice, q_)
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        # end

        return q_.type_as(q), k_.type_as(k)
    # end
# end


class Activation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> Activation:
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.silu:
            return cast(Activation, SiLU(inplace=False))
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0

class SiLU(nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0

class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


def alibi_attention_bias(seq_len: int, config: ModelConfig, device: torch.device) -> torch.FloatTensor:
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, 1, seq_len)

    # shape: (1, 1, seq_len, seq_len)
    alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, seq_len, 1)
    alibi_bias.abs_().mul_(-1)

    # shape: (n_heads,)
    m = torch.arange(1, config.n_heads + 1, dtype=torch.float, device=device)
    m.mul_(config.alibi_bias_max / config.n_heads)

    # shape: (1, n_heads, seq_len, seq_len)
    return alibi_bias * (1.0 / (2 ** m.view(1, config.n_heads, 1, 1)))  # type: ignore


class LLaDABlock(nn.Module):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0

        self._activation_checkpoint_fn = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout)

        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)

        # Activation function.
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        self.attn_out = nn.Linear(
            config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
        )

        # Feed-forward output projection.
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore

        # Rotary embeddings.
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config, self.__cache)

        self.flash_attn_func = None
        if config.flash_attention:
            try:
                from flash_attn import flash_attn_func  # type: ignore

                self.flash_attn_func = flash_attn_func
            except ModuleNotFoundError:
                pass
            # end
        # end

    # write a same-idx-for-rows version first
    # [current] = [refresh|denoising
    # @torch.compile()
    def concat_and_replace(self, matrix_origin, matrix_current, idx_current, shape_target): # (B, Hd, L, H)

        if matrix_origin.shape[-2] < shape_target[-2]:   # need patch
            length_patch = shape_target[-2] - matrix_origin.shape[-2]

            assert matrix_current.shape[-2] >= length_patch,\
                f'current shape should be >= patch shape, {matrix_current.shape[-2]} >= {length_patch}'
            matrix_patch = matrix_current[:, :, -length_patch:, :]   # TODO: check this

            matrix_origin = torch.cat([matrix_origin, matrix_patch], dim=-2)
        # end

        assert matrix_origin.shape[-2] == shape_target[-2],\
            f'origin shape should equal to target shape after patch, {matrix_origin.shape[-2]} == {shape_target[-2]}'

        # return matrix_origin.scatter(2, idx_current.view(1,1,-1,1).expand(1,32,-1,matrix_current.shape[-1]), matrix_current)
        matrix_origin[:, :, idx_current, :] = matrix_current
        return matrix_origin
    # end

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        init_weights(
            self.config,
            self.attn_out,
            d=self.config.d_model,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )
        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_features,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
        """
        if self.flash_attn_func is not None and attn_mask is None:
            r = self.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=False
            )
            return r.transpose(1, 2)
        else:
            # torch's sdpa doesn't support GQA, so we're doing this
            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

            # Modify: MDM set causal to False, and with no attn_mask.
            return scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
        # end if
    # end def


    def get_attn_score_avg(self, Q, K):
        # Q, K, V: (batch, heads, seq_len, head_dim)
        scale = Q.size(-1) ** -0.5
        # (batch, heads, seq_len, seq_len)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights_avg = attn_weights.mean(dim=1)
        return attn_weights_avg
    # end


    # jinyu attention
    '''[current] = [refresh|denoising]'''
    def attention(
        self,
        q_current: torch.Tensor,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        idx_current: Optional[torch.Tensor] = None,
        shape_target: Optional[Tuple[int, int, int]] = None

    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        B, T, C = q_current.size()  # batch size, sequence length, d_model
        dtype = k_current.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None: #self.q_norm: None, self.k_norm: None
            q_current = self.q_norm(q_current).to(dtype=dtype)
            k_current = self.k_norm(k_current).to(dtype=dtype)
        # end


        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        # self.config.n_heads: 32
        q_current = q_current.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k_current = k_current.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v_current = v_current.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        k_final, v_final = self.plugin_cache_past_kv.load()   

        max_replace_pos = k_final.shape[1]
        q_current_rotated, k_final_rotated = self.rotary_emb(
            q_current, k_final, max_replace_pos, idx_current
        )

        self.plugin_cache_attn.save()

        hidden = self._scaled_dot_product_attention(
            q_current_rotated, 
            k_final_rotated,
            v_final,
            attn_mask=None,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=False
        )

        hidden = hidden.transpose(1,2).contiguous().view(B,T,C)

        self.plugin_cache_past_kv.save()

        return self.attn_out(hidden)
    # end


    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        idx_current: Optional[torch.Tensor] = None,
        shape_target: Tuple[int, int, int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: ModelConfig, cache: BufferCache) -> LLaDABlock:
        if config.block_type == BlockType.sequential:
            # return LLaDASequentialBlock (layer_id, config, cache)
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")
        elif config.block_type == BlockType.llama:
            return LLaDALlamaBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class LLaDALlamaBlock(LLaDABlock):

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        head_dim = config.d_model // config.n_heads
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * head_dim
        v_proj_out_dim = config.effective_n_kv_heads * head_dim
        self.q_proj = nn.Linear(
            config.d_model,
            q_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device
        )

        self.k_proj = nn.Linear(
            config.d_model,
            k_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device
        )

        self.v_proj = nn.Linear(
            config.d_model,
            v_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device
        )

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
            device=config.init_device
        )

        # new add
        self.up_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
            device=config.init_device
        )
    # end __init__

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.up_proj, d=self.config.d_model, layer_id=None)  # new add
    # end
    

    # jinyu forward
    def forward(
        self,
        x_current: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        idx_current: Optional[torch.Tensor] = None,
        shape_target: Tuple[int, int, int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        x_normed_current = self.attn_norm(x_current) #x:torch.Size([2, 168, 4096])
        v = self.v_proj(x_normed_current) #v:torch.Size([2, 168, 4096])

        idx_current, x_current, x_normed_current, v = self.plugin_cache_vo.select_hidden(idx_current, x_current, x_normed_current, v)

        q_current = self.q_proj(x_normed_current) #q:torch.Size([B, budget, 4096])
        k = self.k_proj(x_normed_current) #k:torch.Size([B, budget, 4096])

        if self._activation_checkpoint_fn is not None:
            attn_current = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q_current, k, v, attention_bias=attention_bias,
                idx_current=idx_current,
                shape_target=shape_target
            )
        else:
            attn_current = self.attention(
                q_current, k, v,
                attention_bias=attention_bias,
                idx_current=idx_current,
                shape_target=shape_target
            )
        # end

        # Add attention scores.
        # shape: (B, T, C)

        x_final = x_current + self.dropout(attn_current)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x_final = x_final

        if self._activation_checkpoint_fn is not None:
            x_final = self._activation_checkpoint_fn(self.ff_norm, x_final)  # type: ignore
        else:
            x_final = self.ff_norm(x_final)
        # end

        x_final, x_final_up = self.ff_proj(x_final), self.up_proj(x_final) # new add

        if self._activation_checkpoint_fn is not None:
            x_final = self._activation_checkpoint_fn(self.act, x_final)  # type: ignore
        else:
            x_final = self.act(x_final)
        # end

        x_final = x_final * x_final_up # new add
        x_final = self.ff_out(x_final)
        x_final = self.dropout(x_final)
        x_final = og_x_final + x_final

        x_final = self.plugin_cache_vo.load_merge_and_update_hidden(x_final)
        return x_final
    # end
# end



class LLaDAOutput(NamedTuple):
    logits: torch.FloatTensor



class LLaDAGenerateOutput(NamedTuple):
    token_ids: torch.LongTensor
    """
    The generated token IDs, a tensor of shape `(batch_size, beam_size, max_steps)`.
    These do *not* include the original input IDs.
    """

    scores: torch.FloatTensor
    """
    The scores of the generated sequences, a tensor of shape `(batch_size, beam_size)`.
    """


class LLaDAModel(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

        # Validate config.
        if self.config.alibi and self.config.flash_attention:
            raise Exception("ALiBi is currently not supported with FlashAttention")

        if self.config.alibi and self.config.rope:
            raise Exception("ALiBi and RoPE are mutually exclusive")

        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise Exception("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise Exception("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )

        blocks = [LLaDABlock.build(i, config, self.__cache) for i in range(config.n_layers)]

        self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )
        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        # end

        self.__num_fwd_flops: Optional[int] = None
    # end

    def fill_plugin(self, klass_plugin):
        for _, block in enumerate(self.transformer.blocks):
            instance_plugin = klass_plugin()
            setattr(block, instance_plugin.get_plugin_name(), instance_plugin)
        # end
    # end

    def collect_plugins_with_index(self, klass_plugin):
        name_plugin = klass_plugin().get_plugin_name()
        plugins= []

        for idx_block, block in enumerate(self.transformer.blocks):
            instance_plugin = getattr(block, name_plugin)
            plugins.append((idx_block, instance_plugin))
        # end

        return plugins
    # end

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        if self.config.block_group_size != 1:
            for block_group in self.transformer.block_groups:
                block_group.set_activation_checkpointing(strategy)
            # end for
        else:
            for block in self.transformer.blocks:
                block.set_activation_checkpointing(strategy)
            # end for 
        # end if
    # end def

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def reset_parameters(self):
        log.info("Initializing model parameters...")
        # Top-level embeddings / linear layers.
        init_weights(
            self.config,
            self.transformer.wte,  # type: ignore
            std_factor=(0.5 * math.sqrt(self.config.d_model)) if self.config.scale_logits else 1.0,
            type_of_module=ModuleType.emb,
        )
        if hasattr(self.transformer, "wpe"):
            init_weights(self.config, self.transformer.wpe, type_of_module=ModuleType.emb)  # type: ignore

        # Top-level layer norm.
        self.transformer.ln_f.reset_parameters()  # type: ignore

        # Output weights.
        if hasattr(self.transformer, "ff_out"):
            init_weights(self.config, self.transformer.ff_out, type_of_module=ModuleType.final_out)  # type: ignore

        # Let the blocks handle themselves.
        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()

    def get_alibi_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (alibi_bias := self.__cache.get("alibi_attention_bias")) is not None and alibi_bias.shape[
            -1
        ] >= seq_len:
            if alibi_bias.device != device:
                alibi_bias = alibi_bias.to(device)
                self.__cache["alibi_attention_bias"] = alibi_bias
            return alibi_bias
        with torch.autocast(device.type, enabled=False):
            alibi_bias = alibi_attention_bias(seq_len, self.config, device)
        self.__cache["alibi_attention_bias"] = alibi_bias
        return alibi_bias
    # end


    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        idx_current: Optional[torch.Tensor] = None,
        shape_target: Tuple[int, int, int] = None,
        last_logits_only: bool = False,
    ) -> LLaDAOutput:

        # Add Basic MDM Model config check
        assert not self.config.alibi, "Alibi length extrapolation is not supported for MDM."
        assert self.config.rope, "Rope must be used in Llama-Encoder for MDM."
        # assert (past_key_values is None and not use_cache), "The kvcache is not suppotred for MDM."

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]

        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore

        if self.config.input_emb_norm:
            x = x * (self.config.d_model**0.5)

        x = self.transformer.emb_drop(x)  # type: ignore

        if attention_mask is not None and 0.0 in attention_mask:
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
        else:
            attention_mask = None

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                x = block(
                    x,attention_bias=attention_bias,
                    idx_current=idx_current,
                    shape_target=shape_target,
                )

            # end for block_id
        else:
            raise NotImplemented('block is removed from yukai llada version')
        # end if

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            x = x[:, -1, :].unsqueeze(1)
        # end

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        else:
            logits = self.transformer.ff_out(x)  # type: ignore
        # end

        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))
        # end

        return LLaDAOutput(logits=logits)  # type: ignore[arg-type]
    # end def forward


def create_model_config_from_pretrained_config(config: LLaDAConfig):
    kwargs = {}
    for field in fields(ModelConfig):
        kwargs[field.name] = getattr(config, field.name)

    model_config = ModelConfig(**kwargs)
    return model_config


class LLaDAModelLM(PreTrainedModel):
    """
    Extremely barebones HF model wrapper.
    """

    config_class = LLaDAConfig
    base_model_prefix = "model"
    _no_split_modules = ["LLaDABlock", "LLaDASequentialBlock", "LLaDALlamaBlock"]

    def __init__(self, config: LLaDAConfig, model: Optional[LLaDAModel] = None, init_params: bool = False):
        super().__init__(config)

        if not model:
            model_config = create_model_config_from_pretrained_config(config)
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            self.model = LLaDAModel(model_config, init_params=init_params)
        else:
            self.model = model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        idx_current: Optional[torch.Tensor] = None,
        shape_target: Tuple[int, int, int] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if output_attentions:
            raise ValueError("output_attentions is not yet supported in LLaDA")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # import pdb; pdb.set_trace()
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            idx_current=idx_current,
            shape_target=shape_target,
        )
            
        # import pdb; pdb.set_trace()
        logits = outputs.logits

        loss = None
        if labels is not None:
            import warnings
            warnings.warn("Note that for LLaDA, you cannot calculate the loss here.", UserWarning)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            logits=logits,
        )
    
    def fill_plugin(self, klass_plugin):
        self.model.fill_plugin(klass_plugin)
        return self
    # end

    def collect_plugins_with_index(self, klass_plugin):
        return self.model.collect_plugins_with_index(klass_plugin)
    # end

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple]] = None, **kwargs
    ):
        if past_key_values:
            # This is because we want the model to only process the last generated token.
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}

        model_inputs.update(kwargs)
        return model_inputs


    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        if self.config.weight_tying:
            return self.model.transformer.wte
        else:
            return self.model.transformer.ff_out

    def set_output_embeddings(self, value: torch.nn.Module):
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.model.transformer.ff_out = value

    def tie_weights(self):
        if self.config.weight_tying:
            self.model.transformer.ff_out = self.model.transformer.wte

# Register the model so that it is available for transformer pipelines, auto-loading, etc.
AutoModel.register(LLaDAConfig, LLaDAModelLM)