# coding=utf-8
# Copyright 2024 Jingze Shi, Bingheng Wu and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on the Wonderful Matrices paper implementation.
#
#     https://arxiv.org/abs/2412.11834
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
"""PyTorch Doge Vision model."""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    # is_einx_available,
    logging,
    replace_return_docstrings,
)
from .configuration_doge_vision import DogeConfig

try:
    from einx import add as einx_add
except ImportError:
    einx_add = None


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DogeConfig"


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class RotaryEmbedding(nn.Module):
    def __init__(self, config: Optional[DogeConfig] = None):
        super().__init__()
        self.rope_kwargs = {}

        if config.rope_scaling is None:
            self.rope_type = "default"
        else:
            self.rope_type = config.rope_scaling
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.base = config.rope_theta

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """
    Rotates half the hidden dims of the input.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_QK_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DogeInnerFuncAttn(nn.Module):
    """Inner Function Attention from 'Wonderful Matrices' paper."""

    def __init__(self, config: DogeConfig, layer_idx: Optional[int] = None):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_dropout = config.attention_dropout

        # for accuracy of attention scores, we do not use GQA
        self.attention_head_dim = self.hidden_dim // self.num_attention_heads
        self.num_inner_values = config.num_inner_values
        self.num_inner_value_heads = config.num_inner_value_heads
        self.num_value_per_head = config.num_value_per_head
        self.inner_values_retrieval_dim = config.inner_values_retrieval_size
        
        # Q and K projections
        self.q_proj = nn.Linear(
            self.hidden_dim,
            self.attention_head_dim * self.num_attention_heads,
            bias=config.hidden_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_dim,
            self.attention_head_dim * self.num_attention_heads,
            bias=config.hidden_bias,
        )

        # dynamic mask for the QK^T attention score matrix
        self.dynamic_mask = nn.Parameter(
            torch.round(torch.ones(self.num_attention_heads, config.max_position_embeddings))
        )

        # queries and keys for retrieval V
        self.v_queries = nn.Linear(
            self.hidden_dim,
            self.num_inner_value_heads * self.inner_values_retrieval_dim,
            bias=config.hidden_bias,
        )
        self.v_keys = nn.Parameter(
            torch.zeros(
                self.num_inner_value_heads,
                self.inner_values_retrieval_dim,
                self.num_inner_values,
            )
        )

        # V for inner function
        self.v_embed = nn.Embedding(
            self.num_inner_values,
            self.hidden_dim,
        )

        self.o_proj = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=config.hidden_bias,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor = None,
        input_tensor: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        past_key_values: Cache = None,
        output_attentions: bool = False,
    ):
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # in case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position_and_dynamic_mask(
            attention_mask=attention_mask,
            dynamic_mask=self.dynamic_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position_and_dynamic_mask(
        attention_mask: torch.Tensor = None,
        dynamic_mask: torch.Tensor = None,
        sequence_length: int = None,
        target_length: int = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
        cache_position: torch.Tensor = None,
        batch_size: int = None,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            dynamic_mask (`torch.Tensor`):
                A 2D dynamic mask of shape `(num_heads, max_position_embeddings)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            num_heads = 1 if dynamic_mask is None else dynamic_mask.size(0)
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, num_heads, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                attention_mask = attention_mask[:, None, None, :].expand(-1, num_heads, 1, -1)
                if dynamic_mask is not None:
                    dynamic_mask = dynamic_mask[None, :, None, :mask_length].expand(batch_size, -1, 1, -1)
                    attention_mask = attention_mask.clone() * dynamic_mask

                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask == 0, min_dtype
                )

        return causal_mask

    def inner_func(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Each value can share weights with other values to increase the expressive power
        """
        bsz, seq_len, _ = hidden_states.shape

        v_queries = self.v_queries(hidden_states)
        v_queries = v_queries.view(bsz, seq_len, self.num_inner_value_heads, -1).transpose(1, 2)
        sim = torch.matmul(v_queries, self.v_keys)
        v_embed = self.v_embed(sim.topk(k=self.num_value_per_head, dim=-1).indices)
        # b h t k d -> b t d
        v = hidden_states * v_embed.sum(dim=-2).sum(dim=-3)
        return v


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Cache]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.inner_func(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_dim).transpose(
            1, 2
        )
        key_states = key_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_dim).transpose(
            1, 2
        )

        cos, sin = position_embeddings
        query_states, query_states = apply_QK_rotary_pos_emb(query_states, query_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # compute attention scores matrix
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.attention_head_dim)

        # add mask to attention scores
        causal_mask = self._update_causal_mask(attention_mask, hidden_states, cache_position, past_key_value)
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
       
        # apply attention scores to value states
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class DogeSdpaInnerFuncAttn(DogeInnerFuncAttn):
    """
    Doge Inner Function Attention module using torch.nn.functional.scaled_dot_product_attention.
    This module inherits from `DogeInnerFuncAttn` as the weights of the module stays untouched.
    The only changes are on the forward pass to adapt to SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Cache]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.inner_func(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, query_states = apply_QK_rotary_pos_emb(query_states, query_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        causal_mask = self._update_causal_mask(attention_mask, hidden_states, cache_position, past_key_value)
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


DOGE_ATTENTION_CLASSES = {
    "eager": DogeInnerFuncAttn,
    "sdpa": DogeSdpaInnerFuncAttn,
}


class DogeCDMoE(nn.Module):
    """Cross-Domain Mixture of Experts from 'Wonderful Matrices' paper."""

    def __init__(self, config: DogeConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.intermediate_dim = config.intermediate_size

        self.private_expert_retrieval_dim = config.private_expert_retrieval_size
        self.num_cdmmoe_experts = config.num_cdmmoe_experts
        self.num_cdmmoe_heads = config.num_cdmmoe_heads
        self.num_cdmmoe_experts_per_head = config.num_cdmmoe_experts_per_head

        # cross domain
        self.up_proj = nn.Linear(
            self.hidden_dim,
            self.intermediate_dim,
            bias=config.hidden_bias,
        )
        self.down_proj = nn.Linear(
            self.intermediate_dim,
            self.hidden_dim,
            bias=config.hidden_bias,
        )

        # queries and keys for retrieval private experts
        self.queries = nn.Linear(
            self.hidden_dim,
            self.num_cdmmoe_heads * self.private_expert_retrieval_dim,
            bias=False,
        )
        self.num_keys = int(math.sqrt(self.num_cdmmoe_experts))
        self.keys = nn.Parameter(
            torch.zeros(
                self.num_cdmmoe_heads,
                self.num_keys,
                2,
                self.private_expert_retrieval_dim // 2,
            )
        )

        # private experts
        self.down_embed  = nn.Embedding(
            self.num_cdmmoe_experts,
            self.hidden_dim,
        )
        self.up_embed = nn.Embedding(
            self.num_cdmmoe_experts,
            self.hidden_dim,
        )
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        # get similarity with queries and keys
        queries = self.queries(hidden_states)
        queries = queries.view(bsz, seq_len, 2, self.num_cdmmoe_heads, -1).permute(2, 0, 1, 3, 4)
        sim = torch.einsum("p b t h n, h k p n -> p b t h k", queries, self.keys)

        # get expert scores and indices with the highest similarity
        (scores_x, scores_y), (indices_x, indices_y) = sim.topk(self.num_cdmmoe_experts_per_head, dim=-1)
        if einx_add is not None:
            all_scores = einx_add("... i, ... j -> ... (i j)", scores_x, scores_y)
            all_indices = einx_add("... i, ... j -> ... (i j)", indices_x * self.num_keys, indices_y)
        else:
            all_scores = scores_x.unsqueeze(-1) + scores_y.unsqueeze(-2)
            all_scores = all_scores.view(*scores_x.shape[:-1], -1)
            all_indices = (indices_x.unsqueeze(-1) * self.num_keys) + indices_y.unsqueeze(-2)
            all_indices = all_indices.view(*indices_x.shape[:-1], -1)
        scores, pk_indices = all_scores.topk(self.num_cdmmoe_experts_per_head, dim=-1)
        indices = all_indices.gather(-1, pk_indices)

        # get related expert embeddings based on indices
        down_embed = self.down_embed(indices)
        up_embed = self.up_embed(indices)

        # efficient retrieval of private experts
        experts_weights = self.act_fn(torch.einsum("b t d, b t h k d -> b t h k", hidden_states, down_embed) * scores.softmax(dim=-1))
        experts_states = torch.einsum("b t h k, b t h k d -> b t d", experts_weights, up_embed)

        # mix with shared parameters of cross domain
        hidden_states = self.down_proj(self.act_fn(self.up_proj(hidden_states)))
        hidden_states = hidden_states + experts_states
        return hidden_states


class DogeDecoderLayer(nn.Module):
    def __init__(self, config: DogeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_dropout = config.hidden_dropout

        self.in_attn_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = DOGE_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.in_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = DogeCDMoE(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        # sequence transformation
        residual = hidden_states
        hidden_states = self.in_attn_layernorm(hidden_states)
        hidden_states, present_key_value = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        self_attn_weights = None
        hidden_states = F.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        hidden_states = residual + hidden_states

        # state transformation
        residual = hidden_states
        hidden_states = self.in_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@add_start_docstrings("The bare Doge Model outputting raw hidden-states without any specific head on top.")
class DogePreTrainedModel(PreTrainedModel):
    config_class = DogeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DogeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


DOGE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings("The bare Doge Model outputting raw hidden-states without any specific head on top.")
class DogeModel(DogePreTrainedModel):
    def __init__(self, config: DogeConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.rotary_emb = RotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [DogeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DOGE_INPUTS_DOCSTRING)
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    """Move to DogeInnerFuncAttn"""
    # def _update_causal_mask(
    #     self,
    #     attention_mask: torch.Tensor,
    #     input_tensor: torch.Tensor,
    #     cache_position: torch.Tensor,
    #     past_key_values: Cache,
    #     output_attentions: bool,
    # ):
    #     # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    #     # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    #     # to infer the attention mask.
    #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    #     using_static_cache = isinstance(past_key_values, StaticCache)

    #     dtype, device = input_tensor.dtype, input_tensor.device
    #     sequence_length = input_tensor.shape[1]
    #     if using_static_cache:
    #         target_length = past_key_values.get_max_cache_shape()
    #     else:
    #         target_length = (
    #             attention_mask.shape[-1]
    #             if isinstance(attention_mask, torch.Tensor)
    #             else past_seen_tokens + sequence_length + 1
    #         )

    #     # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
    #     causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
    #         attention_mask,
    #         sequence_length=sequence_length,
    #         target_length=target_length,
    #         dtype=dtype,
    #         device=device,
    #         cache_position=cache_position,
    #         batch_size=input_tensor.shape[0],
    #     )

    #     return causal_mask

    # @staticmethod
    # def _prepare_4d_causal_attention_mask_with_cache_position(
    #     attention_mask: torch.Tensor,
    #     sequence_length: int,
    #     target_length: int,
    #     dtype: torch.dtype,
    #     device: torch.device,
    #     cache_position: torch.Tensor,
    #     batch_size: int,
    #     **kwargs,
    # ):
    #     """
    #     Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    #     `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    #     Args:
    #         attention_mask (`torch.Tensor`):
    #             A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
    #             `(batch_size, 1, query_length, key_value_length)`.
    #         sequence_length (`int`):
    #             The sequence length being processed.
    #         target_length (`int`):
    #             The target length: when generating with static cache, the mask should be as long as the static cache,
    #             to account for the 0 padding, the part of the cache that is not filled yet.
    #         dtype (`torch.dtype`):
    #             The dtype to use for the 4D attention mask.
    #         device (`torch.device`):
    #             The device to plcae the 4D attention mask on.
    #         cache_position (`torch.Tensor`):
    #             Indices depicting the position of the input sequence tokens in the sequence.
    #         batch_size (`torch.Tensor`):
    #             Batch size.
    #     """
    #     if attention_mask is not None and attention_mask.dim() == 4:
    #         # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
    #         causal_mask = attention_mask
    #     else:
    #         min_dtype = torch.finfo(dtype).min
    #         causal_mask = torch.full(
    #             (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
    #         )
    #         if sequence_length != 1:
    #             causal_mask = torch.triu(causal_mask, diagonal=1)
    #         causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    #         causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    #         if attention_mask is not None:
    #             causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
    #             mask_length = attention_mask.shape[-1]
    #             padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
    #             padding_mask = padding_mask == 0
    #             causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
    #                 padding_mask, min_dtype
    #             )

    #     return causal_mask


class DogePatchEmbedding(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` of shape `(batch_size, 1, hidden_size)` to be consumed by a Transformer.
    """

    def __init__(self, config: DogeConfig):
        super().__init__()
        image_size, patch_size, num_channels = config.image_size, config.patch_size, config.num_channels

        image_size = image_size if isinstance(image_size, list) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, list) else (patch_size, patch_size)
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.hidden_size = config.hidden_size

        self.proj = nn.Conv2d(num_channels, self.hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                f"Input image should have {self.num_channels} number of channels, but got {num_channels} instead."
            )
        
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image should have size {self.image_size}, but got {height}*{width} instead."
            )
        
        image_embedding = self.proj(pixel_values).flatten(2).transpose(1, 2)
        return image_embedding


class DogeForCausalVLM(DogePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DogeConfig):
        super().__init__(config)
        self.config = config
        self.word_embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pixel_embed = DogePatchEmbedding(config)
        self.model = DogeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embed

    def set_input_embeddings(self, value):
        self.word_embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(DOGE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            inputs_embeds = self.word_embed(input_ids)
        if pixel_values is not None:
            pixel_embeds = self.pixel_embed(pixel_values)
            inputs_embeds = torch.cat([inputs_embeds, pixel_embeds], dim=1)

        # decoder output consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        # only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class DogeObjectDetectionOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class DogeForObjectDetection(DogePreTrainedModel):
    def __init__(self, config: DogeConfig):
        super().__init__(config)
        self.config = config

        # pixel embedding
        self.pixel_embed = DogePatchEmbedding(config)

        # model backbone
        self.model = DogeModel(config)
        self.num_labels = config.num_labels

        # classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels + 1)

        # detection head
        self.detector = nn.Linear(config.hidden_size, 4)

        # Initialize weights and apply final processing
        self.post_init()
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, DogeObjectDetectionOutputWithPast]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: `'class_labels'` and `'boxes'` (the class labels and bounding boxes of an image in the
            batch respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding
            boxes in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image,
            4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModelForObjectDetection
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
        >>> model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
        ...     0
        ... ]

        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected remote with confidence 0.991 at location [46.48, 72.78, 178.98, 119.3]
        Detected remote with confidence 0.908 at location [336.48, 79.27, 368.23, 192.36]
        Detected cat with confidence 0.934 at location [337.18, 18.06, 638.14, 373.09]
        Detected cat with confidence 0.979 at location [10.93, 53.74, 313.41, 470.67]
        Detected remote with confidence 0.974 at location [41.63, 72.23, 178.09, 119.99]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is not None:
            inputs_embeds = self.pixel_embed(pixel_values)
        
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Take the final hidden states of the detection tokens
        hidden_states = hidden_states[:, -self.config.num_detection_tokens:, :]

        logits = self.classifier(hidden_states)
        pred_boxes = self.detector(hidden_states).sigmoid()

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            outputs_class, outputs_coord = None, None
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits, labels, self.device, pred_boxes, self.config, outputs_class, outputs_coord
            )
        
        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            return ((loss, loss_dict) + output) if loss is not None else output
    
        return DogeObjectDetectionOutputWithPast(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        




