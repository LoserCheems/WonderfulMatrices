# coding=utf-8
# Copyright 2024 Jingze Shi.    All rights reserved.
#
# This code is based on the Wonderful Matrices paper implementation.
#
#     https://arxiv.org/abs/2407.16958
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

import math
from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Cache, StaticCache


class InnerFuncAttn(nn.Module):
    """Inner Function Attention from 'Wonderful Matrices' paper."""

    def __init__(
        self, 
        d_model: int,
        n_heads: int,
        n_inner_values: int,
        d_inner_values_retrieval: int,
        max_position_embeddings: int,
        layer_idx: Optional[int] = None
    ):
        super().__init__()

        self.layer_idx = layer_idx

        self.hidden_dim = d_model
        self.num_attention_heads = n_heads

        # for accuracy of attention scores, we do not use GQA
        self.attention_head_dim = self.hidden_dim // self.num_attention_heads
        self.num_inner_values = n_inner_values
        self.inner_values_retrieval_dim = d_inner_values_retrieval

        self.q_proj = nn.Linear(
            self.hidden_dim,
            self.attention_head_dim * self.num_attention_heads,
        )
        self.k_proj = nn.Linear(
            self.hidden_dim,
            self.attention_head_dim * self.num_attention_heads,
        )
        self.dynamic_mask = nn.Parameter(
            torch.round(torch.ones(self.num_attention_heads, max_position_embeddings))
        )
        self.v_queries = nn.Linear(
            self.hidden_dim,
            self.inner_values_retrieval_dim,
        )
        self.v_keys = nn.Parameter(
            torch.zeros(
                self.num_inner_values,
                self.inner_values_retrieval_dim,
            )
        )
        self.v_embed = nn.Embedding(
            self.num_inner_values,
            self.hidden_dim,
        )
        self.o_proj = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor = None,
        input_tensor: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        past_key_values: Cache = None,
        output_attentions: bool = False,
    ):
        # for SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
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
        causal_mask = self.prepare_4d_causal_attention_mask_with_cache_position_and_dynamic_mask(
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
    def prepare_4d_causal_attention_mask_with_cache_position_and_dynamic_mask(
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
            if cache_position is not None:
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
        v_queries = self.v_queries(hidden_states)
        sim = torch.matmul(v_queries, self.v_keys.transpose(-1, -2))
        v_embed = self.v_embed(sim.topk(k=1, dim=-1).indices)
        v = hidden_states * v_embed.sum(dim=-2)
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
        bsz, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.inner_func(hidden_states)

        query_states = query_states.view(bsz, seq_len, self.num_attention_heads, self.attention_head_dim).transpose(
            1, 2
        )
        key_states = key_states.view(bsz, seq_len, self.num_attention_heads, self.attention_head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(bsz, seq_len, self.num_attention_heads, self.attention_head_dim).transpose(
            1, 2
        )

        # RoPE is not mandatory, you can use other positional embeddings
        # cos, sin = position_embeddings
        # query_states, query_states = apply_QK_rotary_pos_emb(query_states, query_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.attention_head_dim)

        causal_mask = self._update_causal_mask(attention_mask, hidden_states, cache_position, past_key_value)
        # no matter the length, we just slice it
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (
            bsz,
            self.num_attention_heads,
            seq_len,
            self.attention_head_dim,
        ):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_attention_heads, seq_len, self.attention_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value