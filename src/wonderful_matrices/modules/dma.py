# coding=utf-8
# Copyright 2024 Jingze Shi. All rights reserved.
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

import math
from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Cache, StaticCache


class DMA(nn.Module):
    """Dynamic Masked Attention from 'Wonderful Matrices' paper."""

    def __init__(
        self, 
        d_model: int,
        n_heads: int,
        max_position_embeddings: int,
        layer_idx: Optional[int] = None
    ):
        super().__init__()

        self.layer_idx = layer_idx

        self.hidden_dim = d_model
        self.num_attention_heads = n_heads
        self.attention_head_dim = self.hidden_dim // self.num_attention_heads
     
        # Q K V O projections
        self.q_proj = nn.Linear(
            self.hidden_dim,
            self.attention_head_dim * self.num_attention_heads,
        )
        self.k_proj = nn.Linear(
            self.hidden_dim,
            self.attention_head_dim * self.num_attention_heads,
        )
        # dynamic mask for the QK^T attention score matrix
        self.A = nn.Parameter(
            torch.ones(self.num_attention_heads)
        )
        self.dt_proj = nn.Linear(
            self.num_attention_heads * self.attention_head_dim,
            self.num_attention_heads,
        )
        self.v_proj = nn.Linear(
            self.hidden_dim,
            self.attention_head_dim * self.num_attention_heads,
        )
        self.o_proj = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
        )

    def update_causal_mask(
        self,
        attention_mask: torch.Tensor = None,
        input_tensor: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        past_key_values: Cache = None,
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
        causal_mask = self.prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask=attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        return causal_mask

    @staticmethod
    def prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor = None,
        sequence_length: int = None,
        target_length: int = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
        cache_position: torch.Tensor = None,
        batch_size: int = None,
        **kwargs,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
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
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                attention_mask = attention_mask[:, None, None, :].expand(-1, 1, 1, -1)
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask == 0, min_dtype
                )

        return causal_mask
    
    def prepare_dynamic_mask(
        self,
        hidden_states: torch.Tensor,
        dynamic_mask: torch.Tensor,
        dynamic_mask_ratio: float = 0.0,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Combine `dynamic_mask` with `attention_mask` to generate the final `attn_mask`.

        Args:
            hidden_states (`torch.Tensor`): The input hidden_states, used to determine the minimum value of the current input precision.
            dynamic_mask (`torch.Tensor`): dynamic mask of shape `(batch_size, num_heads, key_sequence_length)`.
            dynamic_mask_ratio (`float`, *optional*): Ratio from 0.0 to 1.0 used to control the proportion of the dynamic mask filled with the minimum value.
            attention_mask (`torch.Tensor`, *optional*): attention mask of shape `(batch_size, 1, query_sequence_length, key_sequence_length)`.
        """
        min_type = torch.finfo(hidden_states.dtype).min
        attn_mask = dynamic_mask[:, :, None, :]
        if 0.0 < dynamic_mask_ratio < 1.0:
            num_dynamic_mask = int(attn_mask.shape[-1] * dynamic_mask_ratio)
            if num_dynamic_mask > 0:
                rate_value = torch.kthvalue(attn_mask, num_dynamic_mask, dim=-1, keepdim=True).values
                attn_mask = attn_mask.masked_fill(attn_mask < rate_value, min_type)
        if attention_mask is not None:
            attn_mask = attn_mask.masked_fill(attention_mask[:, :, :, : hidden_states.shape[-2]] == min_type, min_type)
        return attn_mask

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
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_dim).transpose(
            1, 2
        )
        key_states = key_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_dim).transpose(
            1, 2
        )

        # RoPE is not mandatory, you can use other positional embeddings
        # cos, sin = position_embeddings
        # query_states, key_states = apply_QK_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        
        # calculate dynamic mask from value_states
        dt_states = self.dt_proj(value_states.transpose(1, 2).reshape(bsz, value_states.shape[-2], -1))
        dynamic_mask = torch.exp(self.A * F.softplus(dt_states)).transpose(-1, -2)

        # compute attention scores matrix
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.attention_head_dim)

        # add mask to attention scores
        attn_mask = self.prepare_dynamic_mask(
            hidden_states=hidden_states,
            dynamic_mask=dynamic_mask,
            attention_mask=attention_mask,
        )
        attn_weights = attn_weights + attn_mask

        # upcast attention scores to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # apply attention scores to value states
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value
