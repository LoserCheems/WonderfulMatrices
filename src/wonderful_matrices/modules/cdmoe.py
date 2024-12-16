# coding=utf-8
# Copyright 2024 Jingze Shi. All rights reserved.
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
from typing import Tuple
import torch
from torch import nn
from transformers.activations import ACT2FN

try:
    from einx import add as einx_add
except ImportError:
    einx_add = None


class CDMoE(nn.Module):
    """Cross Domain Mixture of Experts from 'Wonderful Matrices' paper."""

    def __init__(
        self, 
        d_model: int,
        act_fn: str,
        d_cd: int,
        d_expert_retrieval: int,
        n_experts: int,
        n_experts_heads: int,
        n_experts_per_head: int,
    ):
        super().__init__()
        self.hidden_dim = d_model
        self.act_fn = ACT2FN[act_fn]
        self.intermediate_dim = d_cd

        self.expert_retrieval_dim = d_expert_retrieval
        self.num_cdmmoe_experts = n_experts
        self.num_cdmmoe_heads = n_experts_heads
        self.num_cdmmoe_experts_per_head = n_experts_per_head
        self.num_keys = int(math.sqrt(self.num_cdmmoe_experts))

        # cross domain
        self.gate_proj = nn.Linear(
            self.hidden_dim,
            self.intermediate_dim,
        )
        self.up_proj = nn.Linear(
            self.hidden_dim,
            self.intermediate_dim,
        )
        self.down_proj = nn.Linear(
            self.intermediate_dim,
            self.hidden_dim,
        )

        # queries and keys for retrieval experts
        self.queries = nn.Linear(
            self.hidden_dim,
            self.expert_retrieval_dim * self.num_cdmmoe_heads,
            bias=False,
        )
        self.keys = nn.Parameter(
            torch.zeros(
                self.num_cdmmoe_heads,
                self.num_keys,
                2,
                self.expert_retrieval_dim // 2,
            )
        )

        # experts
        self.down_embed = nn.Embedding(
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

        # get experts with the highest similarity
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
        down_embed = self.down_embed(indices)
        up_embed = self.up_embed(indices)

        # mix experts states with cross domain states
        experts_weights = torch.einsum("b t d, b t h k d -> b t h k", hidden_states, down_embed)
        experts_weights = self.act_fn(experts_weights) * scores.softmax(dim=-1)
        experts_states = torch.einsum("b t h k, b t h k d -> b t d", experts_weights, up_embed)
        hidden_states = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        hidden_states = hidden_states + experts_states
        return hidden_states