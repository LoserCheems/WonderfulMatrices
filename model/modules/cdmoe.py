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
from typing import Tuple
import torch
from torch import nn
from transformers.activations import ACT2FN
from einx import add as einx_add

class CDMoE(nn.Module):
    """Cross-Domain Mixture of Experts from 'Wonderful Matrices' paper."""

    def __init__(
        self, 
        d_model: int,
        act_fn: str,
        d_cross_domain: int,
        d_private_expert_retrieval: int,
        d_private_expert: int,
        n_experts: int,
        n_experts_heads: int,
        n_experts_per_head: int,
    ):
        super().__init__()
        self.hidden_dim = d_model
        self.act_fn = ACT2FN[act_fn]

        self.cross_domain_intermediate_dim = d_cross_domain
        self.private_expert_retrieval_dim = d_private_expert_retrieval
        self.private_expert_intermediate_dim = d_private_expert

        self.num_cdmmoe_experts = n_experts
        self.num_cdmmoe_heads = n_experts_heads
        self.num_cdmmoe_experts_per_head = n_experts_per_head

        # shared parameter up Linear
        self.shared_up_proj = nn.Linear(
            self.hidden_dim,
            self.cross_domain_intermediate_dim,
        )
        # shared parameter down Linear
        self.shared_down_proj = nn.Linear(
            self.cross_domain_intermediate_dim,
            self.private_expert_intermediate_dim,
        )

        # queries and keys for retrieval private experts
        self.queries = nn.Linear(
            self.private_expert_intermediate_dim,
            self.private_expert_retrieval_dim * self.num_cdmmoe_heads,
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
        self.down_embed = nn.Embedding(
            self.num_cdmmoe_experts,
            self.private_expert_intermediate_dim,
        )
        self.up_embed = nn.Embedding(
            self.num_cdmmoe_experts,
            self.hidden_dim,
        )  

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        # cross-domain
        hidden_states = self.shared_down_proj(self.act_fn(self.shared_up_proj(hidden_states)))

        # queries
        queries = self.queries(hidden_states)
        queries = queries.reshape(bsz, seq_len, 2, self.num_cdmmoe_heads, -1).permute(2, 0, 1, 3, 4)
        # get similarity with keys
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
        hidden_states = torch.einsum("b t d, b t h k d -> b t h k", hidden_states, down_embed)
        hidden_states = self.act_fn(hidden_states * scores.softmax(dim=-1))
        hidden_states = torch.einsum("b t h k, b t h k d -> b t d", hidden_states, up_embed)
        return hidden_states