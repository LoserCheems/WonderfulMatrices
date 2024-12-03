from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers.activations import ACT2FN


class MLP(nn.Module):

    def __init__(
        self,
        d_model: int,
        act_fn: str,
        d_ff: int,
    ):
        super().__init__()
        self.hidden_dim = d_model
        self.act_fn = ACT2FN[act_fn]
        self.intermediate_dim = d_ff

        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_dim)
        self.down_proj = nn.Linear(self.intermediate_dim, self.hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.down_proj(self.act_fn(self.up_proj(hidden_states)))
        return hidden_states

class GatedMLP(nn.Module):

    def __init__(
        self,
        d_model: int,
        act_fn: str,
        d_ff: int,
    ):
        super().__init__()
        self.hidden_dim = d_model
        self.act_fn = ACT2FN[act_fn]
        self.intermediate_dim = d_ff

        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_dim)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_dim)
        self.down_proj = nn.Linear(self.intermediate_dim, self.hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return hidden_states
