import torch
from torch import nn
import torch.nn.functional as F
from .mlp import MLP, GatedMLP

class SEIMoE(nn.Module):

    def __init__(
        self,
        d_model: int,
        act_fn: str,
        d_ff: int,
        n_experts: int,
        n_experts_per_topk: int,
    ):
        super().__init__()
        self.hidden_size = d_model
        self.intermediate_size = d_ff
        self.num_experts = n_experts
        self.num_experts_per_topk = n_experts_per_topk

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MLP(self.hidden_size, act_fn, self.intermediate_size) for _ in range(self.num_experts)]
        )

        self.shared_expert = MLP(self.hidden_size, act_fn, self.intermediate_size)
        self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, hidden_size = hidden_states.shape
        # router_logits: (batch * sequence_length, num_experts)
        hidden_states = hidden_states.view(-1, hidden_size)
        router_logits = self.router(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_topk, dim=-1)

        final_hidden_states = torch.zeros(
            (bsz * seq_len, hidden_size),
            dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_size)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        # Compute the output of the shared expert
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = (final_hidden_states + shared_expert_output).reshape(bsz, seq_len, hidden_size)
        return final_hidden_states
