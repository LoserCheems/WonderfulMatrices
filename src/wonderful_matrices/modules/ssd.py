from typing import Optional, Tuple
import torch
from torch import nn


class SSD(nn.Module):
    """State Space Duality from 'Transformers are SSMs' paper.
    """

    def __init__(
        self, 
        d_model: int,
        n_heads: str,
        d_state: int,
        n_groups: int,
        chunk_len: int,
        layer_idx: Optional[int] = None
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_state = d_state
        self.n_groups = n_groups
        self.chunk_len = chunk_len

        A = torch.arange(1, self.n_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.B_proj = nn.Linear(d_model, self.n_groups * self.d_state)
        self.C_proj = nn.Linear(d_model, self.n_groups * self.d_state)
        self.x_proj = nn.Linear(d_model, d_model)
        self.dt_proj = nn.Linear(d_model, self.n_heads)
        self.D = nn.Parameter(torch.ones(self.n_heads))

        self.out_proj = nn.Linear(d_model, d_model)

    def pad_tensor_by_size(
        self, 
        input_tensor: torch.Tensor, 
        pad_size: int
    ) -> torch.Tensor:
        # Pad seq_len to be multiple of chunk_size
        pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)
        return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)

    def reshape_into_chunks(
        self,
        input_tensor: torch.Tensor,
        pad_size: int,
        chunk_size: int
    ) -> torch.Tensor:
        """
        Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and simultaneously splitting it into chunk sequences.
        """
        # b t ... -> b (l c) ...
        input_tensor = self.pad_tensor_by_size(input_tensor, pad_size)
        if len(input_tensor.shape) == 3:
            return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
        else:
            return input_tensor.reshape(
                input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
            )

    def segment_sum(
        self, 
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
        """
        chunk_size = input_tensor.size(-1)
        # 1. expand input tensor to have an additional dimension and repeat along that dimension
        # [..., chunk_size] -> [..., chunk_size, chunk_size]
        input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
        # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
        mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
        input_tensor = input_tensor.masked_fill(~mask, 0)
        # 3. compute actual cumsum
        tensor_segsum = torch.cumsum(input_tensor, dim=-2)

        # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
        mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
        tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
        return tensor_segsum

    def ssd_algorithm(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        chunk_len: int,
        D: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bsz, seq_len, n_heads, d_head = x.size()
        pad_size = (chunk_len - seq_len % chunk_len) % chunk_len
        if D is not None:
            D_residual = D[..., None] * self.pad_tensor_by_size(x, pad_size)
        
        # Discretize x and A
        x = x * dt[..., None]
        A = A.to(x.dtype) * dt
    
        # Rearrange into blocks/chunks
        x, A, B, C = [self.reshape_into_chunks(t, pad_size, chunk_len) for t in (x, A, B, C)]

        # Compute cumulative sum of A
        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)

        # 1. Compute the output for each intra-chunk (diagonal blocks)
        # This is the analog of a causal mask
        L = torch.exp(self.segment_sum(A))
        
        # First, contraction of C and B to get G (attention-weights like)
        G = (C[:, :, :, None, :, :] * B[:, :, None, :, : ,:]).sum(dim=-1)

        # Step 2: Compute M, equivalent to applying attention mask to weights
        M = (G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]).sum(dim=-1)

        # Step 3: Compute Y_diag (apply to values)
        Y_diag = (M[..., None] * x[:, :, None]).sum(3)

        # (right term of low-rank factorization of off-diagonal blocks; B terms)
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        B_decay_contraction = B * decay_states.permute(0, 2, 3, 1)[..., None]
        # permute back B * decay states
        states = (B_decay_contraction.permute(0, 1, 3, 2, 4)[..., None]  * x.permute(0, 1, 3, 2, 4)[..., None, :]).sum(dim=3).permute(0, 1, 2, 4, 3)
        previous_states = torch.zeros_like(states[:, :1])
        states = torch.cat([previous_states, states], dim=1)
        decay_chunk = torch.exp(self.segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
        states_permuted = states.permute(0, 2, 1, 3, 4)
        result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(dim=2)
        new_states = result.permute(0, 2, 1, 3, 4)
        states = new_states[:, :-1]

        # Compute state -> output conversion per chunk
        # (left term of low-rank factorization of off-diagonal blocks; C terms)
        # compute Yoff
        C_times_states = (C[..., None, :] * states[:, :, None, ...])
        Y_off = (C_times_states.sum(-1) * (torch.exp(A_cumsum).permute(0, 2, 3, 1))[..., None])
        # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
        y = (Y_diag + Y_off).reshape(bsz, seq_len, n_heads, d_head)
        if D is not None:
            y = y + D_residual

        # Cutting off padded chunks
        if pad_size > 0:
            y = y[:, :seq_len, :, :]
        return y           
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, hidden_size = hidden_states.shape
        dtype = hidden_states.dtype

        A = -torch.exp(self.A_log.float())
        B = self.B_proj(hidden_states).reshape(bsz, seq_len, self.n_groups, self.d_state).repeat(1, 1, self.n_heads // self.n_groups, 1)
        C = self.C_proj(hidden_states).reshape(bsz, seq_len, self.n_groups, self.d_state).repeat(1, 1, self.n_heads // self.n_groups, 1)
        dt = nn.functional.softplus(self.dt_proj(hidden_states))
        hidden_states = self.x_proj(hidden_states).reshape(bsz, seq_len, self.n_heads, hidden_size // self.n_heads)

        hidden_states = self.ssd_algorithm(hidden_states, dt, A, B, C, self.chunk_len, D=self.D)
        hidden_states = self.out_proj(hidden_states.reshape(bsz, seq_len, hidden_size).to(dtype))
        return hidden_states
