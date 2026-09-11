import torch
import torch.nn as nn
import torch.nn.functional as F

from graphormer.modules.model_data import ModelData


class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        if num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by a positive num_heads")
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_size = hidden_dim // num_heads
        self.scale = self.head_size**-0.5
        self.linear_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.att_dropout = nn.Dropout(dropout_rate)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, data: ModelData) -> ModelData:
        x = data.normalized_input
        batch, nodes, _ = x.shape
        q, k, v = [projection(x).view(batch, nodes, self.num_heads, self.head_size).transpose(1, 2)
                   for projection in (self.linear_q, self.linear_k, self.linear_v)]
        padding = data.x_pad_mask if data.x_pad_mask is not None else (x == 0).all(dim=-1)
        bias = data.attention_prior.masked_fill(padding[:, None, None, :], float("-inf"))
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, scale=self.scale,
            dropout_p=self.att_dropout.p if self.att_dropout.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch, nodes, self.hidden_dim)
        data.attention_output = self.linear_out(output).masked_fill(padding[..., None], 0)
        return data
