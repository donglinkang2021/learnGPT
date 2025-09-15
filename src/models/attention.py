import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

def split_heads(x:torch.Tensor, head_size:int):
    return rearrange(x, 'B T (nH Hs) -> B nH T Hs', Hs=head_size)

def cat_heads(x:torch.Tensor):
    return rearrange(x, 'B nH T Hs -> B T (nH Hs)')

class MHA(nn.Module):
    def __init__(self, d_model, n_heads, **kwargs):
        super(MHA, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.fc_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.fc_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        qkv = self.fc_qkv(x)
        qkv = split_heads(qkv, self.head_dim)
        xq, xk, xv = qkv.chunk(chunks=3, dim=1)
        xo = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, is_causal=True if mask is None else False)
        xo = cat_heads(xo)
        return self.fc_out(xo)

def repeat_kv(x:torch.Tensor, n_rep:int):
    if n_rep == 1:
        return x
    B, n_kv, T, Hs = x.shape
    return (
        x.unsqueeze(2)
        .expand(B, n_kv, n_rep, T, Hs)
        .reshape(B, n_kv * n_rep, T, Hs)
    )

class GQA(nn.Module):
    def __init__(self, d_model, n_heads, kv_heads, **kwargs):
        super(GQA, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % kv_heads == 0, "n_heads must be divisible by kv_heads"
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.kv_heads = kv_heads
        self.n_rep = self.n_heads // self.kv_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, self.kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, self.kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk, xv = map(lambda x: split_heads(x, self.head_dim), (xq, xk, xv))
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        xo = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, is_causal=True if mask is None else False)
        xo = cat_heads(xo)
        return self.wo(xo)

class MQA(GQA):
    def __init__(self, d_model, n_heads, **kwargs):
        super(MQA, self).__init__(d_model, n_heads, 1)

class LinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, **kwargs):
        super(LinearAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.fc_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.fc_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        qkv = self.fc_qkv(x)
        qkv = split_heads(qkv, self.head_dim)
        q, k, v = qkv.chunk(chunks=3, dim=1)

        # Use elu as activation to ensure positivity
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Causal linear attention
        # incoming q, k, v are (B, nH, T, Hs)
        # we want to compute the output for each timestep
        # can be optimized with custom cuda kernel
        if mask is not None and mask.bool().any() or (mask is None):
            k_cumsum = torch.cumsum(k, dim=2)

            # kv_state: (B, nH, T, Hs, Hs)
            kv_state = torch.einsum('bhtd,bhtf->bhtdf', k, v)
            kv_cumsum = torch.cumsum(kv_state, dim=2)
            
            # q @ (sum of k*v)
            q_kv = torch.einsum('bhtd,bhtdf->bhtf', q, kv_cumsum)
            
            # q @ (sum of k)
            q_k_sum = torch.einsum('bhtd,bhtd->bht', q, k_cumsum) # bhtd,bhtd -> bht1d bhtd1 -> bht1 -> bht
            
            # Add a small epsilon for numerical stability
            return self.fc_out(cat_heads(q_kv / (q_k_sum.unsqueeze(-1) + 1e-6)))
        else:
            # Global linear attention
            kv = torch.einsum('bhtd,bhtf->bhdf', k, v)
            z = 1 / (torch.einsum('bhtd,bhd->bht', q, k.sum(dim=2)) + 1e-6)
            # (B, nH, T, Hs) @ (B, nH, Hs, Hs) * (B, nH, T, 1) -> (B, nH, T, Hs)
            return self.fc_out(cat_heads(torch.einsum('bhtd,bhdf->bhtf', q, kv) * z.unsqueeze(-1)))
