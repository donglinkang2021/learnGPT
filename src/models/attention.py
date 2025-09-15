import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

def split_heads(x:torch.Tensor, head_size:int):
    return rearrange(x, 'B T (nH Hs) -> B nH T Hs', Hs=head_size)

def cat_heads(x:torch.Tensor):
    return rearrange(x, 'B nH T Hs -> B T (nH Hs)')

def manual_scaled_dot_product_attention(q, k, v, mask=None, is_causal=False):
    # q, k, v: (B, nH, T, Hs)
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if is_causal:
        # This is equivalent to is_causal=True in F.scaled_dot_product_attention
        # It will be ignored if an explicit mask is passed.
        B, nH, T, _ = q.shape
        causal_mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float('-inf'))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output

class MHA(nn.Module):
    def __init__(self, d_model, n_heads, use_flash_attention: bool = True, **kwargs):
        super(MHA, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.use_flash_attention = use_flash_attention
        self.fc_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.fc_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        qkv = self.fc_qkv(x)
        qkv = split_heads(qkv, self.head_dim)
        xq, xk, xv = qkv.chunk(chunks=3, dim=1)
        is_causal = True if mask is None else False
        if self.use_flash_attention:
            xo = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, is_causal=is_causal)
        else:
            xo = manual_scaled_dot_product_attention(xq, xk, xv, mask=mask, is_causal=is_causal)
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
    def __init__(self, d_model, n_heads, kv_heads, use_flash_attention: bool = True, **kwargs):
        super(GQA, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % kv_heads == 0, "n_heads must be divisible by kv_heads"
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.kv_heads = kv_heads
        self.n_rep = self.n_heads // self.kv_heads
        self.use_flash_attention = use_flash_attention
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, self.kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, self.kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk, xv = map(lambda t: split_heads(t, self.head_dim), (xq, xk, xv))
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        is_causal = True if mask is None else False
        if self.use_flash_attention:
            xo = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, is_causal=is_causal)
        else:
            xo = manual_scaled_dot_product_attention(xq, xk, xv, mask=mask, is_causal=is_causal)
        xo = cat_heads(xo)
        return self.wo(xo)

class MQA(GQA):
    def __init__(self, d_model, n_heads, use_flash_attention: bool = True, **kwargs):
        super(MQA, self).__init__(d_model, n_heads, 1, use_flash_attention=use_flash_attention)

class LinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, use_loop_impl: bool = False, **kwargs):
        super(LinearAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.use_loop_impl = use_loop_impl
        self.fc_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.fc_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        qkv = self.fc_qkv(x)
        qkv = split_heads(qkv, self.head_dim)
        q, k, v = qkv.chunk(chunks=3, dim=1)

        # Use elu as activation to ensure positivity
        phi = lambda t: F.elu(t) + 1
        q, k = phi(q), phi(k)

        # Causal linear attention
        is_causal = mask is None or mask.bool().any()
        if is_causal:
            if self.use_loop_impl:
                # Iterative/Loop implementation for causal linear attention
                B, nH, T, Hs = q.shape
                d_v = v.shape[-1]
                outputs = []
                # S accumulates k.T @ v for each head
                S = torch.zeros(B, nH, Hs, d_v, device=q.device)
                # Z accumulates k for each head
                Z = torch.zeros(B, nH, Hs, device=q.device)

                for i in range(T):
                    k_i = k[:, :, i, :]  # (B, nH, Hs)
                    v_i = v[:, :, i, :]  # (B, nH, d_v)
                    q_i = q[:, :, i, :]  # (B, nH, Hs)

                    # Update states
                    # k_i.unsqueeze(-1): (B, nH, Hs, 1)
                    # v_i.unsqueeze(-2): (B, nH, 1, d_v)
                    S += torch.einsum('bhd,bhv->bhdv', k_i, v_i)
                    Z += k_i

                    # Calculate output for timestep i
                    # q_i @ S
                    qS = torch.einsum('bhd,bhdv->bhv', q_i, S)
                    # q_i @ Z
                    qZ = torch.einsum('bhd,bhd->bh', q_i, Z)

                    out_i = qS / (qZ.unsqueeze(-1) + 1e-6) # (B, nH, d_v)
                    outputs.append(out_i.unsqueeze(2)) # Append as (B, nH, 1, d_v)
                
                xo = torch.cat(outputs, dim=2) # (B, nH, T, d_v)
                return self.fc_out(cat_heads(xo))
            else:
                # Vectorized implementation for causal linear attention
                k_cumsum = torch.cumsum(k, dim=2)
                kv_state = torch.einsum('bhtd,bhtf->bhtdf', k, v)
                kv_cumsum = torch.cumsum(kv_state, dim=2)
                
                q_kv = torch.einsum('bhtd,bhtdf->bhtf', q, kv_cumsum)
                q_k_sum = torch.einsum('bhtd,bhtd->bht', q, k_cumsum)
                
                # Add a small epsilon for numerical stability
                return self.fc_out(cat_heads(q_kv / (q_k_sum.unsqueeze(-1) + 1e-6)))
        else:
            # Global (non-causal) linear attention
            kv = torch.einsum('bhtd,bhtf->bhdf', k, v)
            z = 1 / (torch.einsum('bhtd,bhd->bht', q, k.sum(dim=2)) + 1e-6)
            xo = torch.einsum('bhtd,bhdf->bhtf', q, kv) * z.unsqueeze(-1)
            return self.fc_out(cat_heads(xo))
