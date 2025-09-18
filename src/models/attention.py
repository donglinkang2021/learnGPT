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

class LucidLinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, bucket_size=64, **kwargs):
        """
        just implement the bucket-level/block-wise causality (not token-level causalty) to speed up the linear attention
        refer https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py#L204
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.bucket_size = bucket_size
        self.fc_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.fc_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        qkv = self.fc_qkv(x)
        qkv = split_heads(qkv, self.head_dim)
        q, k, v = qkv.chunk(chunks=3, dim=1)
        q = q.softmax(dim=-1)
        q = q * (self.head_dim ** -0.5)

        is_causal = mask is None or mask.bool().any()

        if is_causal:
            # Causal linear attention with bucketing
            bucket_size = self.bucket_size
            assert (T % bucket_size) == 0, f'Sequence length {T} must be divisible by bucket size {bucket_size}'

            k = torch.exp(k) # Use exp as in lucidrains implementation for causal

            if mask is not None:
                # Assuming mask is (B, T), convert to (B, 1, T, 1) for broadcasting
                b_mask = mask[:, None, :, None]
                k = k.masked_fill(~b_mask, 0.)
                v = v.masked_fill(~b_mask, 0.)

            bucket_fn = lambda t: rearrange(t, 'b h (n s) d -> b h n s d', s=bucket_size)
            b_q, b_k, b_v = map(bucket_fn, (q, k, v)) # batch_size, n_heads, n_buckets, bucket_size, head_dim
            
            # bucket-wise causality implement, only consider the causality between the buckets
            # ignore the causalty of tokens in the bucket
            b_k_sum = torch.sum(b_k, dim=-2) # batch_size, n_heads, n_buckets, head_dim
            b_k_cumsum = torch.cumsum(b_k_sum, dim=-2) # batch_size, n_heads, n_buckets, head_dim

            context = torch.einsum('bhnsd,bhnse->bhnde', b_k, b_v)
            context = torch.cumsum(context, dim=-3)

            # Pad and shift for causal context
            # the 
            context = F.pad(context, (0, 0, 0, 0, 1, 0), value=0.) # pad 0 on the n dim on the left
            context = context[:, :, :-1]

            b_k_cumsum = F.pad(b_k_cumsum, (0, 0, 1, 0), value=0.)
            b_k_cumsum = b_k_cumsum[:, :, :-1]

            D_inv = 1. / (torch.einsum('bhnsd,bhnd->bhns', b_q, b_k_cumsum).clamp(min=1e-6))
            attn = torch.einsum('bhnsd,bhnde,bhns->bhnse', b_q, context, D_inv)
            
            xo = rearrange(attn, 'b h n s e -> b h (n s) e')

        else:
            # Global (non-causal) linear attention
            if mask is not None:
                # Assuming mask is (B, T), convert to (B, 1, T, 1) for broadcasting
                b_mask = mask[:, None, :, None]
                k = k.masked_fill(~b_mask, -torch.finfo(k.dtype).max)
                v = v.masked_fill(~b_mask, 0.)

            k = k.softmax(dim=-2) # Softmax over sequence length
            context = torch.einsum('bhtd,bhte->bhde', k, v)
            xo = torch.einsum('bhtd,bhde->bhte', q, context)

        return self.fc_out(cat_heads(xo))

class SoftmaxLinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, use_loop_impl: bool = False, **kwargs):
        super().__init__()
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
        q = q.softmax(dim=-1)
        q = q * (self.head_dim ** -0.5)
        # Causal linear attention with softmax
        is_causal = mask is None or mask.bool().any()
        if is_causal:
            if self.use_loop_impl:
                # Iterative/Loop implementation for causal softmax linear attention
                B, nH, T, Hs = q.shape
                d_v = v.shape[-1]
                outputs = []
                
                # KV accumulates the weighted sum of values
                KV = torch.zeros(B, nH, Hs, d_v, device=q.device, dtype=v.dtype)
                # S accumulates the softmax denominator
                S = torch.zeros(B, nH, Hs, device=q.device, dtype=k.dtype)
                # K_max tracks the running max of keys for numerical stability
                K_max = torch.full((B, nH, Hs), -torch.finfo(k.dtype).max, device=k.device, dtype=k.dtype)

                for i in range(T):
                    k_i = k[:, :, i, :]  # (B, nH, Hs)
                    v_i = v[:, :, i, :]  # (B, nH, d_v)
                    q_i = q[:, :, i, :]  # (B, nH, Hs)

                    # Rescale accumulated states with the change in max
                    K_max_new = torch.maximum(K_max, k_i) # (B, nH, Hs)
                    exp_diff = torch.exp(K_max - K_max_new) # (B, nH, Hs)
                    exp_k = torch.exp(k_i - K_max_new) # (B, nH, Hs)
                    K_max = K_max_new                    
                    S = S * exp_diff + exp_k # (B, nH, Hs)
                    KV = KV * exp_diff.unsqueeze(-1) + torch.einsum('bhd,bhv->bhdv', exp_k, v_i)

                    # Calculate output for timestep i
                    out_i = torch.einsum('bhd,bhdv->bhv', q_i, KV) / S
                    outputs.append(out_i.unsqueeze(2))
                
                xo = torch.cat(outputs, dim=2)
            else:
                k_exp = torch.exp(k - k.max(dim=2, keepdim=True).values) # (B, H, T, C)
                k_exp_cumsum = k_exp.cumsum(dim=2) # (B, H, T, C)
                k_exp_v_cumsum = torch.einsum('bhtc, bhtd -> bhtcd', k_exp, v).cumsum(dim=2) # (B, H, T, C, D)
                kv = k_exp_v_cumsum / (k_exp_cumsum.unsqueeze(-1) + 1e-9) # (B, H, T, C, D)
                xo = torch.einsum('bhtc, bhtcd -> bhtd', q, kv) # (B, H, T, D)
        else:
            k = k.softmax(dim=-2) # Softmax over sequence length
            context = torch.einsum('bhtd,bhte->bhde', k, v)
            xo = torch.einsum('bhtd,bhde->bhte', q, context)
            
        return self.fc_out(cat_heads(xo))


class CodeLinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, code_size=None, **kwargs):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        head_dim = d_model // n_heads
        code_size = head_dim // 4 if code_size is None else code_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.fc_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.fc_out = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer('fc_code', torch.randn(1, n_heads, code_size, head_dim, requires_grad=True))

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None):
        qkv = self.fc_qkv(x)
        qkv = split_heads(qkv, self.head_dim)
        q, k, v = qkv.chunk(chunks=3, dim=1) # (B, nH, T, D)
        q = q @ self.fc_code.transpose(-2, -1) * (self.head_dim ** -0.5) # (B, nH, T, C)
        k = k @ self.fc_code.transpose(-2, -1) * (self.head_dim ** -0.5) # (B, nH, T, C)
        q = q.softmax(dim=-1)
        q = q * (self.head_dim ** -0.5)
        # Causal linear attention with softmax
        is_causal = mask is None or mask.bool().any()
        if is_causal:
            k_exp = torch.exp(k - k.max(dim=2, keepdim=True).values) # (B, H, T, C)
            # k_exp = torch.exp(k) # (B, H, T, C)
            k_exp_cumsum = k_exp.cumsum(dim=2) # (B, H, T, C)
            k_exp_v_cumsum = torch.einsum('bhtc, bhtd -> bhtcd', k_exp, v).cumsum(dim=2) # (B, H, T, C, D)
            kv = k_exp_v_cumsum / (k_exp_cumsum.unsqueeze(-1) + 1e-9) # (B, H, T, C, D)
            xo = torch.einsum('bhtc, bhtcd -> bhtd', q, kv) # (B, H, T, D)
        else:
            k = k.softmax(dim=-2) # Softmax over sequence length
            context = torch.einsum('bhtd,bhte->bhde', k, v)
            xo = torch.einsum('bhtd,bhde->bhte', q, context)
            
        return self.fc_out(cat_heads(xo))