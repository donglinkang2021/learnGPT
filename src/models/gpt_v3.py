import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils import get_pe
import math

__all__ = ['GPTLanguageModel_v3']

# Transformer Language Model (*exactly* as used in GPT-2)

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()
        self.sqrt_2_over_pi = math.sqrt(2.0 / math.pi)

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(self.sqrt_2_over_pi * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd:int, n_head:int, dropout:float):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd) # output projection
        self.attn_dropout = dropout
        self.proj_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=True, dropout_p=self.attn_dropout)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.proj_dropout(self.c_proj(y)) # (B, T, C)
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd:int, n_head:int, dropout:float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlpf = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # c_fc
            NewGELU(), # act
            nn.Linear(4 * n_embd, n_embd), # c_proj
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPTLanguageModel_v3(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, n_layer:int, n_head:int, n_embd:int, block_size:int, vocab_size:int, dropout, pe_type:str='randn'):
        super().__init__()
        self.block_size = block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = get_pe(pe_type, n_embd, block_size),
            h = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # forward the GPT model itself
        x = self.transformer.wpe(self.transformer.wte(idx)) # (B,T,C)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
