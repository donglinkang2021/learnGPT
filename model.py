import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import math

"""Head and MultiHeadAttention
(GPT) root@hubert:~/learnGPT# python gpt.py
number of parameters: 10.695233 M 
step 0: train loss 4.2816, val loss 4.2797
step 500: train loss 2.2424, val loss 2.2919
step 1000: train loss 1.7171, val loss 1.8712
step 1500: train loss 1.4769, val loss 1.6802
step 2000: train loss 1.3631, val loss 1.5875
step 2500: train loss 1.2899, val loss 1.5414
step 3000: train loss 1.2358, val loss 1.5219
step 3500: train loss 1.1898, val loss 1.4985
step 4000: train loss 1.1497, val loss 1.4909
step 4500: train loss 1.1083, val loss 1.4952
step 4999: train loss 1.0741, val loss 1.4999

LORD WILLOUGHBY:
This infering may do assist me and yet have
But in some officion shine of wail;
That oft that is Edward parting souls
But let out my wind in negrousies mine own.

PRINCE EDWARD:
What were he had thou talk of hat?

NORFOLK:
Now, with us too right.

TYBALT:
Yes, my lord, fair Richard liek, he was there?

EXETRENBY:
My nobld mother his life importal towards to them?

TRGETH:
Consent the way of course, the sweet longer
Which fast our tongue in of falsehip stuff's foretting
May be go
"""
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.ln = nn.LayerNorm(head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        q = self.ln(q)    # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd, n_head, head_size, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

"""CausalSelfAttention
(GPT) root@hubert:~/learnGPT# python gpt.py
number of parameters: 10.690625 M 
step 0: train loss 4.2152, val loss 4.2165
step 500: train loss 1.9254, val loss 2.0205
step 1000: train loss 1.4628, val loss 1.6575
step 1500: train loss 1.2991, val loss 1.5339
step 2000: train loss 1.2074, val loss 1.4847
step 2500: train loss 1.1357, val loss 1.4710
step 3000: train loss 1.0688, val loss 1.4774
step 3500: train loss 1.0055, val loss 1.4887
step 4000: train loss 0.9450, val loss 1.5177
step 4500: train loss 0.8759, val loss 1.5419
step 4999: train loss 0.8086, val loss 1.5860

KING RICHARD III:

CHARD III:
Say grieve this, I still have the depite,
When Tybalt's creaturelect is present,
That our ment weep'd by his spoil sentence on them;
And swell I seet upon the traitor's news.
The return tender that efford our comfort
That the feedful cedensity of the freezets
And fine the false world of adversaries,
The bloody lord proroud generall in the foot:
Let the flowers of many blood words drink, burns
Small insterior and the story caitors,
And then from the sweet boy wand to
"""  
class CausalSelfAttention(nn.Module):
    """mix the head and the multi-head attention together"""

    def __init__(self, n_embd, n_head, head_size, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # project the queries, keys and values
        q, k, v = self.c_attn(x).split(C, dim=2)
        k = rearrange(k, 'B T (nh hs) -> B nh T hs', nh=self.n_head)
        q = rearrange(q, 'B T (nh hs) -> B nh T hs', nh=self.n_head)
        v = rearrange(v, 'B T (nh hs) -> B nh T hs', nh=self.n_head)

        # casual self-attention: ignore "future" keys during attention
        # masked attention
        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, 
            dropout_p=self.dropout if self.training else 0,
            is_causal=True
        )
        
        # re-assemble all head outputs side by side
        y = rearrange(y, 'B nh T hs -> B T (nh hs)')
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class MLP(nn.Module):
    def __init__(self, n_embd, dropout, bias=True):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        # self.attn = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.attn = CausalSelfAttention(n_embd, n_head, head_size, block_size, dropout)
        self.mlp = MLP(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class GPT(nn.Module):

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        assert vocab_size is not None
        assert block_size is not None
        self.block_size = block_size
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd)
        ))

        self.lm_head = nn.Linear(n_embd, vocab_size)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

        # report number of parameters
        print(f"number of parameters: {self.get_num_params()/1e6:.6f} M ")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.shape
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        if targets is None:
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        else:
            logits = self.lm_head(x) # (B,T,vocab_size)
            logits = rearrange(logits, 'B T C -> (B T) C')
            targets = rearrange(targets, 'B T -> (B T)')
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx