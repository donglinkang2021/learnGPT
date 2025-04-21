import torch.nn as nn
import torch.nn.functional as F
from .utils import get_pe

__all__ = ['BigramLanguageModel_v2']

# super simple bigram model with matrix factorization
class BigramLanguageModel_v2(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, pe_type='randn'):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.fc_qkv = nn.Linear(n_embd, n_embd * 3)
        self.pe = get_pe(pe_type, n_embd, block_size)
        self.n_embd = n_embd
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.pe(tok_emb)
        x = tok_emb + pos_emb
        qkv = self.fc_qkv(x) # (B,T,3*C)
        q, k, v = qkv.split(self.n_embd, dim=-1) # (B,T,C)
        out = F.scaled_dot_product_attention(
            query=q, key=k, value=v,
            is_causal=True
        )
        logits = self.lm_head(out) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
