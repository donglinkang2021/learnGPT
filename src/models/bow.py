import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils import get_pe

__all__ = ['BoW']

# -----------------------------------------------------------------------------
# Bag of Words (BoW) language model

class CausalBoW(nn.Module):
    """
    Causal bag of words. Averages the preceding elements and looks suspiciously like
    a CausalAttention module you'd find in a transformer, for no apparent reason at all ;)
    """
    def __init__(self, block_size:int):
        super().__init__()

        # used to mask out vectors and preserve autoregressive property
        self.block_size = block_size
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                            .view(1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, n_embd

        # do the weighted average of all preceeding token features
        att = torch.zeros((B, T, T), device=x.device)
        att = att.masked_fill(self.bias[:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ x # (B, T, T) x (B, T, C) -> (B, T, C)

        return y

class BoWBlock(nn.Module):
    """ collects BoW features and adds an MLP """

    def __init__(self, n_embd:int, n_embd2:int, block_size:int):
        super().__init__()

        # Causal BoW module
        self.cbow = CausalBoW(block_size)
        # MLP assembler
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, n_embd2),
            c_proj  = nn.Linear(n_embd2, n_embd),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(F.tanh(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.cbow(x)
        x = x + self.mlpf(x)
        return x

class BoW(nn.Module):
    """
    takes the previous block_size tokens, encodes them with a lookup table,
    also encodes their positions with lookup table, then averages all of those
    embeddings up and uses that to predict the next token.
    """

    def __init__(self, n_embd:int, n_embd2:int, block_size:int, vocab_size:int, pe_type='randn'):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        # token embedding
        self.wte = nn.Embedding(vocab_size, n_embd)
        # position embedding
        self.wpe = get_pe(pe_type, n_embd, block_size)
        # context block
        self.context_block = BoWBlock(n_embd, n_embd2, block_size)
        # language model head decoder layer
        self.lm_head = nn.Linear(n_embd, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        # forward the token and position embedding layers
        x = self.wpe(self.wte(idx)) # (B,T,C)
        # run the bag of words context module
        x = self.context_block(x)
        # decode to next token probability
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
