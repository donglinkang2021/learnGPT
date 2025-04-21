import torch
import torch.nn as nn
import torch.nn.functional as F
from .simplepe import *

__all__ = ['generate', 'get_pe']

def generate(
        model:nn.Module, 
        idx:torch.Tensor,
        max_new_tokens:int, 
        block_size:int = None
    ) -> torch.Tensor:
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:] if block_size else idx
        # get the predictions
        logits, loss = model(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx


def get_pe(pe_type:str, d_model:int, block_size:int):
    if pe_type == 'randn':
        return RandomNormalPositionalEncoding(d_model, max_len=block_size)
    elif pe_type == 'sinpe':
        return SinusoidalPositionalEncoding(d_model, max_len=block_size)
    elif pe_type == 'rope':
        return RotaryPositionalEncoding(d_model, max_len=block_size)
    return None
