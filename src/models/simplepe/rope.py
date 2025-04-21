import torch
import torch.nn as nn

# refer to: https://github.com/karpathy/nano-llama31/blob/master/llama31.py
# for simpler implementation, remove scaling function

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    freqs_cis_real = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return freqs_cis_real

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int, theta=10000.0):
        """
        the d_model here in the refered code is head_dim here(head_dim = d_model//n_heads)
        but I have used d_model here as the head_dim
        """
        super(RotaryPositionalEncoding, self).__init__()
        assert d_model % 2 == 0, 'd_model should be even'
        self.d_model = d_model
        self.register_buffer('freqs_cis', precompute_freqs_cis(d_model, max_len, theta))

    def forward(self, x:torch.Tensor):
        # x is (**, seq_len, d_model)
        xshaped = x.reshape(*x.shape[:-1], -1, 2) # -> (**, seq_len, d_model/2, 2)
        # freqs_cis is (seq_len, d_model/2, 2)
        freqs_cis = self.freqs_cis[:x.size(-2)].to(x.device)
        x_out = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 0] * freqs_cis[..., 1] + xshaped[..., 1] * freqs_cis[..., 0]
        ], dim=-1)
        return x_out.reshape(*x.shape).type_as(x)
