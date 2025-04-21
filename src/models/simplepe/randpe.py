import torch
import torch.nn as nn

"""just a simple random normal positional encoding"""

class RandomNormalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(RandomNormalPositionalEncoding, self).__init__()
        self.register_buffer('pe', 0.02 * torch.randn(1, max_len, d_model))

    def forward(self, x:torch.Tensor):
        # x is (**, seq_len, d_model)
        return x + self.pe[:, :x.size(-2)]
