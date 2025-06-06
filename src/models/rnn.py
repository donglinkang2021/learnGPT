import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ['RNN']

# -----------------------------------------------------------------------------
"""
Recurrent Neural Net language model: either a vanilla RNN recurrence or a GRU.
Did not implement an LSTM because its API is a bit more annoying as it has
both a hidden state and a cell state, but it's very similar to GRU and in
practice works just as well.
"""

class RNNCell(nn.Module):
    """
    the job of a 'Cell' is to:
    take input at current time step x_{t} and the hidden state at the
    previous time step h_{t-1} and return the resulting hidden state
    h_{t} at the current timestep
    """
    def __init__(self, n_embd:int, n_embd2:int):
        super().__init__()
        self.xh_to_h = nn.Linear(n_embd + n_embd2, n_embd2)

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)
        ht = F.tanh(self.xh_to_h(xh))
        return ht

class GRUCell(nn.Module):
    """
    same job as RNN cell, but a bit more complicated recurrence formula
    that makes the GRU more expressive and easier to optimize.
    """
    def __init__(self, n_embd:int, n_embd2:int):
        super().__init__()
        # input, forget, output, gate
        self.xh_to_z = nn.Linear(n_embd + n_embd2, n_embd2)
        self.xh_to_r = nn.Linear(n_embd + n_embd2, n_embd2)
        self.xh_to_hbar = nn.Linear(n_embd + n_embd2, n_embd2)

    def forward(self, xt, hprev):
        # first use the reset gate to wipe some channels of the hidden state to zero
        xh = torch.cat([xt, hprev], dim=1)
        r = F.sigmoid(self.xh_to_r(xh))
        hprev_reset = r * hprev
        # calculate the candidate new hidden state hbar
        xhr = torch.cat([xt, hprev_reset], dim=1)
        hbar = F.tanh(self.xh_to_hbar(xhr))
        # calculate the switch gate that determines if each channel should be updated at all
        z = F.sigmoid(self.xh_to_z(xh))
        # blend the previous hidden state and the new candidate hidden state
        ht = (1 - z) * hprev + z * hbar
        return ht

class RNN(nn.Module):

    def __init__(self, cell_type:str, n_embd:int, n_embd2:int, block_size:int, vocab_size:int):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.start = nn.Parameter(torch.zeros(1, n_embd2)) # the starting hidden state
        self.wte = nn.Embedding(vocab_size, n_embd) # token embeddings table
        if cell_type == 'rnn':
            self.cell = RNNCell(n_embd, n_embd2)
        elif cell_type == 'gru':
            self.cell = GRUCell(n_embd, n_embd2)
        self.lm_head = nn.Linear(n_embd2, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()

        # embed all the integers up front and all at once for efficiency
        emb = self.wte(idx) # (b, t, n_embd)

        # sequentially iterate over the inputs and update the RNN state each tick
        hprev = self.start.expand((b, -1)) # expand out the batch dimension
        hiddens = []
        for i in range(t):
            xt = emb[:, i, :] # (b, n_embd)
            ht = self.cell(xt, hprev) # (b, n_embd2)
            hprev = ht
            hiddens.append(ht)

        # decode the outputs
        hidden = torch.stack(hiddens, 1) # (b, t, n_embd2)
        logits = self.lm_head(hidden)

        # if we are given some desired targets also calculate the loss
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
