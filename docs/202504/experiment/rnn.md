# rnn

做了 512 组实验, [实验结果](../results/rnn.md)

- cell_type x 2: rnn, gru
- n_embd x 4: 32, 64, 128, 256
- n_embd2 x 4: 32, 64, 128, 256
- block_size x 4: 8, 16, 32, 64
- learning_rate x 4: 1e-2, 1e-3, 3e-4, 5e-4


## 结论1：cell_type gru > rnn

gru的效果要普遍比rnn好，基本前十名都是gru的结果

| *cell_type*   |   n_embd |   n_embd2 |   block_size |   learning_rate |   Final/train_loss |   Final/val_loss |
|-------------|----------|-----------|--------------|-----------------|--------------------|------------------|
| gru         |      256 |       256 |           64 |          0.001  |            1.28568 |          1.53878 |
| rnn         |      256 |       256 |           64 |          0.001  |            1.40858 |          1.61811 |
| gru         |      256 |       256 |           64 |          0.0005 |            1.33009 |          1.54452 |
| rnn         |      256 |       256 |           64 |          0.0005 |            1.44812 |          1.63835 |
| gru         |      128 |       256 |           64 |          0.001  |            1.28614 |          1.54482 |
| rnn         |      128 |       256 |           64 |          0.001  |            1.41242 |          1.62647 |
| gru         |       64 |       256 |           64 |          0.001  |            1.28961 |          1.55861 |
| rnn         |       64 |       256 |           64 |          0.001  |            1.41674 |          1.63329 |

## 结论2：n_embd/n_embd2 越大越好

基本选对 学习率的情况下，n_embd 越大，效果越好

| cell_type   |   n_embd2 |   block_size |   learning_rate |   *n_embd* |   Final/train_loss |   Final/val_loss |
|-------------|-----------|--------------|-----------------|----------|--------------------|------------------|
| gru         |       256 |           64 |          0.001  |      256 |            1.28568 |          1.53878 |
| gru         |       256 |           64 |          0.001  |      128 |            1.28614 |          1.54482 |
| gru         |       256 |           64 |          0.001  |       64 |            1.28961 |          1.55861 |
| gru         |       256 |           64 |          0.001  |       32 |            1.30634 |          1.56539 |
| rnn         |       256 |           64 |          0.001  |      256 |            1.40858 |          1.61811 |
| rnn         |       256 |           64 |          0.001  |      128 |            1.41242 |          1.62647 |
| rnn         |       256 |           64 |          0.001  |       64 |            1.41674 |          1.63329 |
| rnn         |       256 |           64 |          0.001  |       32 |            1.44311 |          1.64666 |
|-------------|-----------|--------------|-----------------|----------|--------------------|------------------|
| gru         |       256 |           64 |          0.0005 |      256 |            1.33009 |          1.54452 |
| gru         |       256 |           64 |          0.0005 |       64 |            1.35735 |          1.57347 |
| gru         |       256 |           64 |          0.0005 |      128 |            1.35313 |          1.57706 |
| gru         |       256 |           64 |          0.0005 |       32 |            1.385   |          1.59672 |
| rnn         |       256 |           64 |          0.0005 |      256 |            1.44812 |          1.63835 |
| rnn         |       256 |           64 |          0.0005 |      128 |            1.458   |          1.65101 |
| rnn         |       256 |           64 |          0.0005 |       64 |            1.47286 |          1.66212 |
| rnn         |       256 |           64 |          0.0005 |       32 |            1.50998 |          1.69252 |

n_embd2 越大，效果越好

| cell_type   |   block_size |   learning_rate |   n_embd |   *n_embd2* |   Final/train_loss |   Final/val_loss |
|-------------|--------------|-----------------|----------|-----------|--------------------|------------------|
| gru         |           64 |          0.001  |      256 |       256 |            1.28568 |          1.53878 |
| rnn         |           64 |          0.001  |      256 |       256 |            1.40858 |          1.61811 |
| gru         |           64 |          0.001  |      256 |       128 |            1.42988 |          1.62354 |
| rnn         |           64 |          0.001  |      256 |       128 |            1.55581 |          1.73262 |
| gru         |           64 |          0.001  |      256 |        64 |            1.60685 |          1.76937 |
| rnn         |           64 |          0.001  |      256 |        64 |            1.73563 |          1.88905 |
| gru         |           64 |          0.001  |      256 |        32 |            1.79544 |          1.9344  |
| rnn         |           64 |          0.001  |      256 |        32 |            1.9375  |          2.03312 |
|-------------|--------------|-----------------|----------|-----------|--------------------|------------------|
| gru         |           64 |          0.0005 |      256 |       256 |            1.33009 |          1.54452 |
| rnn         |           64 |          0.0005 |      256 |       256 |            1.44812 |          1.63835 |
| gru         |           64 |          0.0005 |      256 |       128 |            1.47809 |          1.65738 |
| rnn         |           64 |          0.0005 |      256 |       128 |            1.59499 |          1.76452 |
| gru         |           64 |          0.0005 |      256 |        64 |            1.65844 |          1.80956 |
| rnn         |           64 |          0.0005 |      256 |        64 |            1.77832 |          1.91686 |
| gru         |           64 |          0.0005 |      256 |        32 |            1.85267 |          1.97294 |
| rnn         |           64 |          0.0005 |      256 |        32 |            1.97241 |          2.0457  |

## 结论3：block_size 越大越好

无论在何种配置下，都是 block_size 越大越好

| cell_type   |   n_embd |   n_embd2 |   learning_rate |   block_size |   Final/train_loss |   Final/val_loss |
|-------------|----------|-----------|-----------------|--------------|--------------------|------------------|
| gru         |      256 |       256 |          0.001  |           64 |            1.28568 |          1.53878 |
| gru         |      256 |       256 |          0.001  |           32 |            1.38365 |          1.5921  |
| gru         |      256 |       256 |          0.001  |           16 |            1.51896 |          1.68664 |
| gru         |      256 |       256 |          0.001  |            8 |            1.6983  |          1.83568 |
|-------------|----------|-----------|-----------------|--------------|--------------------|------------------|
| gru         |      256 |       256 |          0.0005 |           64 |            1.33009 |          1.54452 |
| gru         |      256 |       256 |          0.0005 |           32 |            1.42569 |          1.61408 |
| gru         |      256 |       256 |          0.0005 |           16 |            1.55104 |          1.71031 |
| gru         |      256 |       256 |          0.0005 |            8 |            1.72421 |          1.85694 |
|-------------|----------|-----------|-----------------|--------------|--------------------|------------------|
| gru         |      128 |       256 |          0.001  |           64 |            1.28614 |          1.54482 |
| gru         |      128 |       256 |          0.001  |           32 |            1.384   |          1.59006 |
| gru         |      128 |       256 |          0.001  |           16 |            1.51293 |          1.68645 |
| gru         |      128 |       256 |          0.001  |            8 |            1.69755 |          1.84882 |
|-------------|----------|-----------|-----------------|--------------|--------------------|------------------|
| rnn         |      256 |       256 |          0.001  |           64 |            1.40858 |          1.61811 |
| rnn         |      256 |       256 |          0.001  |           32 |            1.48759 |          1.66813 |
| rnn         |      256 |       256 |          0.001  |           16 |            1.59533 |          1.75763 |
| rnn         |      256 |       256 |          0.001  |            8 |            1.76138 |          1.90377 |
|-------------|----------|-----------|-----------------|--------------|--------------------|------------------|
| rnn         |      128 |       256 |          0.001  |           64 |            1.41242 |          1.62647 |
| rnn         |      128 |       256 |          0.001  |           32 |            1.48814 |          1.66909 |
| rnn         |      128 |       256 |          0.001  |           16 |            1.59879 |          1.75567 |
| rnn         |      128 |       256 |          0.001  |            8 |            1.76034 |          1.89603 |

## 结论4：learning rate

玄学参数供参考

| cell_type   |   n_embd |   n_embd2 |   block_size |   learning_rate |   Final/train_loss |   Final/val_loss |
|-------------|----------|-----------|--------------|-----------------|--------------------|------------------|
| gru         |      256 |       256 |           64 |          0.001  |            1.28568 |          1.53878 |
| gru         |      256 |       256 |           64 |          0.0005 |            1.33009 |          1.54452 |
| gru         |      256 |       256 |           64 |          0.0003 |            1.37845 |          1.56763 |
| gru         |      256 |       256 |           64 |          0.01   |            1.49246 |          1.70471 |
|-------------|----------|-----------|--------------|-----------------|--------------------|------------------|
| gru         |      128 |       256 |           64 |          0.001  |            1.28614 |          1.54482 |
| gru         |      128 |       256 |           64 |          0.0005 |            1.35313 |          1.57706 |
| gru         |      128 |       256 |           64 |          0.0003 |            1.39844 |          1.59829 |
| gru         |      128 |       256 |           64 |          0.01   |            1.50984 |          1.70493 |
|-------------|----------|-----------|--------------|-----------------|--------------------|------------------|
| gru         |       64 |       256 |           64 |          0.001  |            1.28961 |          1.55861 |
| gru         |       64 |       256 |           64 |          0.0005 |            1.35735 |          1.57347 |
| gru         |       64 |       256 |           64 |          0.0003 |            1.42295 |          1.61503 |
| gru         |       64 |       256 |           64 |          0.01   |            2.42339 |          2.49157 |
|-------------|----------|-----------|--------------|-----------------|--------------------|------------------|
| rnn         |      256 |       256 |           64 |          0.001  |            1.40858 |          1.61811 |
| rnn         |      256 |       256 |           64 |          0.0005 |            1.44812 |          1.63835 |
| rnn         |      256 |       256 |           64 |          0.0003 |            1.4924  |          1.66984 |
| rnn         |      256 |       256 |           64 |          0.01   |            1.6389  |          1.82614 |
|-------------|----------|-----------|--------------|-----------------|--------------------|------------------|
| rnn         |      128 |       256 |           64 |          0.001  |            1.41242 |          1.62647 |
| rnn         |      128 |       256 |           64 |          0.0005 |            1.458   |          1.65101 |
| rnn         |      128 |       256 |           64 |          0.0003 |            1.51176 |          1.68556 |
| rnn         |      128 |       256 |           64 |          0.01   |            1.63158 |          1.82966 |

## Appendix

### 训练脚本

```bash
python train.py --multirun \
    hydra=multi_run \
    model=rnn \
    model.cell_type=rnn,gru \
    model.n_embd=32,64,128,256 \
    model.n_embd2=32,64,128,256 \
    logger.name=rnn_exp \
    training.learning_rate=1e-2,1e-3,3e-4,5e-4 \
    training.block_size=8,16,32,64 \
```

### 模型代码

```python
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
```
