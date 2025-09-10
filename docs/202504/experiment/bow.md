# bow

> 20250423 13:10

在本实验中，我们将使用 Bag of Words (BoW) 模型来进行语言建模任务。我们将实现两种不同的 BoW 模型：`BoW` 和 `BoW_v2`。这两种模型的主要区别在于它们使用的`CausalBoW`模块的实现方式。

- `BoW` 通过对前面的元素依次进行加权平均；
- `BoW_v2` 则使用了 PyTorch 内置的 `scaled_dot_product_attention` 函数，将qkv的输入都设置为`x`，并将`is_causal`参数设置为`True`，以实现自回归。

做了384组实验 [实验结果](../results/bow.md)

- model x 2: bow-origin, bow-causal_self_attention
- pe_type x 3: randn,sinpe,rope
- n_embd x 4: 32,64,128,256
- block_size x 4: 8,16,32,64
- learning_rate x 4: 1e-2,1e-3,3e-4,5e-4

## TL;DR

对于bag of words这种模型，发现仅仅改动一个地方就会有非常不一样的结果

- model: 将原本的加权平均的方法去掉之后变成自注意力加权平均的方法反而更好
- pe_type：哪种模型都喜欢 rope > sinpe > randnpe
- n_embd：两个模型的行为是相反的bow_exp中是n_embd越大越好，而bow_v2_exp中则是 rope + n_embd32 最好，sinpe + n_embd128
- block_size: 两个模型完全相反，bow希望长度越小越好，bow_v2希望越长越好，说明bow_v2可以处理更多的信息
- learning_rate：玄学参数，前者是0.001，后者可能还嫌0.01不够大

## 结论1：model: bow_v2 > bow

在sinpe和rope两种编码下，bow-causal_self_attention的效果都是远远好于bow-origin；而在randn的编码下不符合这个规律；

| *Experiment*   | pe_type   |   block_size |   n_embd |   n_embd2 |   learning_rate |   Final/train_loss |   Final/val_loss |
|--------------|-----------|--------------|----------|-----------|-----------------|--------------------|------------------|
| bow_v2_exp   | rope1      |           64 |       32 |        64 |          0.01   |            2.07721 |          2.16586 |
| bow_exp      | rope1      |           64 |       32 |        64 |          0.01   |            2.33187 |          2.38089 |
| bow_v2_exp   | rope1      |           32 |       32 |        64 |          0.01   |            2.09291 |          2.17166 |
| bow_exp      | rope1      |           32 |       32 |        64 |          0.01   |            2.2906  |          2.33856 |
| bow_v2_exp   | sinpe1     |           32 |       64 |        64 |          0.01   |            2.20275 |          2.26219 |
| bow_exp      | sinpe1     |           32 |       64 |        64 |          0.01   |            2.3249  |          2.38756 |
| bow_v2_exp   | sinpe1     |           64 |      256 |        64 |          0.01   |            2.22656 |          2.26318 |
| bow_exp      | sinpe1     |           64 |      256 |        64 |          0.01   |            2.42404 |          2.45822 |
| bow_exp      | randn1     |            8 |      256 |        64 |          0.0005 |            2.24295 |          2.30507 |
| bow_v2_exp   | randn1     |            8 |      256 |        64 |          0.0005 |            2.47028 |          2.50509 |
| bow_exp      | randn1     |            8 |      256 |        64 |          0.001  |            2.23307 |          2.30615 |
| bow_v2_exp   | randn1     |            8 |      256 |        64 |          0.001  |            2.48011 |          2.51566 |

## 结论2：pe_type: rope > sinpe > randnpe

| Experiment   | *pe_type*   |   block_size |   n_embd |   learning_rate |   Final/train_loss |   Final/val_loss |
|--------------|-----------|--------------|----------|-----------------|--------------------|------------------|
| bow_exp      | rope      |            8 |      256 |          0.001  |            2.16335 |          2.2234  |
| bow_exp      | sinpe     |            8 |      256 |          0.001  |            2.24376 |          2.2851  |
| bow_exp      | randn     |            8 |      256 |          0.001  |            2.23307 |          2.30615 |
| bow_exp      | rope      |            8 |      256 |          0.0005 |            2.18912 |          2.23515 |
| bow_exp      | sinpe     |            8 |      256 |          0.0005 |            2.25098 |          2.29192 |
| bow_exp      | randn     |            8 |      256 |          0.0005 |            2.24295 |          2.30507 |
| bow_exp      | rope      |            8 |      256 |          0.0003 |            2.22182 |          2.26341 |
| bow_exp      | sinpe     |            8 |      256 |          0.0003 |            2.27242 |          2.30934 |
| bow_exp      | randn     |            8 |      256 |          0.0003 |            2.26253 |          2.31293 |

| Experiment   | *pe_type*   |   block_size |   n_embd |   learning_rate |   Final/train_loss |   Final/val_loss |
|--------------|-----------|--------------|----------|-----------------|--------------------|------------------|
| bow_v2_exp   | rope      |           64 |       32 |          0.01   |            2.07721 |          2.16586 |
| bow_v2_exp   | sinpe     |           64 |       32 |          0.01   |            2.24733 |          2.30624 |
| bow_v2_exp   | randn     |           64 |       32 |          0.01   |            2.36346 |          2.42458 |
| bow_v2_exp   | rope      |           32 |       32 |          0.01   |            2.09291 |          2.17166 |
| bow_v2_exp   | sinpe     |           32 |       32 |          0.01   |            2.23908 |          2.30265 |
| bow_v2_exp   | randn     |           32 |       32 |          0.01   |            2.35117 |          2.40852 |
| bow_v2_exp   | rope      |           64 |      256 |          0.01   |            2.0891  |          2.17185 |
| bow_v2_exp   | sinpe     |           64 |      256 |          0.01   |            2.22656 |          2.26318 |
| bow_v2_exp   | randn     |           64 |      256 |          0.01   |            2.40093 |          2.46626 |

## 结论3：n_embd

在原始的bow_exp中是n_embd越大越好，这个比较容易理解，因为越高的维度就代表可以压缩更多的信息

| Experiment   | pe_type   |   block_size |   learning_rate |   *n_embd* |   Final/train_loss |   Final/val_loss |
|--------------|-----------|--------------|-----------------|----------|--------------------|------------------|
| bow_exp      | rope      |            8 |          0.001  |      256 |            2.16335 |          2.2234  |
| bow_exp      | rope      |            8 |          0.001  |      128 |            2.18805 |          2.25075 |
| bow_exp      | rope      |            8 |          0.001  |       64 |            2.23621 |          2.28109 |
| bow_exp      | rope      |            8 |          0.001  |       32 |            2.31093 |          2.32896 |
| bow_exp      | sinpe     |            8 |          0.001  |      256 |            2.24376 |          2.2851  |
| bow_exp      | sinpe     |            8 |          0.001  |      128 |            2.24568 |          2.30322 |
| bow_exp      | sinpe     |            8 |          0.001  |       64 |            2.26897 |          2.31385 |
| bow_exp      | sinpe     |            8 |          0.001  |       32 |            2.32714 |          2.3459  |
| bow_exp      | randn     |            8 |          0.001  |      256 |            2.23307 |          2.30615 |
| bow_exp      | randn     |            8 |          0.001  |      128 |            2.24904 |          2.3062  |
| bow_exp      | randn     |            8 |          0.001  |       64 |            2.26514 |          2.31928 |
| bow_exp      | randn     |            8 |          0.001  |       32 |            2.32115 |          2.34374 |

但是在改进后的bow_v2_exp这个规律就不太符合了，最好的rope中反而是32的表示最好，而次之的sinpe中反而是128最好；但是rope + 小的head_size已经有很大的优势在这里了；

| Experiment   | pe_type   |   block_size |   learning_rate |   *n_embd* |   Final/train_loss |   Final/val_loss |
|--------------|-----------|--------------|-----------------|----------|--------------------|------------------|
| bow_v2_exp   | rope      |           64 |          0.01   |       32 |            2.07721 |          2.16586 |
| bow_v2_exp   | rope      |           64 |          0.01   |      256 |            2.0891  |          2.17185 |
| bow_v2_exp   | rope      |           64 |          0.01   |       64 |            2.09876 |          2.20107 |
| bow_v2_exp   | rope      |           64 |          0.01   |      128 |            2.11565 |          2.22777 |
|--------------|-----------|--------------|-----------------|----------|--------------------|------------------|
| bow_v2_exp   | rope      |           32 |          0.01   |       32 |            2.09291 |          2.17166 |
| bow_v2_exp   | rope      |           32 |          0.01   |      256 |            2.13776 |          2.21305 |
| bow_v2_exp   | rope      |           32 |          0.01   |       64 |            2.14147 |          2.21356 |
| bow_v2_exp   | rope      |           32 |          0.01   |      128 |            2.1587  |          2.24227 |
|--------------|-----------|--------------|-----------------|----------|--------------------|------------------|
| bow_v2_exp   | rope      |           16 |          0.01   |       32 |            2.14856 |          2.20508 |
| bow_v2_exp   | rope      |           16 |          0.01   |       64 |            2.18304 |          2.26182 |
| bow_v2_exp   | rope      |           16 |          0.01   |      128 |            2.20118 |          2.2733  |
| bow_v2_exp   | rope      |           16 |          0.01   |      256 |            2.24347 |          2.30043 |
|--------------|-----------|--------------|-----------------|----------|--------------------|------------------|
| bow_v2_exp   | sinpe     |           64 |          0.01   |      128 |            2.18681 |          2.25851 |
| bow_v2_exp   | sinpe     |           64 |          0.01   |      256 |            2.22656 |          2.26318 |
| bow_v2_exp   | sinpe     |           64 |          0.01   |       64 |            2.19857 |          2.27792 |
| bow_v2_exp   | sinpe     |           64 |          0.01   |       32 |            2.24733 |          2.30624 |
|--------------|-----------|--------------|-----------------|----------|--------------------|------------------|
| bow_v2_exp   | sinpe     |           32 |          0.01   |       64 |            2.20275 |          2.26219 |
| bow_v2_exp   | sinpe     |           32 |          0.01   |      128 |            2.22077 |          2.28687 |
| bow_v2_exp   | sinpe     |           32 |          0.01   |       32 |            2.23908 |          2.30265 |
| bow_v2_exp   | sinpe     |           32 |          0.01   |      256 |            2.28697 |          2.33533 |
|--------------|-----------|--------------|-----------------|----------|--------------------|------------------|
| bow_v2_exp   | rope      |            8 |          0.01   |       32 |            2.23728 |          2.2725  |
| bow_v2_exp   | rope      |            8 |          0.01   |       64 |            2.25263 |          2.29831 |
| bow_v2_exp   | rope      |            8 |          0.01   |      128 |            2.33055 |          2.38623 |
| bow_v2_exp   | rope      |            8 |          0.01   |      256 |            2.36351 |          2.39508 |

## 结论4：block_size

在bow_exp中 block_size是越小越好

| Experiment   | pe_type   |   n_embd |   learning_rate |   *block_size* |   Final/train_loss |   Final/val_loss |
|--------------|-----------|----------|-----------------|--------------|--------------------|------------------|
| bow_exp      | rope      |      256 |          0.001  |            8 |            2.16335 |          2.2234  |
| bow_exp      | rope      |      256 |          0.001  |           16 |            2.18881 |          2.25959 |
| bow_exp      | rope      |      256 |          0.001  |           32 |            2.24239 |          2.31014 |
| bow_exp      | rope      |      256 |          0.001  |           64 |            2.29714 |          2.34424 |
|--------------|-----------|----------|-----------------|--------------|--------------------|------------------|
| bow_exp      | rope      |      256 |          0.0005 |            8 |            2.18912 |          2.23515 |
| bow_exp      | rope      |      256 |          0.0005 |           16 |            2.21777 |          2.2683  |
| bow_exp      | rope      |      256 |          0.0005 |           32 |            2.27799 |          2.32888 |
| bow_exp      | rope      |      256 |          0.0005 |           64 |            2.33267 |          2.36949 |
|--------------|-----------|----------|-----------------|--------------|--------------------|------------------|
| bow_exp      | rope      |      128 |          0.001  |            8 |            2.18805 |          2.25075 |
| bow_exp      | rope      |      128 |          0.001  |           16 |            2.20992 |          2.27211 |
| bow_exp      | rope      |      128 |          0.001  |           32 |            2.27203 |          2.31795 |
| bow_exp      | rope      |      128 |          0.001  |           64 |            2.32788 |          2.36527 |

在bow_exp_v2中 block_size是越大越好，这可以说明这个自注意力回归部分是获得越长的 block_size 就可以压缩越多的信息（相当于记住了更加复杂的模式）

| Experiment   | pe_type   |   n_embd |   learning_rate |   block_size |   Final/train_loss |   Final/val_loss |
|--------------|-----------|----------|-----------------|--------------|--------------------|------------------|
| bow_v2_exp   | rope      |       32 |          0.01   |           64 |            2.07721 |          2.16586 |
| bow_v2_exp   | rope      |       32 |          0.01   |           32 |            2.09291 |          2.17166 |
| bow_v2_exp   | rope      |       32 |          0.01   |           16 |            2.14856 |          2.20508 |
| bow_v2_exp   | rope      |       32 |          0.01   |            8 |            2.23728 |          2.2725  |
|--------------|-----------|----------|-----------------|--------------|--------------------|------------------|
| bow_v2_exp   | rope      |      256 |          0.01   |           64 |            2.0891  |          2.17185 |
| bow_v2_exp   | rope      |      256 |          0.01   |           32 |            2.13776 |          2.21305 |
| bow_v2_exp   | rope      |      256 |          0.01   |           16 |            2.24347 |          2.30043 |
| bow_v2_exp   | rope      |      256 |          0.01   |            8 |            2.36351 |          2.39508 |
|--------------|-----------|----------|-----------------|--------------|--------------------|------------------|
| bow_v2_exp   | rope      |       64 |          0.01   |           64 |            2.09876 |          2.20107 |
| bow_v2_exp   | rope      |       64 |          0.01   |           32 |            2.14147 |          2.21356 |
| bow_v2_exp   | rope      |       64 |          0.01   |           16 |            2.18304 |          2.26182 |
| bow_v2_exp   | rope      |       64 |          0.01   |            8 |            2.25263 |          2.29831 |
|--------------|-----------|----------|-----------------|--------------|--------------------|------------------|
| bow_v2_exp   | sinpe     |      128 |          0.01   |           64 |            2.18681 |          2.25851 |
| bow_v2_exp   | sinpe     |      128 |          0.01   |           32 |            2.22077 |          2.28687 |
| bow_v2_exp   | sinpe     |      128 |          0.01   |           16 |            2.28669 |          2.32828 |
| bow_v2_exp   | sinpe     |      128 |          0.01   |            8 |            2.36651 |          2.39171 |

## 结论5：learning_rate

0.001 对 bow_exp来说是一个比较好的 learning_rate

| Experiment   | pe_type   |   n_embd |   block_size |   learning_rate |   Final/train_loss |   Final/val_loss |
|--------------|-----------|----------|--------------|-----------------|--------------------|------------------|
| bow_exp      | rope      |      256 |            8 |          0.001  |            2.16335 |          2.2234  |
| bow_exp      | rope      |      256 |            8 |          0.0005 |            2.18912 |          2.23515 |
| bow_exp      | rope      |      256 |            8 |          0.0003 |            2.22182 |          2.26341 |
| bow_exp      | rope      |      256 |            8 |          0.01   |            2.30666 |          2.36102 |
| bow_exp      | rope      |      128 |            8 |          0.001  |            2.18805 |          2.25075 |
| bow_exp      | rope      |      128 |            8 |          0.0005 |            2.23102 |          2.27737 |
| bow_exp      | rope      |      128 |            8 |          0.0003 |            2.27198 |          2.31048 |
| bow_exp      | rope      |      128 |            8 |          0.01   |            2.25251 |          2.33921 |
| bow_exp      | sinpe     |      256 |            8 |          0.001  |            2.24376 |          2.2851  |
| bow_exp      | sinpe     |      256 |            8 |          0.0005 |            2.25098 |          2.29192 |
| bow_exp      | sinpe     |      256 |            8 |          0.0003 |            2.27242 |          2.30934 |
| bow_exp      | sinpe     |      256 |            8 |          0.01   |            2.47121 |          2.4945  |

而 对于 bow_v2来说希望 learning_rate 大一点好

| Experiment   | pe_type   |   n_embd |   block_size |   learning_rate |   Final/train_loss |   Final/val_loss |
|--------------|-----------|----------|--------------|-----------------|--------------------|------------------|
| bow_v2_exp   | rope      |       32 |           64 |          0.01   |            2.07721 |          2.16586 |
| bow_v2_exp   | rope      |       32 |           32 |          0.01   |            2.09291 |          2.17166 |
| bow_v2_exp   | rope      |      256 |           64 |          0.01   |            2.0891  |          2.17185 |
| bow_v2_exp   | rope      |       64 |           64 |          0.01   |            2.09876 |          2.20107 |
| bow_v2_exp   | rope      |       32 |           16 |          0.01   |            2.14856 |          2.20508 |
| bow_v2_exp   | rope      |      256 |           32 |          0.01   |            2.13776 |          2.21305 |
| bow_v2_exp   | rope      |       64 |           32 |          0.01   |            2.14147 |          2.21356 |
| bow_v2_exp   | rope      |      128 |           64 |          0.01   |            2.11565 |          2.22777 |
| bow_v2_exp   | rope      |      128 |           32 |          0.01   |            2.1587  |          2.24227 |
| bow_v2_exp   | sinpe     |      128 |           64 |          0.01   |            2.18681 |          2.25851 |
| bow_v2_exp   | rope      |       64 |           16 |          0.01   |            2.18304 |          2.26182 |
| bow_v2_exp   | sinpe     |       64 |           32 |          0.01   |            2.20275 |          2.26219 |
| bow_v2_exp   | sinpe     |      256 |           64 |          0.01   |            2.22656 |          2.26318 |

## Appendix

### 训练脚本

```bash
python train.py --multirun \
    hydra=multi_run \
    model=bow \
    model.pe_type=randn,sinpe,rope \
    model.n_embd=32,64,128,256 \
    logger.name=bow_exp \
    training.learning_rate=1e-2,1e-3,3e-4,5e-4 \
    training.block_size=8,16,32,64 \
python train.py --multirun \
    hydra=multi_run \
    model=bow_v2 \
    model.pe_type=randn,sinpe,rope \
    model.n_embd=32,64,128,256 \
    logger.name=bow_v2_exp \
    training.learning_rate=1e-2,1e-3,3e-4,5e-4 \
    training.block_size=8,16,32,64 \
```

### 模型代码

*bow*

```python
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
```

*bow_v2*

```python
import torch.nn as nn
from torch.nn import functional as F
from .utils import get_pe

__all__ = ['BoW_v2']

# -----------------------------------------------------------------------------
# Bag of Words (BoW) language model v2: uses scaled dot product attention
# -----------------------------------------------------------------------------

class CausalBoW(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = F.scaled_dot_product_attention(
            query=x, key=x, value=x,
            is_causal=True
        )
        return y

class BoWBlock(nn.Module):
    """ collects BoW features and adds an MLP """

    def __init__(self, n_embd:int, n_embd2:int):
        super().__init__()

        # Causal BoW module
        self.cbow = CausalBoW()
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

class BoW_v2(nn.Module):
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
        self.context_block = BoWBlock(n_embd, n_embd2)
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
```

