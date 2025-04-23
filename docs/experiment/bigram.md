# bigram

> 20250421 20:09

在最原始的最简单的bigram模型上做了38组实验，[实验结果](../results/bigram.md)

- bigram-origin x1
- bigram-matrix_factorization x1
    - n_embd: 384
- bigram-mf_pe_causal_att x36
    - pe_type: randn, sinpe, rope
    - n_embd: 64, 128, 256
    - block_size: 8, 16, 32, 64

## TL;DR

- 加上pe和causal_att的效果是有效的
- rope > sinpe > randnpe
- block size越大，模型性能越好
- n_embd不一定越大越好

## 结论1：仅仅加上mf的模型比原始的模型效果变差了

刚加了mf的模型比原始的模型效果变差了，觉得有可能是因为n_embd在我们这个任务中是远大于vocab_size，加上mf时参数变多，也没有激活函数（或者是因果性），所以导致性能稍微下降了一下；

| Description                               | val_loss |
|-------------------------------------------|----------|
| origin                                    | 2.48322  |
| matrix factorization(n_embd=384)          | 2.50986  |

## 结论2：rope > sinpe > randnpe

rope > sinpe > randnpe, 在其它的所有相同配置(n_embd=xx,block_size=xx)上都符合这个规律，没有例外

| Description                               | val_loss |
|-------------------------------------------|----------|
| origin                                    | 2.48322  |
| attention randnpe(n_embd=64,block_size=16) | 2.44766  |
| attention sinpe(n_embd=64,block_size=16)   | 2.40813   |
| attention rope(n_embd=64,block_size=16)   | 2.37046  |
| attention randnpe(n_embd=64,block_size=32) | 2.46568  |
| attention sinpe(n_embd=64,block_size=32)   | 2.40418   |
| attention rope(n_embd=64,block_size=32)   | 2.36591  |

## 结论3：block_size越大，模型性能越好

64 > 32 > 16 > 8, block_size越大，模型性能越好；但是对于randn而言不符合这个规律

| pe_type   | n_embd   | block_size   |   Final/train_loss |   Final/val_loss |
|-----------|----------|--------------|--------------------|------------------|
| rope      | 128      | 64           |            2.29301 |          2.34159 |
| rope      | 128      | 32           |            2.31559 |          2.35512 |
| rope      | 128      | 16           |            2.34388 |          2.3811  |
| rope      | 128      | 8            |            2.38531 |          2.40827 |
| rope      | 256      | 64           |            2.29731 |          2.35153 |
| rope      | 256      | 32           |            2.31707 |          2.3629  |
| rope      | 256      | 16           |            2.34317 |          2.38346 |
| rope      | 256      | 8            |            2.39172 |          2.42304 |
| rope      | 64       | 64           |            2.30827 |          2.35685 |
| rope      | 64       | 32           |            2.32758 |          2.36591 |
| rope      | 64       | 16           |            2.34373 |          2.37046 |
| rope      | 64       | 8            |            2.39232 |          2.40226 |
| sinpe     | 128      | 64           |            2.3452  |          2.39416 |
| sinpe     | 128      | 32           |            2.36007 |          2.40295 |
| sinpe     | 128      | 16           |            2.37735 |          2.41932 |
| sinpe     | 128      | 8            |            2.40868 |          2.43876 |
| sinpe     | 256      | 64           |            2.34554 |          2.39477 |
| sinpe     | 256      | 32           |            2.36206 |          2.40468 |
| sinpe     | 256      | 16           |            2.37756 |          2.41735 |
| sinpe     | 256      | 8            |            2.41717 |          2.44974 |
| sinpe     | 64       | 64           |            2.36056 |          2.39928 |
| sinpe     | 64       | 32           |            2.36861 |          2.40418 |
| sinpe     | 64       | 16           |            2.37837 |          2.40813 |
| sinpe     | 64       | 8            |            2.41873 |          2.43779 |
| randn     | 64       | 16           |            2.42484 |          2.44766 |
| randn     | 64       | 8            |            2.4437  |          2.46126 |
| randn     | 64       | 32           |            2.43196 |          2.46568 |
| randn     | 64       | 64           |            2.43417 |          2.47043 |
| randn     | 256      | 16           |            2.43223 |          2.45635 |
| randn     | 256      | 8            |            2.44862 |          2.46461 |
| randn     | 256      | 32           |            2.44278 |          2.46921 |
| randn     | 256      | 64           |            2.4397  |          2.48324 |
| randn     | 128      | 32           |            2.43224 |          2.46413 |
| randn     | 128      | 16           |            2.43121 |          2.46648 |
| randn     | 128      | 8            |            2.44028 |          2.46785 |
| randn     | 128      | 64           |            2.43702 |          2.4764  |

## 结论4：n_embd不一定越大越好

| pe_type   | block_size   | n_embd   |   Final/train_loss |   Final/val_loss |
|-----------|--------------|----------|--------------------|------------------|
| rope      | 64           | 128      |            2.29301 |          2.34159 |
| rope      | 64           | 256      |            2.29731 |          2.35153 |
| rope      | 64           | 64       |            2.30827 |          2.35685 |
| rope      | 32           | 128      |            2.31559 |          2.35512 |
| rope      | 32           | 256      |            2.31707 |          2.3629  |
| rope      | 32           | 64       |            2.32758 |          2.36591 |
| rope      | 16           | 64       |            2.34373 |          2.37046 |
| rope      | 16           | 128      |            2.34388 |          2.3811  |
| rope      | 16           | 256      |            2.34317 |          2.38346 |
| rope      | 8            | 64       |            2.39232 |          2.40226 |
| rope      | 8            | 128      |            2.38531 |          2.40827 |
| rope      | 8            | 256      |            2.39172 |          2.42304 |
| sinpe     | 64           | 128      |            2.3452  |          2.39416 |
| sinpe     | 64           | 256      |            2.34554 |          2.39477 |
| sinpe     | 64           | 64       |            2.36056 |          2.39928 |
| sinpe     | 32           | 128      |            2.36007 |          2.40295 |
| sinpe     | 32           | 64       |            2.36861 |          2.40418 |
| sinpe     | 32           | 256      |            2.36206 |          2.40468 |
| sinpe     | 16           | 64       |            2.37837 |          2.40813 |
| sinpe     | 16           | 256      |            2.37756 |          2.41735 |
| sinpe     | 16           | 128      |            2.37735 |          2.41932 |
| sinpe     | 8            | 64       |            2.41873 |          2.43779 |
| sinpe     | 8            | 128      |            2.40868 |          2.43876 |
| sinpe     | 8            | 256      |            2.41717 |          2.44974 |
| randn     | 64           | 64       |            2.43417 |          2.47043 |
| randn     | 64           | 128      |            2.43702 |          2.4764  |
| randn     | 64           | 256      |            2.4397  |          2.48324 |
| randn     | 32           | 128      |            2.43224 |          2.46413 |
| randn     | 32           | 64       |            2.43196 |          2.46568 |
| randn     | 32           | 256      |            2.44278 |          2.46921 |
| randn     | 16           | 64       |            2.42484 |          2.44766 |
| randn     | 16           | 256      |            2.43223 |          2.45635 |
| randn     | 16           | 128      |            2.43121 |          2.46648 |
| randn     | 8            | 64       |            2.4437  |          2.46126 |
| randn     | 8            | 256      |            2.44862 |          2.46461 |
| randn     | 8            | 128      |            2.44028 |          2.46785 |

## Appendix

### 训练脚本

```bash
python train.py --multirun \
    hydra=multi_run \
    model=bigram_v2 \
    model.pe_type=randn,sinpe,rope \
    model.n_embd=64,128,256 \
    logger.name=bigram_v2_exp \
    training.learning_rate=1e-2 \
    training.block_size=8,16,32,64 \
```

### 模型代码

```python
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_pe

__all__ = ['BigramLanguageModel_v2']

# super simple bigram model with matrix factorization
class BigramLanguageModel_v2(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, pe_type='randn'):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = get_pe(pe_type, n_embd, block_size)
        self.n_embd = n_embd
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        x = self.wpe(self.wte(idx)) # (B,T,C)
        out = F.scaled_dot_product_attention(
            query=x, key=x, value=x,
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
```

### 原始模型

```python
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['BigramLanguageModel']

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
```
