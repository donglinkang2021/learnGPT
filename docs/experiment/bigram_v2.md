# 实验记录

> 20250421 20:09

做了三十八组的实验关于在原始的最简单的bigram模型上加上了不同的pe、不同block size attention的实验，结果如下：

| Description                               | val_loss |
|-------------------------------------------|----------|
| origin                                    | 2.48322  |
| matrix factorization(n_embd=384)          | 2.50986  |
| attention randnpe(n_embd=64,block_size=16) | 2.50837  |
| attention randnpe(n_embd=64,block_size=32) | 2.46541  |
| attention sinpe(n_embd=64,block_size=16)   | 2.45632  |
| attention sinpe(n_embd=64,block_size=32)   | 2.44508  |
| attention rope(n_embd=64,block_size=32)    | 2.34567  |
| attention sinpe(n_embd=64,block_size=64)   | 2.32952  |
| attention rope(n_embd=64,block_size=64)   | 2.32164  |

可以看到从上到下模型的能力在一直增强，可以得出几点结论：

- 刚加了mf的模型比原始的模型效果变差了，觉得有可能是因为n_embd在我们这个任务中是远大于vocab_size，加上mf时参数变多，也没有激活函数（或者是因果性），所以导致性能稍微下降了一下；
- 当在mf的基础上进一步加入 随机噪声的pe时和因果注意力时，模型性能有了稍微的提升，可以看到随着block size的增大，模型性能也在提升；
- 当pe从随机噪声变成sinpe时，模型性能也有了提升；这里也再次验证了block size的增大对模型性能的提升；
- 当pe从sinpe变成rope时，模型性能也有了非常大的提升；这里也再次验证了block size的增大对模型性能的提升；但随着block size的增大，用rope取代sinpe的提升幅度在减小；
- n_embd不一定越大越好，这里其实没有放出来n_embd=128和n_embd=256的结果，主要是其结果不够好，详见看附录部分的结果；

TL;DR

- rope > sinpe > randnpe
- block size越大，模型性能越好
- n_embd不一定越大越好

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

### 结果(top15)

| Model                                                              |   Final/train_loss |   Final/val_loss | pe_type   | n_embd   | block_size   |
|--------------------------------------------------------------------|--------------------|------------------|-----------|----------|--------------|
| [BigramLanguageModel_v2](./logs/bigram_v2_exp/2025-04-21-17-39-20) |            2.25039 |          2.32164 | rope      | 64       | 64           |
| [BigramLanguageModel_v2](./logs/bigram_v2_exp/2025-04-21-17-25-17) |            2.27057 |          2.32952 | sinpe     | 64       | 64           |
| [BigramLanguageModel_v2](./logs/bigram_v2_exp/2025-04-21-17-38-07) |            2.27883 |          2.34567 | rope      | 64       | 32           |
| [BigramLanguageModel_v2](./logs/bigram_v2_exp/2025-04-21-17-24-09) |            2.4103  |          2.44508 | sinpe     | 64       | 32           |
| [BigramLanguageModel_v2](./logs/bigram_v2_exp/2025-04-21-17-22-59) |            2.42673 |          2.45632 | sinpe     | 64       | 16           |
| [BigramLanguageModel_v2](./logs/bigram_v2_exp/2025-04-21-17-10-07) |            2.42835 |          2.46541 | randn     | 64       | 32           |
| [BigramLanguageModel](./logs/bigram_exp/2025-04-21-12-20-37)       |            2.45193 |          2.48322 | None      | None     | None         |
| [BigramLanguageModel_v2](./logs/bigram_v2_exp/2025-04-21-17-11-20) |            2.43593 |          2.48389 | randn     | 64       | 64           |
| [BigramLanguageModel_v2](./logs/bigram_v2_exp/2025-04-21-17-44-14) |            2.46747 |          2.48691 | rope      | 128      | 64           |
| [BigramLanguageModel_v2](./logs/bigram_v2_exp/2025-04-21-17-29-50) |            2.47422 |          2.49168 | sinpe     | 128      | 64           |
| [BigramLanguageModel_v2](./logs/bigram_v2_exp/2025-04-21-17-49-11) |            2.47612 |          2.49762 | rope      | 256      | 64           |
| [BigramLanguageModel_v2](./logs/bigram_v2_exp/2025-04-21-17-08-50) |            2.48728 |          2.50837 | randn     | 64       | 16           |
| [BigramLanguageModel_v2](./logs/bigram_v2_exp/2025-04-21-17-36-54) |            2.48251 |          2.50845 | rope      | 64       | 16           |
| [BigramLanguageModel_mf](./logs/bigram_mf_exp/2025-04-21-15-51-30) |            2.48206 |          2.50986 | None      | 384      | None         |
| [BigramLanguageModel_v2](./logs/bigram_v2_exp/2025-04-21-17-43-01) |            2.48283 |          2.51181 | rope      | 128      | 32           |
