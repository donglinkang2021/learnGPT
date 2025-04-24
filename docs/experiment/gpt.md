# gpt

做了 56 个实验，这里的分析需要仔细一点，不能和以前一样只是分析val_loss（以前的情况基本都是train_loss1 < train_loss2, 那么必然会有 val_loss1 < val_loss2）, 但是这里的很多结果往往都是出现了train_loss1 < train_loss2 而 val_loss1 > val_loss2的情况，说明模型拟合数据的能力变强大了，用gpt2去压缩一小部分数据进来发生了过拟合，因此：这里将会同时分析 train_loss 和 val_loss的影响。

另外由于显存因素的影响，自己在做实验的时候没有敢一开始就选取比较大的 block_size，而是固定 block_size=256 来调整其它参数；

实验设计如下：

1. gpt x 1 : origin version of [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture)
2. gpt_v2 x 1 : modify the class `Head` to use `F.scaled_dot_product_attention` instead of `torch.matmul` attention mask `tril` to speed up the training process.
3. gpt_v3 x 54 : a clean modified version of the class `Transformer` [karpathy/makemore](https://github.com/karpathy/makemore)
    - `pe_type` : `randn`, `sinpe`, `rope`
    - `n_embd` : `384`, `768`
    - `n_head` : `6`, `8`, `12`
    - `n_layer` : `6`, `8`, `12`

The difference between gpt_v2 and gpt_v3:

1. relu -> gelu
2. MultiHeadAttention -> CausalSelfAttention
3. better positional encoding(sinpe/rope)
4. lm head set bias=False

## Experiment

鉴于这里的 gpt 和 gpt_v2 的差别实际上并不大，这里着重分析 gpt_v3 的实验结果。

| Experiment   | pe_type   |   n_embd |   n_head |   n_layer |   Final/train_loss |   Final/val_loss |
|--------------|-----------|----------|----------|-----------|--------------------|------------------|
| gpt_exp      | None      |      384 |        6 |         6 |           0.853996 |          1.57244 |
| gpt_v2_exp   | None      |      384 |        6 |         6 |           0.896341 |          1.58272 |

### pe_type

如果按照 val_loss 排序，基本看不出特别的规律，但普遍的 sinpe 是要 优于 rope 和 randpe 的

| Experiment   | pe_type   |   n_embd |   n_head |   n_layer |   Final/val_loss |
|--------------|-----------|----------|----------|-----------|------------------|
| gpt_v3_exp   | sinpe     |      384 |        8 |         6 |          1.50614 |
| gpt_v3_exp   | randn     |      384 |        8 |         6 |          1.5254  |
| gpt_v3_exp   | rope      |      384 |        8 |         6 |          1.5443  |
|--------------|-----------|----------|----------|-----------|------------------|
| gpt_v3_exp   | sinpe     |      384 |        6 |         6 |          1.51259 |
| gpt_v3_exp   | rope      |      384 |        6 |         6 |          1.53221 |
| gpt_v3_exp   | randn     |      384 |        6 |         6 |          1.53602 |
|--------------|-----------|----------|----------|-----------|------------------|
| gpt_v3_exp   | sinpe     |      384 |       12 |         6 |          1.51984 |
| gpt_v3_exp   | randn     |      384 |       12 |         6 |          1.52628 |
| gpt_v3_exp   | rope      |      384 |       12 |         6 |          1.5338  |
|--------------|-----------|----------|----------|-----------|------------------|
| gpt_v3_exp   | randn     |      384 |        6 |         8 |          1.5475  |
| gpt_v3_exp   | sinpe     |      384 |        6 |         8 |          1.5605  |
| gpt_v3_exp   | rope      |      384 |        6 |         8 |          1.62199 |

如果按照 train_loss 排序，rope > randnpe > sinpe，而且可以发现大模型的拟合能力更强（train_loss更低），但此时sinpe相比于randn的就差许多。

| Experiment   | pe_type   |   n_embd |   n_head |   n_layer |   Final/train_loss |
|--------------|-----------|----------|----------|-----------|--------------------|
| gpt_v3_exp   | rope      |      768 |        8 |        12 |           0.170163 |
| gpt_v3_exp   | randn     |      768 |        8 |        12 |           0.220199 |
| gpt_v3_exp   | sinpe     |      768 |        8 |        12 |           0.296302 |
|--------------|-----------|----------|----------|-----------|--------------------|
| gpt_v3_exp   | randn     |      768 |       12 |        12 |           0.186352 |
| gpt_v3_exp   | rope      |      768 |       12 |        12 |           0.217138 |
| gpt_v3_exp   | sinpe     |      768 |       12 |        12 |           0.288609 |
|--------------|-----------|----------|----------|-----------|--------------------|
| gpt_v3_exp   | rope      |      768 |        6 |        12 |           0.204015 |
| gpt_v3_exp   | randn     |      768 |        6 |        12 |           0.207572 |
| gpt_v3_exp   | sinpe     |      768 |        6 |        12 |           0.408973 |
|--------------|-----------|----------|----------|-----------|--------------------|
| gpt_v3_exp   | rope      |      768 |        6 |         8 |           0.270419 |
| gpt_v3_exp   | randn     |      768 |        6 |         8 |           0.332755 |
| gpt_v3_exp   | sinpe     |      768 |        6 |         8 |           0.638026 |
|--------------|-----------|----------|----------|-----------|--------------------|
| gpt_v3_exp   | rope      |      768 |        8 |         8 |           0.296335 |
| gpt_v3_exp   | randn     |      768 |        8 |         8 |           0.307031 |
| gpt_v3_exp   | sinpe     |      768 |        8 |         8 |           0.39059  |

### n_embd

如果按照 val_loss 排序，n_embd=384 > n_embd=768, 说明小模型的val_loss更高；但是我们发现了 n_embd 大的模型的train_loss更低，这说明了过拟合；

| pe_type   |   n_head |   n_layer |   n_embd |   Final/train_loss |   Final/val_loss |
|-----------|----------|-----------|----------|--------------------|------------------|
| sinpe     |        8 |         6 |      384 |           1.08104  |          1.50614 |
| sinpe     |        8 |         6 |      768 |           0.653782 |          1.85919 |
|-----------|----------|-----------|----------|--------------------|------------------|
| sinpe     |        6 |         6 |      384 |           1.0392   |          1.51259 |
| sinpe     |        6 |         6 |      768 |           0.546295 |          1.95698 |
|-----------|----------|-----------|----------|--------------------|------------------|
| sinpe     |       12 |         6 |      384 |           1.03259  |          1.51984 |
| sinpe     |       12 |         6 |      768 |           0.558815 |          1.96209 |
|-----------|----------|-----------|----------|--------------------|------------------|
| randn     |        8 |         6 |      384 |           1.08019  |          1.5254  |
| randn     |        8 |         6 |      768 |           0.416049 |          2.11739 |

如果更加关注 数据压缩能力，根据 train_loss 排序，我们可以发现

1. rope + 较大的 n_embd 更容易数据压缩，更加容易过拟合
2. 如果模型足够大，比如 n_layer=12, n_head=12, 那么 即使用 randnpe 的编码也能取得比较好的结果
3. 排在前面的 没有 sinpe，也可能说明sinpe并不适合这种很大维度的位置编码，至少没有 randnpe 或者 rope适合

| pe_type   |   n_head |   n_layer |   n_embd |   Final/train_loss |   Final/val_loss |
|-----------|----------|-----------|----------|--------------------|------------------|
| rope      |        8 |        12 |      768 |           0.170163 |          2.78921 |
| rope      |        8 |        12 |      384 |           0.736511 |          1.76095 |
|-----------|----------|-----------|----------|--------------------|------------------|
| randn     |       12 |        12 |      768 |           0.186352 |          2.7187  |
| randn     |       12 |        12 |      384 |           0.698273 |          1.7789  |
|-----------|----------|-----------|----------|--------------------|------------------|
| rope      |        6 |        12 |      768 |           0.204015 |          2.64065 |
| rope      |        6 |        12 |      384 |           0.745283 |          1.73058 |
|-----------|----------|-----------|----------|--------------------|------------------|
| randn     |        6 |        12 |      768 |           0.207572 |          2.63155 |
| randn     |        6 |        12 |      384 |           0.819226 |          1.65891 |
|-----------|----------|-----------|----------|--------------------|------------------|
| rope      |       12 |        12 |      768 |           0.217138 |          2.68057 |
| rope      |       12 |        12 |      384 |           0.723379 |          1.76504 |

### n_head

关注泛化能力，按照 val_loss 排序，这里基本对于 n_head 没有说明规律而言，很有可能是这个参数对于模型能力不那么敏感（需要调学习率才能知道最优的模型达到什么长度，或者可能跟数据有关），但直觉肯定是越大越好

| pe_type   |   n_layer |   n_embd |   n_head |   Final/train_loss |   Final/val_loss |
|-----------|-----------|----------|----------|--------------------|------------------|
| sinpe     |         6 |      384 |        8 |           1.08104  |          1.50614 |
| sinpe     |         6 |      384 |        6 |           1.0392   |          1.51259 |
| sinpe     |         6 |      384 |       12 |           1.03259  |          1.51984 |
| randn     |         6 |      384 |        8 |           1.08019  |          1.5254  |
| randn     |         6 |      384 |       12 |           1.06979  |          1.52628 |
| rope      |         6 |      384 |        6 |           1.0892   |          1.53221 |
| rope      |         6 |      384 |       12 |           1.04415  |          1.5338  |
| randn     |         6 |      384 |        6 |           1.03537  |          1.53602 |
| rope      |         6 |      384 |        8 |           1.00815  |          1.5443  |

关注数据压缩能力，按照 train_loss 排序，发现

1. 这里排在前面的借本都是 n_embd=768 的模型，说明 n_embd=384 的模型压缩能力更差
2. rope + 较大的 n_head + 较大的 n_embd 更容易数据压缩，更加容易过拟合；但是如果 n_head 更多我们的模型反而会不那么好，这里的直觉告诉我们应该是加位置编码的位置不对的原有，导致如果 注意力头变多的话 各个头的位置信息就不明显了
3. 此时也是有 n_layer出现在更前面；

| pe_type   |   n_layer |   n_embd |   n_head |   Final/train_loss |   Final/val_loss |
|-----------|-----------|----------|----------|--------------------|------------------|
| rope      |        12 |      768 |        8 |           0.170163 |          2.78921 |
| randn     |        12 |      768 |       12 |           0.186352 |          2.7187  |
| rope      |        12 |      768 |        6 |           0.204015 |          2.64065 |
| randn     |        12 |      768 |        6 |           0.207572 |          2.63155 |
| rope      |        12 |      768 |       12 |           0.217138 |          2.68057 |
| randn     |        12 |      768 |        8 |           0.220199 |          2.59765 |
| rope      |         8 |      768 |        6 |           0.270419 |          2.40501 |
| sinpe     |        12 |      768 |       12 |           0.288609 |          2.44934 |
| sinpe     |        12 |      768 |        8 |           0.296302 |          2.42164 |
| rope      |         8 |      768 |        8 |           0.296335 |          2.33454 |

### n_layer

基本都是小模型的 val_loss 更低，大模型更容易过拟合

| pe_type   |   n_embd |   n_head |   n_layer |   Final/train_loss |   Final/val_loss |
|-----------|----------|----------|-----------|--------------------|------------------|
| sinpe     |      384 |        8 |         6 |           1.08104  |          1.50614 |
| sinpe     |      384 |        6 |         6 |           1.0392   |          1.51259 |
| sinpe     |      384 |       12 |         6 |           1.03259  |          1.51984 |
| randn     |      384 |        8 |         6 |           1.08019  |          1.5254  |
| randn     |      384 |       12 |         6 |           1.06979  |          1.52628 |
| rope      |      384 |        6 |         6 |           1.0892   |          1.53221 |
| rope      |      384 |       12 |         6 |           1.04415  |          1.5338  |
| randn     |      384 |        6 |         6 |           1.03537  |          1.53602 |
| rope      |      384 |        8 |         6 |           1.00815  |          1.5443  |
| randn     |      384 |        6 |         8 |           0.976899 |          1.5475  |
| sinpe     |      384 |        6 |         8 |           0.959327 |          1.5605  |
| sinpe     |      384 |        6 |        12 |           0.982303 |          1.5698  |

越大越好，scale up!

| pe_type   |   n_embd |   n_head |   n_layer |   Final/train_loss |   Final/val_loss |
|-----------|----------|----------|-----------|--------------------|------------------|
| rope      |      768 |        8 |        12 |           0.170163 |          2.78921 |
| rope      |      768 |        8 |         8 |           0.296335 |          2.33454 |
| rope      |      768 |        8 |         6 |           0.431042 |          2.11614 |
|-----------|----------|----------|-----------|--------------------|------------------|
| randn     |      768 |       12 |        12 |           0.186352 |          2.7187  |
| randn     |      768 |       12 |         8 |           0.308402 |          2.36178 |
| randn     |      768 |       12 |         6 |           0.466691 |          2.05952 |
|-----------|----------|----------|-----------|--------------------|------------------|
| rope      |      768 |        6 |        12 |           0.204015 |          2.64065 |
| rope      |      768 |        6 |         8 |           0.270419 |          2.40501 |
| rope      |      768 |        6 |         6 |           0.434488 |          2.12951 |
|-----------|----------|----------|-----------|--------------------|------------------|
| randn     |      768 |        6 |        12 |           0.207572 |          2.63155 |
| randn     |      768 |        6 |         8 |           0.332755 |          2.28096 |
| randn     |      768 |        6 |         6 |           0.485289 |          2.04311 |
|-----------|----------|----------|-----------|--------------------|------------------|
| sinpe     |      768 |       12 |        12 |           0.288609 |          2.44934 |
| sinpe     |      768 |       12 |         8 |           0.38907  |          2.2224  |
| sinpe     |      768 |       12 |         6 |           0.558815 |          1.96209 |

## Later

后续可以做的事情是在不同的block_size上进行实验。

另外，我们这里训练关注更多的是模型简单训练神经网络下降的趋势和简单的调参，没有做很大规模数据的学习，也没有微调的部分，后续还得持续follow up才行。

- 数据集
    - 可以先考虑小的数据：先从这个简单的shakespeare数据开始，可以考虑把 [makemore](https://github.com/donglinkang2021/makemore) 中的数据也拿过来；玩一下生成人名的任务；
    - 进而可以参考 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)的做法用 openwebtext 数据集来训练一下，看看效果如何；
    - 或者用这些数据集： [minipile](https://huggingface.co/datasets/JeanKaddour/minipile) 和 [openwebtext2](https://openwebtext2.readthedocs.io/en/latest/)
- 模型
    - 考虑rope的位置问题，这里是放在最开始的位置了，但其实原来的实现应该是放在分head之后的qk中，这样对于注意力矩阵来说位置编码才更加明显；sinpe应该也受这个问题影响所以导致还不如randnpe；之前所有的模型之所以一致可以得出 rope > sinpe > randpe 的结论其实是因为没有分head；
    - 考虑把 mha, mqa, gqa, mla实现一下 refer: [MHA、MQA、GQA和MLA](https://zhuanlan.zhihu.com/p/21151178690)
    - 考虑把自己的 vq 代替 mlp 的部分实现一下 refer: [simple sequence prediction](https://github.com/donglinkang2021/simple-sequence-prediction)

> 思考一件事：越大越好 == scale up!

## Appendix

### gpt

模型代码见 [src/models/gpt.py](../../src/models/gpt.py)

训练脚本

```bash
python train.py \
    model=gpt \
    logger.name=gpt_exp
```

原本 [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture) 中的参数

```python
# default config
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
torch_seed = 1337
```

### gpt_v2

模型代码见 [src/models/gpt_v2.py](../../src/models/gpt_v2.py)

训练脚本

```bash
python train.py \
    model=gpt_v2 \
    logger.name=gpt_v2_exp
```

### gpt_v3

模型代码见 [src/models/gpt_v3.py](../../src/models/gpt_v3.py)

训练脚本

```bash
python train.py --multirun \
    hydra=multi_run \
    model=gpt_v3 \
    model.pe_type=randn,sinpe,rope \
    model.n_embd=384,768 \
    model.n_head=6,8,12 \
    model.n_layer=6,8,12 \
    logger.name=gpt_v3_exp

# to test later: training.block_size=256,512,1024 \
```
