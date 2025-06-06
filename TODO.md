# TODO

- [x] 跟着视频学完ng-video-lecture
- [x] 重写一遍代码，完善各个细节

------

20250420 update

打算全部重写一遍代码，把之前所有的代码都删掉。

打算在这里先死磕到底，把bigram模型和gpt模型自己先写出来，加上自己觉得可以加进去的技术，然后做一下消融实验，看看效果如何。

- [x] 先把bigram模型和gpt模型自己写出来
- [x] 用tensorboard可视化各个指标结果

------

20250421 update

- [x] 后续工作可以用hydra配置一下参数更好训练一点

```bash
conda activate linkdom
export CUDA_VISIBLE_DEVICES=0
bash scripts/train_bigram.sh # cost 54s
conda activate linkdom
export CUDA_VISIBLE_DEVICES=1
bash scripts/train_gpt.sh # cost 15:01
bash scripts/train_gpt_v2.sh # cost 12:48 faster
```

跑起两组实验了，第二组占4036MB，第一组模型太小甚至都没有显示占用显存。

在原来的基础上把attention mask版本去掉换成了等效的 `scaled_dot_product_attention` 训练一版，估计会快一点;

打算把之前在[simple sequence prediction](https://github.com/donglinkang2021/simple-sequence-prediction)做过的事情重新训一下，包括positional encoding, quantize部分的思考都思考一下；考虑把 [makemore](https://github.com/donglinkang2021/makemore) 部分几个模型也加进来做一下对比实验；

- [x] bigram + matrix factorization 效果基本没有变化，反而因为多了参数时间变长了变成了58s，现在的想法是看可不可以加入最简单的attention看一下有没有效果
- [x] bigram + pe 看看有没有效果，引入了一些简单的注意力机制，但没有分头也没有多层，不确定会不会有效果，先训练再看看，训练细节见 [docs/experiment/bigram_v2.md](docs/experiment/bigram_v2.md)
- [x] bow 见 [docs/experiment/bow.md](docs/experiment/bow.md)
- [x] mlp 见 [docs/experiment/mlp.md](docs/experiment/mlp.md)
- [x] rnn 见 [docs/experiment/rnn.md](docs/experiment/rnn.md)，考虑之后再补lstm
- [x] gpt 见 [docs/experiment/gpt.md](docs/experiment/gpt.md)，后续工作也在文档中
