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
bash scripts/train_bigram.sh
conda activate linkdom
export CUDA_VISIBLE_DEVICES=1
bash scripts/train_gpt.sh
```

跑起两组实验了，第二组占4036MB，第一组模型太小甚至都没有显示占用显存。
