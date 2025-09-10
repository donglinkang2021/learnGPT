python train.py --multirun \
    hydra=multi_run \
    model=gpt_v3 \
    model.pe_type=randn,sinpe,rope \
    model.n_embd=384,768 \
    model.n_head=6,8,12 \
    model.n_layer=6,8,12 \
    logger.name=gpt_v3_exp

# to test later: training.block_size=256,512,1024 \

# default config
# batch_size = 64
# block_size = 256
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
# device = 'cuda'
# eval_iters = 200
# n_embd = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2
# torch_seed = 1337