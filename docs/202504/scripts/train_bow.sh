python train.py --multirun \
    hydra=multi_run \
    model=bow \
    model.pe_type=randn,sinpe,rope \
    model.n_embd=32,64,128,256 \
    logger.name=bow_exp \
    training.learning_rate=1e-2,1e-3,3e-4,5e-4 \
    training.block_size=8,16,32,64 \

# default config
# batch_size = 64
# block_size = 8
# max_iters = 5000
# eval_interval = 500
# learning_rate = 1e-2
# device = 'cuda'
# eval_iters = 200
# torch_seed = 1337
