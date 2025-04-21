python train.py \
    model=bigram \
    logger.name=bigram_exp \
    training.learning_rate=1e-2 \
    training.block_size=8

# default config
# batch_size = 64
# block_size = 8
# max_iters = 5000
# eval_interval = 500
# learning_rate = 1e-2
# device = 'cuda'
# eval_iters = 200
# torch_seed = 1337