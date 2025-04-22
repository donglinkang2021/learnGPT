python train.py --multirun \
    hydra=multi_run \
    model=mlp \
    model.n_embd=32,64,128,256 \
    model.n_embd2=32,64,128,256 \
    logger.name=mlp_exp \
    training.learning_rate=1e-2,1e-3,3e-4,5e-4 \
    training.block_size=8,16,32,64 \
