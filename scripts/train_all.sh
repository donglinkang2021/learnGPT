uv run train.py --multirun \
    logger=wandb \
    model=bigram,bow,mlp,rnn,gpt,bigram_v2,bow_v2,gpt_v2,gpt_v3,gpt_rope \
    training.block_size=256,512,1024

uv run train.py --multirun \
    logger=wandb \
    model=gpt_attn-mha,gpt_attn-gqa,gpt_attn-mqa,gpt_attn-cla \
    training.block_size=256,512,1024
