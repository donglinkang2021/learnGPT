uv run python train.py --multirun \
    logger=wandb \
    logger.project_name=learnGPT-attn \
    model=gpt_attn-mha,gpt_attn-gqa,gpt_attn-mqa,gpt_attn-linear,gpt_attn-lla,gpt_attn-sla,gpt_attn-cla \
    training.block_size=256,512,1024