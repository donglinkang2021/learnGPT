uv run python train.py --multirun \
    model=gpt_attn-mha,gpt_attn-gqa,gpt_attn-mqa,gpt_attn-linear \
    training.block_size=256,512,768,1024
