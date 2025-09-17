# for debug
# uv run python train.py model=gpt_attn-mha

# uv run python train.py --multirun \
#     training.block_size=256 \
#     model=gpt_attn-mha,gpt_attn-gqa,gpt_attn-mqa,gpt_attn-linear

# uv run python train.py --multirun \
#     training.block_size=256,512,768,1024 \
#     model=gpt_attn-mha,gpt_attn-gqa,gpt_attn-mqa,gpt_attn-linear

uv run python train.py --multirun model=gpt_attn-cla \
    training.block_size=256,512,768,1024
    