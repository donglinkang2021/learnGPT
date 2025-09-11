uv run train.py --multirun \
    model=gpt_rope \
    model.n_embd=384,768 \
    model.n_head=6,8,12 \
    model.n_layer=6,8,12