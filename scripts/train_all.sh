uv run train.py --multirun \
    model=bigram,bow,mlp,rnn,gpt,bigram_v2,bow_v2,gpt_v2,gpt_v3

uv run train.py --multirun \
    model=gpt_v3 \
    model.pe_type=randn,sinpe \
    model.n_embd=384,768 \
    model.n_head=6,8,12 \
    model.n_layer=6,8,12