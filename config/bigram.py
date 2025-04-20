# hyperparameters
batch_size = 128 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-2
device = 'cuda'
eval_iters = 200

# torch
torch_seed = 1337 # random seed
