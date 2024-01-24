import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPT

# train hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# sample hyperparameters
max_new_tokens = 500 # max number of tokens to generate
temperature = 0.8 # > 1.0 = more exploratory, < 1.0 = more conservative
top_k = 200 # consider only the top_k most likely tokens, clamp others to have 0 probability

# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = GPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout
)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(
    context, 
    max_new_tokens, 
    temperature, 
    top_k
)[0].tolist()))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))


# output:
"""
(GPT) root@hubert:~/learnGPT# /opt/data/private/linkdom/miniconda3/envs/GPT/bin/python /root/learnGPT/gpt.py
10.793537 M parameters
step 0: train loss 4.2221, val loss 4.2277
step 500: train loss 2.0665, val loss 2.1390
step 1000: train loss 1.5668, val loss 1.7482
step 1500: train loss 1.3751, val loss 1.5865
step 2000: train loss 1.2840, val loss 1.5371
step 2500: train loss 1.2215, val loss 1.5089
step 3000: train loss 1.1782, val loss 1.4746
step 3500: train loss 1.1385, val loss 1.4812
step 4000: train loss 1.0971, val loss 1.4645
step 4500: train loss 1.0621, val loss 1.4785
step 4999: train loss 1.0282, val loss 1.4751


Shepherd:
You have stoop upon to death; and that fear with the
dukes?
What then?

ELBOW:
Pray, became him, and I had as his faction out would
believed with my budgessies as I call them
that in your gainst my thanks which we
should pose you even you. Could as Minocertain be; know
when it thou must needs advanture, as it as
you would be guiled, at whole spear.

ARCHIDAMUS:
As you remember was while here lies for me.

CAMILLO:
Why, I am in your hearts who is't; and I, as that are
to pass your love
"""