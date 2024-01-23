import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTLanguageModel

# hyperparameters
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

model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout
)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

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
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))


# output:
"""
(GPT) root@hubert:~/learnGPT# /opt/data/private/linkdom/miniconda3/envs/GPT/bin/python /root/learnGPT/gpt.py
10.788929 M parameters
step 0: train loss 4.2221, val loss 4.2306
step 500: train loss 1.7534, val loss 1.9144
step 1000: train loss 1.3946, val loss 1.6078
step 1500: train loss 1.2678, val loss 1.5294
step 2000: train loss 1.1854, val loss 1.5016
step 2500: train loss 1.1209, val loss 1.5039
step 3000: train loss 1.0727, val loss 1.4858
step 3500: train loss 1.0184, val loss 1.5108
step 4000: train loss 0.9610, val loss 1.5256
step 4500: train loss 0.9132, val loss 1.5402
step 4999: train loss 0.8567, val loss 1.5685


Shuposteth you by, sir; but they disturb'd, then
feastical fingers, they shall rehear upon them.

DUKE VINCENTIO:
Go this grace hence the duke that prefermed and ress.
What comes here?

GLOUCESTER:
She shall not we between the king of the king?

KING EDWARD IV:
Is it be passed opinion of women itself:
Had he forgots in all the rest of justice;
For he, then all ourselves, will derive.

EDWARD:
We will not kiss the head, broke with sorrow.

QUEEN MARGARET:
His cares, ay, gave revenge thy woman;
A
"""