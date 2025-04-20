import torch
import torch.nn as nn
from torch.nn import functional as F
# from config.bigram import *
from config.gpt import *
from src.data import get_tokenizer, get_data
from torch.utils.tensorboard import SummaryWriter # Add this import
import datetime # Add this import
from tqdm import tqdm # Add this import

torch.manual_seed(torch_seed)

text = get_data()
vocab_size, encode, decode = get_tokenizer(text)

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

from src.models.utils import generate
# from src.models.bigram import BigramLanguageModel
# model = BigramLanguageModel(vocab_size)
from src.models.gpt import GPTLanguageModel
model = GPTLanguageModel(vocab_size, n_embd, block_size, n_layer, n_head, dropout)

m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Get current timestamp for log directory
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
# log_dir = f'logs/bigram/{timestamp}'
log_dir = f'logs/gpt/{timestamp}'
writer = SummaryWriter(log_dir) # Initialize writer with timestamped directory

# Wrap range(max_iters) with tqdm
for iter in tqdm(range(max_iters), desc="Training"):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        # Use tqdm.write instead of print to avoid interfering with the progress bar
        tqdm.write(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # Log losses to TensorBoard
        writer.add_scalar('Loss/train', losses['train'], iter)
        writer.add_scalar('Loss/val', losses['val'], iter)

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# Use tqdm.write here as well if you want to keep output consistent
tqdm.write(decode(generate(model, context, max_new_tokens=500, block_size=block_size)[0].tolist()))

writer.close() # Close the writer
# tensorboard --logdir logs --bind_all
