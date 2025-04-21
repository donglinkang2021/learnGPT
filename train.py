# filepath: train.py
import torch
from src.data import get_tokenizer, get_data
from src.models.utils import generate
from src.utils import omegaconf2tb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import hydra # Add hydra import
from omegaconf import DictConfig, OmegaConf # Add omegaconf imports

# data loading (adjust to use cfg)
def get_batch(split, cfg: DictConfig, train_data, val_data):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # Use config values
    ix = torch.randint(len(data) - cfg.training.block_size, (cfg.training.batch_size,))
    x = torch.stack([data[i:i+cfg.training.block_size] for i in ix])
    y = torch.stack([data[i+1:i+cfg.training.block_size+1] for i in ix])
    x, y = x.to(cfg.training.device), y.to(cfg.training.device)
    return x, y

# estimate loss (adjust to use cfg)
@torch.no_grad()
def estimate_loss(cfg: DictConfig, model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.training.eval_iters)
        for k in range(cfg.training.eval_iters):
            # Pass cfg to get_batch
            X, Y = get_batch(split, cfg, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Use hydra decorator for the main function
@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Configuration:\n", OmegaConf.to_yaml(cfg)) # Print the resolved config
    torch.manual_seed(cfg.training.torch_seed)

    # --- Data Loading ---
    # Use cfg.data.path
    text = get_data(cfg.data.path)
    vocab_size, encode, decode = get_tokenizer(text)

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(cfg.data.train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # --- Model Initialization ---
    # Pass vocab_size and config values to the model
    model = hydra.utils.instantiate(cfg.model, vocab_size=vocab_size)
    m = model.to(cfg.training.device)
    print(f"{sum(p.numel() for p in m.parameters())/1e6:.6f} M parameters")

    # --- Optimizer ---
    # Use cfg.training.learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)

    # --- Logging ---
    # Hydra automatically changes the working directory. Use hydra.runtime.output_dir
    log_dir = "{}/{}/{}".format(
        cfg.logger.save_dir,
        cfg.logger.name,
        cfg.logger.version
    )   
    print(f"TensorBoard log directory: {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    # --- Training Loop ---
    # Use cfg.training values
    for iter in tqdm(range(cfg.training.max_iters), desc="Training", dynamic_ncols=True):
        if iter % cfg.training.eval_interval == 0 or iter == cfg.training.max_iters - 1:
            # Pass cfg, model, data to estimate_loss
            losses = estimate_loss(cfg, model, train_data, val_data)
            tqdm.write(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            writer.add_scalar('Loss/train', losses['train'], iter)
            writer.add_scalar('Loss/val', losses['val'], iter)

        # Pass cfg to get_batch
        xb, yb = get_batch('train', cfg, train_data, val_data)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    losses = estimate_loss(cfg, model, train_data, val_data)
    metrics = {
        'Final/train_loss': losses['train'],
        'Final/val_loss': losses['val'],
    }

    # --- Generation ---
    context = torch.zeros((1, 1), dtype=torch.long, device=cfg.training.device)
    # Pass model config block_size
    generated_output = decode(generate(model, context, max_new_tokens=1000, block_size=cfg.training.block_size)[0].tolist())
    tqdm.write("\n--- Generated Text ---")
    tqdm.write(generated_output)
    # Optionally save generated text to a file in the output directory
    with open(f"{log_dir}/generated_output.txt", "w") as f:
        f.write(generated_output)
    
    # log hyperparameters to TensorBoard
    writer.add_hparams(
        hparam_dict = omegaconf2tb(cfg),
        metric_dict = metrics,
    )

    writer.close()
    print(f"Training complete. Logs and outputs in: {log_dir}")

    # Save the model
    # torch.save(model.state_dict(), f"{log_dir}/model.pth")
    # Save the config
    OmegaConf.save(cfg, f'{log_dir}/config.yaml')

# Entry point for the script
if __name__ == "__main__":
    main()

# To run tensorboard: tensorboard --logdir logs --bind_all