from omegaconf import OmegaConf, DictConfig

__all__ = ["omegaconf2tb"]

def flatten(d, parent_key='', sep='/'):
    """Flatten the dictionary recursively"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, parent_key=new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def omegaconf2tb(cfg: DictConfig):
    # Convert OmegaConf DictConfig to a flat dictionary
    # This is useful for logging hyperparameters to TensorBoard
    omegaconf_dict = OmegaConf.to_container(cfg, resolve=True) # resolve=True means replace ${} with true values
    return flatten(omegaconf_dict)
