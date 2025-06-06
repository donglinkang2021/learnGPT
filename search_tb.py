import os
import glob
from tqdm import tqdm
from tensorboard.backend.event_processing import event_accumulator
from tabulate import tabulate
from omegaconf import OmegaConf, DictConfig
from itertools import combinations
import fire

def get_run_name(file: str) -> str:
    return os.path.join(*file.split('/')[:4])

def get_cfg(run_name: str) -> DictConfig:
    file_path = os.path.join(run_name, 'config.yaml')
    if os.path.exists(file_path):
        cfg = OmegaConf.load(os.path.join(run_name, 'config.yaml'))
        return cfg
    else:
        return None

def get_model_class_name(model_params: dict) -> str:
    return model_params['_target_'].split('.')[-1]

def simlify(metrics:list) -> list:
    new_metrics = []
    for metric in metrics:
        new_metric = {}
        # new_metric['Model'] = f"[{metric['Model']}]({metric['Run']}/config.yaml)"
        new_metric['Experiment'] = metric['Logger'].get('name', 'None')
        new_metric['pe_type'] = metric['Params'].get('pe_type', 'None')
        new_metric['n_embd'] = metric['Params'].get('n_embd', 'None')
        new_metric['n_head'] = metric['Params'].get('n_head', 'None')
        new_metric['n_layer'] = metric['Params'].get('n_layer', 'None')
        # new_metric['block_size'] = metric['Params'].get('block_size', 'None')     
        # new_metric['learning_rate'] = metric['Training'].get('learning_rate', 'None')
        new_metric['Final/train_loss'] = metric['Final/train_loss']
        new_metric['Final/val_loss'] = metric['Final/val_loss']
        new_metrics.append(new_metric)
    return new_metrics

def search(query: str = None):
    log_dir = './logs'
    # log_dir = '.cache'

    # metrics = { 'file0': { 'metric1' : 0.1, 'metric2': 0.2, ...}, ...}
    metrics = []
    
    event_files = glob.glob(os.path.join(log_dir, '**/events.out.tfevents*'), recursive=True)
    
    def search_condition(x):
        return query in x
    event_files = list(filter(search_condition, event_files))

    # print(f"Found {len(event_files)} event files matching the query: {query}")
    # return

    for file in tqdm(event_files, desc="Processing event files", dynamic_ncols=True):
        ea = event_accumulator.EventAccumulator(file)
        ea.Reload()
        run_name = get_run_name(file)
        cfg = get_cfg(run_name)
        if cfg is None:
            tqdm.write(f"Model params not found for {run_name}")
            continue
        model_params = cfg.model
        model_short = get_model_class_name(model_params)
        metric = {
            'Model': model_short, 
            "Params": model_params, 
            "Run": run_name, 
            "Logger": cfg.logger,
            "Training": cfg.training
        }
        for tag in ea.Tags().get('scalars', []):
            # each ea.Scalars(tag) just contains one scalar
            metric[tag] = ea.Scalars(tag)[0].value
        metrics.append(metric)
        
    metrics = [metric for metric in metrics if 'Final/val_loss' in metric]
    print(f"Found {len(metrics)} event files.")

    # sorted_metrics = sorted(metrics, key=lambda x: x['Final/val_loss'])
    sorted_metrics = sorted(metrics, key=lambda x: x['Final/train_loss'])
    # print(tabulate(simlify(sorted_metrics[:15]), headers='keys', tablefmt='github'))
    print(tabulate(simlify(sorted_metrics), headers='keys', tablefmt='github'))

if __name__ == '__main__':    
    fire.Fire(search)

# search all
# python search_tb.py exp 