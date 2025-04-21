import os
import glob
from tqdm import tqdm
from tensorboard.backend.event_processing import event_accumulator
from tabulate import tabulate
from omegaconf import OmegaConf
from itertools import combinations
import fire

def get_run_name(file: str) -> str:
    return os.path.join(*file.split('/')[:4])

def get_model_params(run_name: str) -> dict:
    cfg = OmegaConf.load(os.path.join(run_name, 'config.yaml'))
    return cfg.model

def model_params_to_str(model_params: dict) -> str:
    return model_params['_target_'].split('.')[-1]

def iter_pairs_kd(pairs: list, k: int):
    for combo in combinations(pairs, k):
        # below code is to skip the combo if any of the value is False
        if k > 1:
            flag = False
            for _, value in combo:
                if isinstance(value, bool):
                    if not value:
                        flag = True
                        break
            if flag:
                continue
        # the above code can delete if you want to include all the combinations
        yield ",".join(f"{key}={value}" for key, value in combo)

def drop_consecutive_same_avg_rank(model_ranks_list: list) -> list:
    result = []
    for i in range(len(model_ranks_list)):
        prev_rank = model_ranks_list[i-1]['avg_rank'] if i > 0 else None
        next_rank = model_ranks_list[i+1]['avg_rank'] if i < len(model_ranks_list)-1 else None
        curr_rank = model_ranks_list[i]['avg_rank']
        
        if prev_rank == None or next_rank == None:
            result.append(model_ranks_list[i])
        elif curr_rank != prev_rank and curr_rank != next_rank:
            result.append(model_ranks_list[i])
    return result

def get_model_ranks(sorted_metrics: list, pair_k:int=1) -> list:
    model_ranks = {} # {param1: {ranks: [1, 2, ...]}, ...}
    for rank, entry in enumerate(sorted_metrics):
        model_params = entry['Params']
        for rank_key in iter_pairs_kd(list(model_params.items()), pair_k):
            if rank_key not in model_ranks:
                model_ranks[rank_key] = {}
                model_ranks[rank_key]["ranks"] = []
            model_ranks[rank_key]["ranks"].append(rank)

    for key in model_ranks:
        model_ranks[key]["param"] = key
        model_ranks[key]["avg_rank"] = sum(model_ranks[key]["ranks"]) / len(model_ranks[key]["ranks"])
        model_ranks[key]["count"] = len(model_ranks[key]["ranks"])
        # remove ranks to save memory
        model_ranks[key].pop("ranks")

    # Remove entries with same consecutive avg_rank    
    model_ranks_list = sorted(model_ranks.values(), key=lambda x: x["avg_rank"])
    model_ranks_list = drop_consecutive_same_avg_rank(model_ranks_list)
    return model_ranks_list

def simlify(metrics:list) -> list:
    new_metrics = []
    for metric in metrics:
        new_metric = {}
        new_metric['Model'] = f"[{metric['Model']}]({metric['Run']})"
        new_metric['Final/train_loss'] = metric['Final/train_loss']
        new_metric['Final/val_loss'] = metric['Final/val_loss']
        new_metric['pe_type'] = metric['Params'].get('pe_type', 'None')
        new_metric['n_embd'] = metric['Params'].get('n_embd', 'None')
        new_metric['block_size'] = metric['Params'].get('block_size', 'None')
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
        model_params = get_model_params(run_name)
        model_short = model_params_to_str(model_params)
        metric = {'Model': model_short, "Params": model_params, "Run": run_name}
        for tag in ea.Tags().get('scalars', []):
            # each ea.Scalars(tag) just contains one scalar
            metric[tag] = ea.Scalars(tag)[0].value
        metrics.append(metric)
        
    metrics = [metric for metric in metrics if 'Final/val_loss' in metric]
    print(f"Found {len(metrics)} event files.")

    sorted_metrics = sorted(metrics, key=lambda x: x['Final/val_loss'])
    print(tabulate(simlify(sorted_metrics[:15]), headers='keys', tablefmt='github'))

    # statistics for each model params
    # print(tabulate(get_model_ranks(sorted_metrics), headers='keys', tablefmt='github'))
    # print(tabulate(get_model_ranks(sorted_metrics,2)[:10], headers='keys', tablefmt='github'))
    # print(tabulate(get_model_ranks(sorted_metrics,3)[:10], headers='keys', tablefmt='github'))
    
if __name__ == '__main__':
    
    fire.Fire(search)
