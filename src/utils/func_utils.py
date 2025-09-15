import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from contextlib import contextmanager
import torch
from functools import partial
import numpy as np

def named_partial(func, name, *args, **kwargs):
    p = partial(func, *args, **kwargs)
    p.__name__ = name
    return p

@contextmanager
def profile_scope(device):
    """A context manager to profile a code block's execution time and memory usage."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    start_mem = torch.cuda.memory_allocated(device)
    start_time = time.time()
    
    # A dictionary to hold the results
    results = {}
    yield results
    
    torch.cuda.synchronize() # Wait for the operation to complete
    end_time = time.time()
    
    peak_mem = torch.cuda.max_memory_allocated(device)
    
    results['exec_time'] = end_time - start_time
    results['mem_used'] = (peak_mem - start_mem) / 1024**2 # in MB

def test_performance(inputs:dict, func_prefix:str, functions_to_test:list=None, device:torch.device=torch.device("cuda"), num_trials:int=1):
    """Test the performance of global functions with a given prefix in their name."""
    console = Console()

    if not torch.cuda.is_available():
        console.print("[bold red]CUDA not available. Skipping memory test.[/bold red]")
        return

    try:
        info_panel = Panel(
            f"[bold]Tensor Shape[/bold]: {', '.join([f'{k}: {v.shape}' for k, v in inputs.items()])}\n"
            f"[bold]Data Type[/bold]: {', '.join([f'{k}: {v.dtype}' for k, v in inputs.items()])}\n"
            f"[bold]Initial Memory[/bold]: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB\n"
            f"[bold]Number of Trials[/bold]: {num_trials}",
            title="[yellow]Test Configuration[/yellow]",
            border_style="blue", expand=False
        )
        console.print(info_panel)

        table = Table(title=f"[bold green]{func_prefix} Function Profiling[/bold green]", show_header=True, header_style="bold magenta")
        table.add_column("Function Name", style="cyan", width=20)
        table.add_column("Execution Time (s)", justify="right", style="green")
        table.add_column("Peak Memory Used (MB)", justify="right", style="yellow")
        table.add_column(f"MSE vs baseline", justify="right", style="red")
        table.add_column("Mean Result", justify="right", style="blue")

        base_results = None
        
        with console.status("[bold]Running profiling...[/bold]") as status:
            for i, func in enumerate(functions_to_test):
                status.update(f"Profiling {func.__name__}...")
                
                trial_times, trial_mems, trial_results = [], [], []
                for _ in range(num_trials):
                    with profile_scope(device) as stats:
                        results = func(**inputs)
                    trial_times.append(stats['exec_time'])
                    trial_mems.append(stats['mem_used'])
                    trial_results.append(results)

                if num_trials > 1:
                    time_mean, time_std = np.mean(trial_times), np.std(trial_times)
                    mem_mean, mem_std = np.mean(trial_mems), np.std(trial_mems)
                    exec_time_str = f"{time_mean:.4f} ± {time_std:.4f}"
                    mem_used_str = f"{mem_mean:.2f} ± {mem_std:.2f}"
                else:
                    exec_time_str = f"{trial_times[0]:.4f}"
                    mem_used_str = f"{trial_mems[0]:.2f}"

                avg_result = torch.stack(trial_results).mean(dim=0)

                if i == 0:
                    base_results = avg_result
                    mse_str = "[dim]baseline[/dim]"
                    mean_val_str = f"{avg_result.mean().item():.6f}"
                else:
                    mse_vals = [(r - base_results).pow(2).mean().item() for r in trial_results]
                    mean_vals = [r.mean().item() for r in trial_results]
                    if num_trials > 1:
                        mse_mean, mse_std = np.mean(mse_vals), np.std(mse_vals)
                        mean_val_mean, mean_val_std = np.mean(mean_vals), np.std(mean_vals)
                        mse_str = f"{mse_mean:.6f} ± {mse_std:.6f}"
                        mean_val_str = f"{mean_val_mean:.6f} ± {mean_val_std:.6f}"
                    else:
                        mse_str = f"{mse_vals[0]:.6f}"
                        mean_val_str = f"{mean_vals[0]:.6f}"

                table.add_row(
                    func.__name__.replace(func_prefix, "func"),
                    exec_time_str,
                    mem_used_str,
                    mse_str,
                    mean_val_str if avg_result.numel() > 0 else "N/A"
                )
        
        console.print(table)

    except torch.cuda.OutOfMemoryError:
        console.print("[red]CUDA out of memory. Please reduce tensor sizes for testing.[/red]")
    except Exception as e:
        console.print(f"[red]An error occurred: {e}[/red]")