# 测试各个attention

- 目前的主要问题是所实现的linear-attention显存消耗太高
- GPU 4090x1

```bash
╭──────────── Test Configuration ─────────────╮
│ Device: cuda                                │
│ Tensor Shape: x: torch.Size([4, 1024, 512]) │
│ Data Type: x: torch.float32                 │
│ Initial Memory: 49.27 MB                    │
│ Number of Trials: 5                         │
╰─────────────────────────────────────────────╯
Running warm-up iteration...
Warm-up complete. Starting profiling...
                                       Causal Attention Function Profiling                                        
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Function Name        ┃ Execution Time (s) ┃ Peak Memory Used (MB) ┃     MSE vs baseline ┃          Mean Result ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ MHA (Manual)         │    0.0062 ± 0.0016 │         569.00 ± 0.00 │            baseline │             0.000208 │
│ MHA (Flash)          │    0.0231 ± 0.0447 │          40.25 ± 0.00 │ 0.001722 ± 0.000000 │ -0.000176 ± 0.000000 │
│ GQA (kv=4, Manual)   │    0.0053 ± 0.0010 │         553.00 ± 0.00 │ 0.001751 ± 0.000000 │  0.000067 ± 0.000000 │
│ GQA (kv=4, Flash)    │    0.0007 ± 0.0001 │          40.25 ± 0.00 │ 0.001701 ± 0.000000 │  0.000353 ± 0.000000 │
│ MQA (kv=1, Manual)   │    0.0118 ± 0.0103 │         554.00 ± 0.00 │ 0.001832 ± 0.000000 │  0.000455 ± 0.000000 │
│ MQA (kv=1, Flash)    │    0.0009 ± 0.0004 │          25.25 ± 0.00 │ 0.001706 ± 0.000000 │ -0.000329 ± 0.000000 │
│ Linear (Vectorized)  │    0.3373 ± 0.6618 │         816.00 ± 0.00 │ 0.001644 ± 0.000000 │  0.000349 ± 0.000000 │
│ Linear (Loop)        │    0.2432 ± 0.0013 │          96.76 ± 0.00 │ 0.001665 ± 0.000000 │  0.000100 ± 0.000000 │
│ Linear (Lucidrains)  │    0.4124 ± 0.8228 │         105.38 ± 0.00 │ 0.001220 ± 0.000000 │ -0.000490 ± 0.000000 │
│ SoftmaxLinear        │    0.0646 ± 0.1157 │         824.02 ± 0.00 │ 0.000909 ± 0.000000 │  0.000022 ± 0.000000 │
│ (Vectorized)         │                    │                       │                     │                      │
│ SoftmaxLinear (Loop) │    0.2527 ± 0.0040 │         368.27 ± 0.00 │ 0.000910 ± 0.000000 │ -0.000081 ± 0.000000 │
│ CodeLinear           │    0.0037 ± 0.0020 │         202.13 ± 0.00 │ 0.000909 ± 0.000000 │  0.000079 ± 0.000000 │
└──────────────────────┴────────────────────┴───────────────────────┴─────────────────────┴──────────────────────┘
```

## test_attention.py

```python
import torch
from src.models.attention import MHA, GQA, MQA, LinearAttention
from src.utils.func_utils import test_performance, named_partial

def run_attention_benchmark():
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model parameters
    d_model = 512
    n_heads = 16
    kv_heads = 4 # For GQA
    
    # Input tensor parameters
    batch_size = 4
    seq_len = 1024
    
    # Test parameters
    num_trials = 5
    
    # --- Model Initialization ---
    mha = MHA(d_model=d_model, n_heads=n_heads).to(device)
    gqa = GQA(d_model=d_model, n_heads=n_heads, kv_heads=kv_heads).to(device)
    mqa = MQA(d_model=d_model, n_heads=n_heads).to(device)
    linear_attn = LinearAttention(d_model=d_model, n_heads=n_heads).to(device)
    
    # --- Prepare Inputs ---
    inputs = {
        'x': torch.randn(batch_size, seq_len, d_model, device=device)
    }
    
    # --- Functions to Test ---
    # The first function in the list is treated as the baseline for comparison.
    functions_to_test = [
        named_partial(mha.forward, "MHA (baseline)"),
        named_partial(gqa.forward, f"GQA (kv={kv_heads})"),
        named_partial(mqa.forward, "MQA (kv=1)"),
        named_partial(linear_attn.forward, "LinearAttention (Causal)"),
    ]
    
    # --- Run Performance Test ---
    test_performance(
        inputs=inputs,
        func_prefix="Attention",
        functions_to_test=functions_to_test,
        device=device,
        num_trials=num_trials
    )

if __name__ == "__main__":
    run_attention_benchmark()
```