# 测试各个attention

- 目前的主要问题是所实现的linear-attention显存消耗太高

```bash
╭──────────── Test Configuration ─────────────╮
│ Tensor Shape: x: torch.Size([4, 1024, 512]) │
│ Data Type: x: torch.float32                 │
│ Initial Memory: 33.25 MB                    │
│ Number of Trials: 5                         │
╰─────────────────────────────────────────────╯
                                       Causal Attention Function Profiling                                        
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Function Name        ┃ Execution Time (s) ┃ Peak Memory Used (MB) ┃     MSE vs baseline ┃          Mean Result ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ MHA (Flash)          │    0.1886 ± 0.3756 │          41.88 ± 3.25 │            baseline │             0.000186 │
│ MHA (Manual)         │    0.0232 ± 0.0354 │         569.00 ± 0.00 │ 0.001773 ± 0.000000 │  0.000106 ± 0.000000 │
│ GQA (kv=4, Flash)    │    0.0011 ± 0.0005 │          40.25 ± 0.00 │ 0.001818 ± 0.000000 │  0.000312 ± 0.000000 │
│ GQA (kv=4, Manual)   │    0.0059 ± 0.0002 │         553.00 ± 0.00 │ 0.001713 ± 0.000000 │ -0.000428 ± 0.000000 │
│ MQA (kv=1, Flash)    │    0.0010 ± 0.0003 │          25.25 ± 0.00 │ 0.001804 ± 0.000000 │ -0.000415 ± 0.000000 │
│ MQA (kv=1, Manual)   │    0.0060 ± 0.0002 │         554.00 ± 0.00 │ 0.001690 ± 0.000000 │  0.000135 ± 0.000000 │
│ Linear (Vectorized)  │    0.0092 ± 0.0068 │         816.00 ± 0.00 │ 0.001704 ± 0.000000 │  0.000058 ± 0.000000 │
│ Linear (Loop)        │    0.2373 ± 0.0028 │          96.76 ± 0.00 │ 0.001672 ± 0.000000 │  0.000301 ± 0.000000 │
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