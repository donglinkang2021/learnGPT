# 测试各个attention

- 目前的主要问题是所实现的linear-attention显存消耗太高

```bash
╭──────────── Test Configuration ─────────────╮
│ Tensor Shape: x: torch.Size([4, 1024, 512]) │
│ Data Type: x: torch.float32                 │
│ Initial Memory: 29.25 MB                    │
│ Number of Trials: 5                         │
╰─────────────────────────────────────────────╯
                                           Attention Function Profiling                                           
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Function Name        ┃ Execution Time (s) ┃ Peak Memory Used (MB) ┃     MSE vs baseline ┃          Mean Result ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ MHA (Flash)          │    0.1985 ± 0.3955 │          41.88 ± 3.25 │            baseline │            -0.000114 │
│ MHA (Manual)         │    0.0225 ± 0.0343 │         569.00 ± 0.00 │ 0.001789 ± 0.000000 │  0.000027 ± 0.000000 │
│ GQA (kv=4, Flash)    │    0.0011 ± 0.0005 │          40.25 ± 0.00 │ 0.001795 ± 0.000000 │  0.000326 ± 0.000000 │
│ GQA (kv=4, Manual)   │    0.0069 ± 0.0004 │         553.00 ± 0.00 │ 0.001815 ± 0.000000 │  0.000474 ± 0.000000 │
│ MQA (kv=1, Flash)    │    0.0012 ± 0.0003 │          25.25 ± 0.00 │ 0.001756 ± 0.000000 │  0.000678 ± 0.000000 │
│ MQA (kv=1, Manual)   │    0.0063 ± 0.0004 │         554.00 ± 0.00 │ 0.001834 ± 0.000000 │  0.000232 ± 0.000000 │
│ Linearfunc (Causal)  │    0.0099 ± 0.0066 │         816.00 ± 0.00 │ 0.001686 ± 0.000000 │ -0.000041 ± 0.000000 │
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