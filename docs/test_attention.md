# 测试各个attention

- 目前的主要问题是所实现的linear-attention显存消耗太高

- CPUx4

```bash
╭──────────── Test Configuration ─────────────╮
│ Device: cpu                                 │
│ Tensor Shape: x: torch.Size([4, 1024, 512]) │
│ Data Type: x: torch.float32                 │
│ Initial Memory: N/A                         │
│ Number of Trials: 5                         │
╰─────────────────────────────────────────────╯
Running warm-up iteration...
Warm-up complete. Starting profiling...
                                       Causal Attention Function Profiling                                        
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Function Name        ┃ Execution Time (s) ┃ Peak Memory Used (MB) ┃     MSE vs baseline ┃          Mean Result ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ MHA (Manual)         │    2.9104 ± 0.2447 │                   N/A │            baseline │            -0.000009 │
│ MHA (Flash)          │    0.6175 ± 0.0402 │                   N/A │ 0.001863 ± 0.000000 │ -0.000124 ± 0.000000 │
│ GQA (kv=4, Flash)    │    1.2395 ± 0.1303 │                   N/A │ 0.001871 ± 0.000000 │  0.000233 ± 0.000000 │
│ GQA (kv=4, Manual)   │    3.0600 ± 0.0822 │                   N/A │ 0.001852 ± 0.000000 │ -0.000080 ± 0.000000 │
│ MQA (kv=1, Flash)    │    0.8996 ± 0.1241 │                   N/A │ 0.001917 ± 0.000000 │  0.000437 ± 0.000000 │
│ MQA (kv=1, Manual)   │    2.9814 ± 0.1461 │                   N/A │ 0.001834 ± 0.000000 │ -0.000305 ± 0.000000 │
│ Linear (Vectorized)  │    3.2011 ± 0.0678 │                   N/A │ 0.001739 ± 0.000000 │ -0.000436 ± 0.000000 │
│ Linear (Loop)        │  45.8211 ± 13.3552 │                   N/A │ 0.001742 ± 0.000000 │  0.000002 ± 0.000000 │
│ Linear (Lucidrains)  │    2.8396 ± 1.2065 │                   N/A │ 0.001301 ± 0.000000 │  0.000272 ± 0.000000 │
└──────────────────────┴────────────────────┴───────────────────────┴─────────────────────┴──────────────────────┘
```

- GPU 4090x1

```bash
╭──────────── Test Configuration ─────────────╮
│ Device: cuda                                │
│ Tensor Shape: x: torch.Size([4, 1024, 512]) │
│ Data Type: x: torch.float32                 │
│ Initial Memory: 37.25 MB                    │
│ Number of Trials: 5                         │
╰─────────────────────────────────────────────╯
Running warm-up iteration...
Warm-up complete. Starting profiling...
                                       Causal Attention Function Profiling                                        
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Function Name        ┃ Execution Time (s) ┃ Peak Memory Used (MB) ┃     MSE vs baseline ┃          Mean Result ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ MHA (Manual)         │    0.0060 ± 0.0015 │         569.00 ± 0.00 │            baseline │            -0.000090 │
│ MHA (Flash)          │    0.0013 ± 0.0011 │          40.25 ± 0.00 │ 0.001811 ± 0.000000 │  0.000219 ± 0.000000 │
│ GQA (kv=4, Flash)    │    0.0010 ± 0.0000 │          40.25 ± 0.00 │ 0.001762 ± 0.000000 │ -0.000057 ± 0.000000 │
│ GQA (kv=4, Manual)   │    0.0053 ± 0.0014 │         553.00 ± 0.00 │ 0.001803 ± 0.000000 │ -0.000191 ± 0.000000 │
│ MQA (kv=1, Flash)    │    0.0010 ± 0.0004 │          25.25 ± 0.00 │ 0.001797 ± 0.000000 │  0.000057 ± 0.000000 │
│ MQA (kv=1, Manual)   │    0.0058 ± 0.0006 │         554.00 ± 0.00 │ 0.001765 ± 0.000000 │  0.000217 ± 0.000000 │
│ Linear (Vectorized)  │    0.0088 ± 0.0063 │         816.00 ± 0.00 │ 0.001755 ± 0.000000 │ -0.000247 ± 0.000000 │
│ Linear (Loop)        │    0.2336 ± 0.0022 │          96.76 ± 0.00 │ 0.001747 ± 0.000000 │ -0.000500 ± 0.000000 │
│ Linear (Lucidrains)  │    0.0099 ± 0.0180 │         106.03 ± 0.34 │ 0.001259 ± 0.000000 │  0.000022 ± 0.000000 │
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