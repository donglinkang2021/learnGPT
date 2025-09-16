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
│ MHA (Manual)         │    1.4817 ± 0.1908 │                   N/A │            baseline │            -0.000527 │
│ MHA (Flash)          │    0.2800 ± 0.0384 │                   N/A │ 0.001853 ± 0.000000 │  0.000525 ± 0.000000 │
│ GQA (kv=4, Manual)   │    1.4199 ± 0.1882 │                   N/A │ 0.001850 ± 0.000000 │  0.000246 ± 0.000000 │
│ GQA (kv=4, Flash)    │    0.2399 ± 0.0489 │                   N/A │ 0.001843 ± 0.000000 │  0.000015 ± 0.000000 │
│ MQA (kv=1, Manual)   │    1.3360 ± 0.1076 │                   N/A │ 0.001920 ± 0.000000 │ -0.000477 ± 0.000000 │
│ MQA (kv=1, Flash)    │    0.2000 ± 0.0616 │                   N/A │ 0.001860 ± 0.000000 │  0.000147 ± 0.000000 │
│ Linear (Vectorized)  │    1.1929 ± 0.0620 │                   N/A │ 0.001784 ± 0.000000 │ -0.000463 ± 0.000000 │
│ Linear (Loop)        │    9.9806 ± 0.7758 │                   N/A │ 0.001828 ± 0.000000 │  0.000372 ± 0.000000 │
│ Linear (Lucidrains)  │    0.3597 ± 0.0484 │                   N/A │ 0.001304 ± 0.000000 │ -0.000062 ± 0.000000 │
│ SoftmaxLinear        │    1.2586 ± 0.0518 │                   N/A │ 0.001745 ± 0.000000 │ -0.000149 ± 0.000000 │
│ (Vectorized)         │                    │                       │                     │                      │
│ SoftmaxLinear (Loop) │   12.3803 ± 1.4508 │                   N/A │ 0.001882 ± 0.000000 │ -0.000132 ± 0.000000 │
└──────────────────────┴────────────────────┴───────────────────────┴─────────────────────┴──────────────────────┘
```

- GPU 4090x1

```bash
╭──────────── Test Configuration ─────────────╮
│ Device: cuda                                │
│ Tensor Shape: x: torch.Size([4, 1024, 512]) │
│ Data Type: x: torch.float32                 │
│ Initial Memory: 45.25 MB                    │
│ Number of Trials: 5                         │
╰─────────────────────────────────────────────╯
Running warm-up iteration...
Warm-up complete. Starting profiling...
                                       Causal Attention Function Profiling                                        
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Function Name        ┃ Execution Time (s) ┃ Peak Memory Used (MB) ┃     MSE vs baseline ┃          Mean Result ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ MHA (Manual)         │    0.0061 ± 0.0015 │         569.00 ± 0.00 │            baseline │             0.000144 │
│ MHA (Flash)          │    0.0138 ± 0.0260 │          40.25 ± 0.00 │ 0.001847 ± 0.000000 │ -0.000232 ± 0.000000 │
│ GQA (kv=4, Manual)   │    0.0057 ± 0.0015 │         553.00 ± 0.00 │ 0.001870 ± 0.000000 │ -0.000553 ± 0.000000 │
│ GQA (kv=4, Flash)    │    0.0007 ± 0.0001 │          40.25 ± 0.00 │ 0.001827 ± 0.000000 │ -0.000078 ± 0.000000 │
│ MQA (kv=1, Manual)   │    0.0055 ± 0.0007 │         554.00 ± 0.00 │ 0.001838 ± 0.000000 │  0.000083 ± 0.000000 │
│ MQA (kv=1, Flash)    │    0.0010 ± 0.0003 │          25.25 ± 0.00 │ 0.001800 ± 0.000000 │  0.000162 ± 0.000000 │
│ Linear (Vectorized)  │    0.0092 ± 0.0070 │         816.00 ± 0.00 │ 0.001736 ± 0.000000 │  0.000288 ± 0.000000 │
│ Linear (Loop)        │    0.2359 ± 0.0028 │          96.76 ± 0.00 │ 0.001765 ± 0.000000 │ -0.000151 ± 0.000000 │
│ Linear (Lucidrains)  │    0.0100 ± 0.0182 │         105.38 ± 0.00 │ 0.001306 ± 0.000000 │ -0.000096 ± 0.000000 │
│ SoftmaxLinear        │    0.0067 ± 0.0006 │         824.02 ± 0.00 │ 0.000966 ± 0.000000 │ -0.000049 ± 0.000000 │
│ (Vectorized)         │                    │                       │                     │                      │
│ SoftmaxLinear (Loop) │    0.2403 ± 0.0047 │         368.27 ± 0.00 │ 0.000965 ± 0.000000 │  0.000020 ± 0.000000 │
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