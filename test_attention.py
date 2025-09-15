import torch
from src.models.attention import MHA, GQA, MQA, LinearAttention, LucidLinearAttention, SoftmaxLinearAttention
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
    mha_flash = MHA(d_model=d_model, n_heads=n_heads, use_flash_attention=True).to(device)
    mha_manual = MHA(d_model=d_model, n_heads=n_heads, use_flash_attention=False).to(device)
    gqa_flash = GQA(d_model=d_model, n_heads=n_heads, kv_heads=kv_heads, use_flash_attention=True).to(device)
    gqa_manual = GQA(d_model=d_model, n_heads=n_heads, kv_heads=kv_heads, use_flash_attention=False).to(device)
    mqa_flash = MQA(d_model=d_model, n_heads=n_heads, use_flash_attention=True).to(device)
    mqa_manual = MQA(d_model=d_model, n_heads=n_heads, use_flash_attention=False).to(device)
    linear_attn_vec = LinearAttention(d_model=d_model, n_heads=n_heads, use_loop_impl=False).to(device)
    linear_attn_loop = LinearAttention(d_model=d_model, n_heads=n_heads, use_loop_impl=True).to(device)
    lucid_linear_attn = LucidLinearAttention(d_model=d_model, n_heads=n_heads, bucket_size=64).to(device)
    softmax_linear_attn_vec = SoftmaxLinearAttention(d_model=d_model, n_heads=n_heads, use_loop_impl=False).to(device)
    softmax_linear_attn_loop = SoftmaxLinearAttention(d_model=d_model, n_heads=n_heads, use_loop_impl=True).to(device)

    # --- Prepare Inputs ---
    inputs = {
        'x': torch.randn(batch_size, seq_len, d_model, device=device)
    }
    
    # --- Functions to Test ---
    # The first function in the list is treated as the baseline for comparison.
    functions_to_test = [
        named_partial(mha_manual.forward, "MHA (Manual)"),
        named_partial(mha_flash.forward, "MHA (Flash)"),
        named_partial(gqa_manual.forward, f"GQA (kv={kv_heads}, Manual)"),
        named_partial(gqa_flash.forward, f"GQA (kv={kv_heads}, Flash)"),
        named_partial(mqa_manual.forward, "MQA (kv=1, Manual)"),
        named_partial(mqa_flash.forward, "MQA (kv=1, Flash)"),
        named_partial(linear_attn_vec.forward, "Linear (Vectorized)"),
        named_partial(linear_attn_loop.forward, "Linear (Loop)"),
        named_partial(lucid_linear_attn.forward, "Linear (Lucidrains)"),
        named_partial(softmax_linear_attn_vec.forward, "SoftmaxLinear (Vectorized)"),
        named_partial(softmax_linear_attn_loop.forward, "SoftmaxLinear (Loop)"),
    ]
    
    # --- Run Performance Test ---
    test_performance(
        inputs=inputs,
        func_prefix="Causal Attention",
        functions_to_test=functions_to_test,
        device=device,
        num_trials=num_trials
    )

if __name__ == "__main__":
    run_attention_benchmark()