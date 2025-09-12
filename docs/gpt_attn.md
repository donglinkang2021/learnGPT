# gpt_attn

modify `gpt` to `gpt_attn`

```diff
[root:learnGPT]$ diff -u src/models/gpt.py src/models/gpt_attn.py 
--- src/models/gpt.py   2025-09-12 14:54:05.902414000 +0000
+++ src/models/gpt_attn.py      2025-09-12 14:51:41.296926078 +0000
@@ -1,50 +1,16 @@
 import torch
 import torch.nn as nn
 import torch.nn.functional as F
+from src.models.attention import MHA, GQA, MQA, LinearAttention
 
-__all__ = ['GPT']
+__all__ = ['GPT_attn']
 
-class Head(nn.Module):
-    """ one head of self-attention """
-
-    def __init__(self, n_embd, head_size, block_size, dropout):
-        super().__init__()
-        self.key = nn.Linear(n_embd, head_size, bias=False)
-        self.query = nn.Linear(n_embd, head_size, bias=False)
-        self.value = nn.Linear(n_embd, head_size, bias=False)
-        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
-
-        self.dropout = nn.Dropout(dropout)
-
-    def forward(self, x):
-        # input of size (batch, time-step, channels)
-        # output of size (batch, time-step, head size)
-        B,T,C = x.shape
-        k = self.key(x)   # (B,T,hs)
-        q = self.query(x) # (B,T,hs)
-        # compute attention scores ("affinities")
-        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
-        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
-        wei = F.softmax(wei, dim=-1) # (B, T, T)
-        wei = self.dropout(wei)
-        # perform the weighted aggregation of the values
-        v = self.value(x) # (B,T,hs)
-        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
-        return out
-
-class MultiHeadAttention(nn.Module):
-    """ multiple heads of self-attention in parallel """
-
-    def __init__(self, num_heads, n_embd, head_size, block_size, dropout):
-        super().__init__()
-        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
-        self.proj = nn.Linear(head_size * num_heads, n_embd)
-        self.dropout = nn.Dropout(dropout)
-
-    def forward(self, x):
-        out = torch.cat([h(x) for h in self.heads], dim=-1)
-        out = self.dropout(self.proj(out))
-        return out
+ATTENTION_REGISTRY = {
+    "mha": MHA,
+    "gqa": GQA,
+    "mqa": MQA,
+    "linear": LinearAttention,
+}
 
 class FeedFoward(nn.Module):
     """ a simple linear layer followed by a non-linearity """
@@ -64,11 +30,13 @@
 class Block(nn.Module):
     """ Transformer block: communication followed by computation """
 
-    def __init__(self, n_embd, n_head, block_size, dropout):
+    def __init__(self, n_embd, n_head, block_size, dropout, attention_type="mha", **kwargs):
         # n_embd: embedding dimension, n_head: the number of heads we'd like
         super().__init__()
-        head_size = n_embd // n_head
-        self.sa = MultiHeadAttention(n_head, n_embd, head_size, block_size, dropout)
+        if attention_type not in ATTENTION_REGISTRY:
+            raise ValueError(f"Unknown attention type: {attention_type}, please choose from {ATTENTION_REGISTRY.keys()}")
+        
+        self.sa = ATTENTION_REGISTRY[attention_type](d_model=n_embd, n_heads=n_head, **kwargs)
         self.ffwd = FeedFoward(n_embd, dropout)
         self.ln1 = nn.LayerNorm(n_embd)
         self.ln2 = nn.LayerNorm(n_embd)
@@ -78,14 +46,14 @@
         x = x + self.ffwd(self.ln2(x))
         return x
 
-class GPT(nn.Module):
+class GPT_attn(nn.Module):
 
-    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head, dropout):
+    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head, dropout, attention_type="mha", **kwargs):
         super().__init__()
         # each token directly reads off the logits for the next token from a lookup table
         self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
         self.position_embedding_table = nn.Embedding(block_size, n_embd)
-        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
+        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout, attention_type, **kwargs) for _ in range(n_layer)])
         self.ln_f = nn.LayerNorm(n_embd) # final layer norm
         self.lm_head = nn.Linear(n_embd, vocab_size)
 
@@ -121,4 +89,3 @@
 
         return logits, loss
 
```
