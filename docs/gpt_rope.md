# gpt_rope

modify `gpt_v3` to `gpt_rope`

```diff
[root:learnGPT]$ diff -u src/models/gpt_v3.py src/models/gpt_rope.py 
--- src/models/gpt_v3.py        2025-09-10 04:27:55.282886000 +0000
+++ src/models/gpt_rope.py      2025-09-11 08:51:33.778434000 +0000
@@ -1,10 +1,10 @@
 import torch
 import torch.nn as nn
 from torch.nn import functional as F
-from .utils import get_pe
+from .simplepe.rope import precompute_freqs_cis, apply_rotary_emb
 import math
 
-__all__ = ['GPT_v3']
+__all__ = ['GPT_rope']
 
 # Transformer Language Model (*exactly* as used in GPT-2)
 
@@ -38,13 +38,15 @@
         self.n_head = n_head
         self.n_embd = n_embd
 
-    def forward(self, x):
+    def forward(self, x, freqs_cis):
         B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
         # calculate query, key, values for all heads in batch and move head forward to be the batch dim
         q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
         k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
         q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
         v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
+        q = apply_rotary_emb(q, freqs_cis)
+        k = apply_rotary_emb(k, freqs_cis)
         y = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=True, dropout_p=self.attn_dropout)
         y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
         y = self.proj_dropout(self.c_proj(y)) # (B, T, C)
@@ -65,25 +67,27 @@
             nn.Dropout(dropout),
         )
 
-    def forward(self, x):
-        x = x + self.attn(self.ln_1(x))
+    def forward(self, x, freqs_cis):
+        x = x + self.attn(self.ln_1(x), freqs_cis)
         x = x + self.mlpf(self.ln_2(x))
         return x
 
-class GPT_v3(nn.Module):
+class GPT_rope(nn.Module):
     """ Transformer Language Model, exactly as seen in GPT-2 """
 
-    def __init__(self, n_layer:int, n_head:int, n_embd:int, block_size:int, vocab_size:int, dropout:float, pe_type:str='randn'):
+    def __init__(self, n_layer:int, n_head:int, n_embd:int, block_size:int, vocab_size:int, dropout:float):
         super().__init__()
         self.block_size = block_size
 
         self.transformer = nn.ModuleDict(dict(
             wte = nn.Embedding(vocab_size, n_embd),
-            wpe = get_pe(pe_type, n_embd, block_size),
             h = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)]),
             ln_f = nn.LayerNorm(n_embd),
         ))
         self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
+        self.freqs_cis = precompute_freqs_cis(
+            n_embd // n_head, block_size*2, theta=10000.0
+        )
 
         self.apply(self._init_weights)
 
@@ -97,9 +101,10 @@
 
     def forward(self, idx, targets=None):
         # forward the GPT model itself
-        x = self.transformer.wpe(self.transformer.wte(idx)) # (B,T,C)
+        x = self.transformer.wte(idx) # (B,T,C)
+        freqs_cis = self.freqs_cis[:x.size(1)].to(x.device)
         for block in self.transformer.h:
-            x = block(x)
+            x = block(x, freqs_cis)
         x = self.transformer.ln_f(x)
         logits = self.lm_head(x)
```