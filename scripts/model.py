import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List, Optional, Tuple

class KVCache:
    def __init__(self, context_length: int):
        # (batch, num_kv_heads, seq_len, head_dim)
        self.cache_k: Optional[torch.Tensor] = None  
        self.cache_v: Optional[torch.Tensor] = None
        self.context_length = context_length 

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_k is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.cat((self.cache_k, new_k), dim=2)
            self.cache_v = torch.cat((self.cache_v, new_v), dim=2)

        if self.cache_k.shape[2] > self.context_length:
            self.cache_k = self.cache_k[:, :, -self.context_length:, :]
            self.cache_v = self.cache_v[:, :, -self.context_length:, :]

        return (self.cache_k, self.cache_v)

    def clear(self):
        self.cache_k = None
        self.cache_v = None


class GPT(nn.Module):
    def __init__(self, vocab_size: int, model_dim: int, num_heads: int, num_kv_heads: int, context_length: int, num_blocks: int = 4, use_mlp: bool = False):
        super().__init__()
        # Store context_length for positional embedding logic
        self.context_length = context_length 
        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, model_dim)
        # Positional embeddings
        self.pos_embeddings = nn.Embedding(context_length, model_dim)
        # Transformer blocks with Grouped Query Attention and Vanilla Neural Network
        self.blocks = nn.ModuleList([self.TransformerBlock(model_dim, num_heads, num_kv_heads, use_mlp) for _ in range(num_blocks)])
        # Layer normalization
        self.ln1 = nn.LayerNorm(model_dim)
        # Projection to vocabulary size
        self.proj = nn.Linear(model_dim, vocab_size)

    def forward(self, context: TensorType[int], kv_caches: Optional[List[KVCache]] = None) -> TensorType[float]:
        B, T = context.shape
        embeddings = self.word_embeddings(context)
        
        prev_len = 0
        if kv_caches is not None and len(kv_caches) > 0 and kv_caches[0].cache_k is not None:
            prev_len = kv_caches[0].cache_k.shape[2]
            
        start_pos = min(prev_len, self.context_length - T)
        positions = torch.arange(start_pos, start_pos + T, device=context.device)
        embeddings = embeddings + self.pos_embeddings(positions)
        
        x = embeddings
        for i, block in enumerate(self.blocks):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, kv_cache)
            
        result = self.ln1(x)
        logits = self.proj(result)
        return logits

    # Transformer block with Grouped Query Attention and Vanilla Neural Network
    class TransformerBlock(nn.Module):

        class GroupedQueryAttention(nn.Module):
            def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
                super().__init__()
                self.num_heads = num_heads
                self.num_kv_heads = num_kv_heads
                self.head_dim = model_dim // num_heads
                
                # Linear projections for query, key, and value
                self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
                self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
                self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
                
                # Output projection
                self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

            def forward(self, x: TensorType[float], kv_cache: Optional[KVCache] = None) -> TensorType[float]:
                B, T, D = x.shape
                q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

                if kv_cache is not None:
                    k, v = kv_cache.update(k, v)

                # Repeat keys and values to match the number of query heads
                repeats = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(repeats, dim=1)
                v = v.repeat_interleave(repeats, dim=1)

                # Scaled dot-product attention
                scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
                T_total = k.shape[2]
                mask = torch.tril(torch.ones(T, T, device=x.device))
                if T_total > T:
                    padding = torch.ones(T, T_total - T, device=x.device)
                    mask = torch.cat([padding, mask], dim=-1)
                scores = scores.masked_fill(mask == 0, float('-inf'))
                weights = torch.softmax(scores, dim=-1)

                out = (weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
                proj = self.output_proj(out)
                return proj
        
        class VanillaNeuralNetwork(nn.Module):

            def __init__(self, model_dim: int):
                super().__init__()
                self.up_projection = nn.Linear(model_dim, model_dim * 4)
                self.relu = nn.ReLU()
                self.down_projection = nn.Linear(model_dim * 4, model_dim)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x: TensorType[float]) -> TensorType[float]:
                return self.dropout(self.down_projection(self.relu(self.up_projection(x))))

        def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int, use_mlp: bool = False):
            super().__init__()
            self.attention = self.GroupedQueryAttention(model_dim, num_heads, num_kv_heads)
            self.use_mlp = use_mlp
            if use_mlp:
                self.linear_network = self.VanillaNeuralNetwork(model_dim)
                self.second_norm = nn.LayerNorm(model_dim)
            self.first_norm = nn.LayerNorm(model_dim)

        def forward(self, embedded: TensorType[float], kv_cache: Optional[KVCache] = None) -> TensorType[float]:
            embedded = embedded + self.attention(self.first_norm(embedded), kv_cache)
            if self.use_mlp:
                embedded = embedded + self.linear_network(self.second_norm(embedded))
            return embedded