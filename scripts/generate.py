import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Optional
from model import KVCache

class ModularTokenizer:
    def __init__(self, p: int = 97):
        self.p = p
        self.eq_token = p
        self.vocab_size = p + 1

    def encode(self, text: str) -> list[int]:
        tokens = []
        for word in text.strip().split():
            if word == '=':
                tokens.append(self.eq_token)
            else:
                tokens.append(int(word))
        return tokens

    def decode(self, tokens: list[int]) -> str:
        parts = []
        for t in tokens:
            if t == self.eq_token:
                parts.append('=')
            else:
                parts.append(str(t))
        return " ".join(parts)

@torch.no_grad()
def generate_text(
    model: nn.Module, 
    prompt: str, 
    tokenizer: ModularTokenizer, 
    max_new_tokens: int, 
    context_length: int, 
    temperature: float = 1.0, 
    top_k: Optional[int] = None, 
    device: str = "cpu"
) -> str:
    """Generates next tokens for modular arithmetic autoregressively."""
    model.eval()
    
    encoded = tokenizer.encode(prompt)
    x = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    num_layers = len(model.blocks) if hasattr(model, 'blocks') else 2
    kv_caches = [KVCache(context_length=context_length) for _ in range(num_layers)]
    
    for step in range(max_new_tokens):
        x_cond = x[:, -context_length:]
        if step > 0:
            x_cond = x[:, -1:]
        
        logits = model(x_cond, kv_caches=kv_caches) 
        logits = logits[:, -1, :] 
        
        if temperature > 0.0:
            logits = logits / temperature
            
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)
        
    return tokenizer.decode(x[0].tolist())
