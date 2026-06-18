import torch
import matplotlib.pyplot as plt
from model import GPT
from generate import ModularTokenizer

checkpoint_path = 'results/gpt_checkpoint.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load(checkpoint_path, map_location=device)
hp = ckpt['hyperparams']

p = hp.get('p', 97)
tokenizer = ModularTokenizer(p=p)

model = GPT(
    vocab_size     = hp['vocab_size'],
    model_dim      = hp['embed_dim'],
    num_heads      = hp['num_heads'],
    num_kv_heads   = hp['num_kv_heads'],
    context_length = hp['context_length'],
    num_blocks     = hp['num_blocks'],
    use_mlp        = hp.get('use_mlp', False),
).to(device)

model.load_state_dict(ckpt['model_state_dict'])
model.eval()

captured_activations = {}

def get_activation_hook(layer_name):
    def hook(module, input_args, output_tensor):
        if layer_name == "layer_1":
            captured_activations["layer_0"] = input_args[0].detach()
        captured_activations[layer_name] = output_tensor.detach()
    return hook

h1 = model.blocks[0].register_forward_hook(get_activation_hook("layer_1"))
h2 = model.blocks[1].register_forward_hook(get_activation_hook("layer_2"))

# sample equation
a, b = 12, 45
correct_answer = (a + b) % p
prompt = f"{a} {b} ="
input_tokens = tokenizer.encode(prompt)
x = torch.tensor([input_tokens], dtype=torch.long, device=device)

# Forward pass (triggers hooks)
with torch.no_grad():
    _ = model(x)

# Unregister hooks
h1.remove()
h2.remove()

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
layers = ["layer_0", "layer_1", "layer_2"]
layer_titles = ["Layer 0 (Embeddings)", "Layer 1 (Block 0 Output)", "Layer 2 (Block 1 Output)"]

for idx, (layer_name, title) in enumerate(zip(layers, layer_titles)):
    residual_state = captured_activations[layer_name]
    state_at_last_token = residual_state[:, -1, :]  # shape: (1, model_dim)
    
    with torch.no_grad():
        norm_state = model.ln1(state_at_last_token)
        logits = model.proj(norm_state)  # shape: (1, vocab_size)
        probs = torch.softmax(logits, dim=-1)[0]  # shape: (vocab_size,)
    
    # = is not used
    num_probs = probs[:p].cpu().numpy()
    
    axes[idx].bar(range(p), num_probs, color='royalblue', alpha=0.8, edgecolor='none')
    axes[idx].axvline(x=correct_answer, color='crimson', linestyle='--', linewidth=1.5, 
                      label=f"Correct Answer ({correct_answer})")
    
    # Get top prediction
    top_val, top_idx = torch.max(probs[:p], dim=-1)
    top_idx = top_idx.item()
    top_val = top_val.item()
    
    axes[idx].set_title(f"{title} | Top Pred: {top_idx}")
    axes[idx].set_ylabel("Probability")
    axes[idx].grid(True, linestyle=':', alpha=0.5)
    axes[idx].legend(loc="upper right")

axes[-1].set_xlabel(f"Modulo Number (0 to {p-1})")
plt.suptitle(f"Logit Lens: {prompt} (Correct: {correct_answer})", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/logit_lens_dist.png')
plt.close()

print(f"Logit Lens plotting complete. Saved distribution chart to results/logit_lens_dist.png")