import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import GPT
from generate import ModularTokenizer
from collections import defaultdict

checkpoint_path = 'results/gpt_checkpoint.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load(checkpoint_path, map_location=device)
hp = ckpt['hyperparams']

p = hp.get('p', 97)
tokenizer = ModularTokenizer(p=p)

# Load the trained model
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

# Define inputs
clean_prompt = "12 45 ="      # Target: 57
corrupted_prompt = "12 23 ="  # Target: 35

clean_target = (12 + 45) % p 
corrupted_target = (12 + 23) % p 

clean_tokens = torch.tensor([tokenizer.encode(clean_prompt)], dtype=torch.long, device=device)
corrupted_tokens = torch.tensor([tokenizer.encode(corrupted_prompt)], dtype=torch.long, device=device)

# Clean Baseline
with torch.no_grad():
    clean_logits = model(clean_tokens)[:, -1, :]

# Corrupted Baseline
with torch.no_grad():
    corr_logits = model(corrupted_tokens)[:, -1, :]

# Measure logit difference
logit_diff = (corr_logits[0, clean_target] - corr_logits[0, corrupted_target]).item()
print(f"Logit Difference: {logit_diff:.4f}")

# Clean Capture (Saving the 'Thoughts') 
clean_z = {}
def make_capture_hook(layer_idx):
    def capture_hook(out):
        # out shape: (B, T, num_heads, head_dim)
        clean_z[layer_idx] = out.clone().detach()
        return out
    return capture_hook

# Register capture hooks
for idx in range(hp['num_blocks']):
    model.blocks[idx].attention.hook_z = make_capture_hook(idx)

# Run clean pass to save activations
with torch.no_grad():
    _ = model(clean_tokens)

# Remove capture hooks
for idx in range(hp['num_blocks']):
    model.blocks[idx].attention.hook_z = None

# Patched Corrupted Run (The 'Brain Swap')
target_layer = 0
target_head = 3

def make_patch_hook(layer_idx, clean_activation):
    def patch_hook(out):
        # out is the corrupted run activation tensor: shape (B, T, num_heads, head_dim)
        # We overwrite only the target head with the clean activation
        out[:, :, target_head, :] = clean_activation[:, :, target_head, :]
        return out
    return patch_hook

# Register the patch hook on Layer 1's attention block
model.blocks[target_layer].attention.hook_z = make_patch_hook(target_layer, clean_z[target_layer])

# Run corrupted pass with the patch hook active
with torch.no_grad():
    patched_logits = model(corrupted_tokens)[:, -1, :]
    patched_probs = torch.softmax(patched_logits, dim=-1)[0]

# Remove the patch hook to restore model state
model.blocks[target_layer].attention.hook_z = None

print(f"Prob of Clean Target ({clean_target}): {patched_probs[clean_target]:.2%}")
print(f"Prob of Corrupted Target ({corrupted_target}): {patched_probs[corrupted_target]:.2%}")
print(f"Prediction: {torch.argmax(patched_logits[:, :p], dim=-1).item()}")

patched_logit_diff = (patched_logits[0, clean_target] - patched_logits[0, corrupted_target]).item()
print(f"Patched Logit Difference: {patched_logit_diff:.4f}")

# Calculate logit difference change
logit_diff_shift = patched_logit_diff - logit_diff
print(f"Shift in Logit Difference: {logit_diff_shift:+.4f}")

# Sweep all heads and perform activation patching
num_layers = hp['num_blocks']
num_heads = hp['num_heads']
results_matrix = np.zeros((num_layers, num_heads))

print(f"Baseline Logit Difference: {logit_diff:.4f}\n")
print(f"{'Layer':<6} | {'Head':<5} | {'Patched Logit Diff':<20} | {'Logit Diff Shift':<18}")
print('\n')

def make_patch_hook(target_head, clean_activation):
    def patch_hook(out):
        # out shape: (B, T, num_heads, head_dim)
        out[:, :, target_head, :] = clean_activation[:, :, target_head, :]
        return out
    return patch_hook

for layer in range(num_layers):
    for head in range(num_heads):
        # Register patch hook for this specific layer & head
        model.blocks[layer].attention.hook_z = make_patch_hook(head, clean_z[layer])
        
        with torch.no_grad():
            patched_logits = model(corrupted_tokens)[:, -1, :]
            
        # Clean up hook
        model.blocks[layer].attention.hook_z = None
        
        # Calculate logit difference and shift
        patched_logit_diff = (patched_logits[0, clean_target] - patched_logits[0, corrupted_target]).item()
        shift = patched_logit_diff - logit_diff
        
        results_matrix[layer, head] = shift
        print(f"Layer {layer:<1} | Head {head:<1} | {patched_logit_diff:<20.4f} | {shift:<+18.4f}")

# Plot Heatmap
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(results_matrix, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')

# Add values to the cells
for l in range(num_layers):
    for h in range(num_heads):
        val = results_matrix[l, h]
        ax.text(h, l, f"{val:+.2f}", ha="center", va="center", fontweight='bold')

# Labels and ticks
ax.set_xticks(range(num_heads))
ax.set_yticks(range(num_layers))
ax.set_xticklabels([f"Head {h}" for h in range(num_heads)])
ax.set_yticklabels([f"Layer {l}" for l in range(num_layers)])
ax.set_xlabel("Attention Heads")
ax.set_ylabel("Transformer Layers")
ax.set_title("Activation Patching Heatmap\n(Logit Difference Shift for clean_target vs corrupted_target)")

fig.colorbar(im, ax=ax, label="Logit Difference Shift")
plt.tight_layout()
plt.savefig('results/activation_patching_heatmap.png')
plt.close()

# Patching Function
def test_combination(heads_to_patch):
    """
    heads_to_patch: list of (layer_idx, head_idx)
    """
    # Group heads by layer
    layer_to_heads = defaultdict(list)
    for layer, head in heads_to_patch:
        layer_to_heads[layer].append(head)
        
    def make_patch_hook(layer_idx, target_heads):
        def patch_hook(out):
            for h in target_heads:
                out[:, :, h, :] = clean_z[layer_idx][:, :, h, :]
            return out
        return patch_hook
    
    for layer_idx, heads in layer_to_heads.items():
        model.blocks[layer_idx].attention.hook_z = make_patch_hook(layer_idx, heads)
        
    with torch.no_grad():
        patched_logits = model(corrupted_tokens)[:, -1, :]
        patched_probs = torch.softmax(patched_logits, dim=-1)[0]
        
    for layer_idx in layer_to_heads.keys():
        model.blocks[layer_idx].attention.hook_z = None
        
    patched_logit_diff = (patched_logits[0, clean_target] - patched_logits[0, corrupted_target]).item()
    shift = patched_logit_diff - logit_diff
    pred_idx = torch.argmax(patched_logits[0]).item()
    
    print(f"Patched combination: {heads_to_patch}")
    print(f"Predicted token: {pred_idx} (Target: {clean_target})")
    print(f"Prob of Clean Target ({clean_target}): {patched_probs[clean_target]:.2%}")
    print(f"Prob of Corrupted Target ({corrupted_target}): {patched_probs[corrupted_target]:.2%}")
    print(f"Logit Diff: {patched_logit_diff:.4f} | Shift: {shift:+.4f}")
    print('\n')

# Test different combinations
print(f"Baseline Corrupted Target Logit Diff: {logit_diff:.4f}\n")
test_combination([(0, 1), (0, 3)])
test_combination([(0, 1), (0, 3), (1, 1)])
test_combination([(0, 3), (1, 1)])
test_combination([(0, 1), (1, 1)])