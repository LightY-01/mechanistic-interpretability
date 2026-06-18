import torch
import matplotlib.pyplot as plt
from model import GPT
from generate import ModularTokenizer
from mod_data_gen import ModularArithmeticDataset

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

# Recreate the exact 65% validation split
dataset = ModularArithmeticDataset(p=p)
data = dataset.data  # shape (9409, 4)

torch.manual_seed(42)
train_size = int(0.35 * len(dataset))
shuffled_indices = torch.randperm(len(data))
val_data = data[shuffled_indices[train_size:]]

val_x = val_data[:, :-1].to(device)  # shape (6116, 3)
val_y = val_data[:, 1:].to(device)   # shape (6116, 3)
targets = val_y[:, -1]               # shape (6116,)

# Dictionary to store captured activations
captured_activations = {}

# Hook function to capture residual stream at inputs and outputs
def get_activation_hook(layer_name):
    def hook(module, input_args, output_tensor):
        if layer_name == "layer_1":
            # input_args[0] is the input to Block 0 (which is Layer 0 / Embedding layer)
            captured_activations["layer_0"] = input_args[0].detach()
        captured_activations[layer_name] = output_tensor.detach()
    return hook

# Register hooks
h1 = model.blocks[0].register_forward_hook(get_activation_hook("layer_1"))
h2 = model.blocks[1].register_forward_hook(get_activation_hook("layer_2"))

# Forward pass on the entire validation dataset (triggers hooks)
with torch.no_grad():
    _ = model(val_x)

# Unregister hooks
h1.remove()
h2.remove()

# Process each layer's activations using logit lens
layers = ["layer_0", "layer_1", "layer_2"]
layer_names = ["Layer 0 (Embeddings)", "Layer 1 (Block 0 Output)", "Layer 2 (Block 1 Output)"]

layer_accs = []
layer_probs = []

for layer_name in layers:
    residual_state = captured_activations[layer_name]
    state_at_last_token = residual_state[:, -1, :]  # shape: (6116, model_dim)
    
    with torch.no_grad():
        norm_state = model.ln1(state_at_last_token)
        logits = model.proj(norm_state)  # shape: (6116, vocab_size)
        probs = torch.softmax(logits, dim=-1)  # shape: (6116, vocab_size)
    
    # Calculate accuracy (only considering outcomes 0 to p-1)
    preds = torch.argmax(logits[:, :p], dim=-1)
    accuracy = (preds == targets).float().mean().item()
    layer_accs.append(accuracy)
    
    # Calculate average probability assigned to the correct answer
    correct_probs = probs[torch.arange(len(targets)), targets]
    avg_prob = correct_probs.mean().item()
    layer_probs.append(avg_prob)

# Plot and save statistics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy Plot
ax1.plot(layer_names, [a * 100 for a in layer_accs], marker='o', color='forestgreen', linewidth=2)
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim(-5, 105)
ax1.set_title("Logit Lens: Validation Accuracy by Layer")
ax1.grid(True, linestyle=':', alpha=0.6)

# Probability Plot
ax2.plot(layer_names, layer_probs, marker='s', color='darkorange', linewidth=2)
ax2.set_ylabel("Average Probability of Correct Token")
ax2.set_ylim(-0.05, 1.05)
ax2.set_title("Logit Lens: Avg Correct Token Probability by Layer")
ax2.grid(True, linestyle=':', alpha=0.6)

plt.suptitle("Logit Lens Analysis across Transformer Layers (Validation Batch)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/logit_lens_batched.png')
plt.close()

print("Logit Lens batched plotting complete. Saved chart to results/logit_lens_batched.png")