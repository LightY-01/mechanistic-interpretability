import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from model import GPT
from generate import ModularTokenizer
from mod_data_gen import ModularArithmeticDataset

checkpoint_path = 'results/gpt_checkpoint.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load(checkpoint_path, map_location=device)
hp = ckpt['hyperparams']

p = hp.get('p', 97)
tokenizer = ModularTokenizer(p=p)

# Load model
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

# Recreate validation split
dataset = ModularArithmeticDataset(p=p)
data = dataset.data  # (9409, 4)

torch.manual_seed(42)
train_size = int(0.35 * len(dataset))
shuffled_indices = torch.randperm(len(data))
val_data = data[shuffled_indices[train_size:]]  # validation set

# Sample 100 prompts
torch.manual_seed(999)  # Seed for prompt generation
sampled_indices = torch.randint(0, len(val_data), (100,))
eval_prompts = val_data[sampled_indices]  # Shape: (100, 4)

# Hook variables
clean_z = {}
def make_capture_hook(layer_idx):
    def capture_hook(out):
        clean_z[layer_idx] = out.clone().detach()
        return out
    return capture_hook

def make_patch_hook(layer_idx, target_heads):
    def patch_hook(out):
        for h in target_heads:
            out[:, :, h, :] = clean_z[layer_idx][:, :, h, :]
        return out
    return patch_hook

# Helper function to run one prompt evaluation
def evaluate_patching(clean_toks, corr_toks, c_clean, c_corr, heads_to_patch=None, target_head_single=None):
    """
    Evaluates patching and returns (patched_logit_diff, shift, pred_flipped_to_clean, clean_prob)
    """
    # Baseline Corrupted Run
    with torch.no_grad():
        corr_logits = model(corr_toks)[:, -1, :]
    baseline_logit_diff = (corr_logits[0, c_clean] - corr_logits[0, c_corr]).item()
    
    # Capture Clean Run
    for idx in range(hp['num_blocks']):
        model.blocks[idx].attention.hook_z = make_capture_hook(idx)
    with torch.no_grad():
        _ = model(clean_toks)
    for idx in range(hp['num_blocks']):
        model.blocks[idx].attention.hook_z = None
        
    # Patching
    if target_head_single is not None:
        layer, head = target_head_single
        model.blocks[layer].attention.hook_z = make_patch_hook(layer, [head])
    elif heads_to_patch is not None:
        layer_to_heads = defaultdict(list)
        for l, h in heads_to_patch:
            layer_to_heads[l].append(h)
        for l, heads in layer_to_heads.items():
            model.blocks[l].attention.hook_z = make_patch_hook(l, heads)
            
    # Run Patched
    with torch.no_grad():
        patched_logits = model(corr_toks)[:, -1, :]
        patched_probs = torch.softmax(patched_logits, dim=-1)[0]
        
    # Unregister hooks
    for idx in range(hp['num_blocks']):
        model.blocks[idx].attention.hook_z = None
        
    patched_logit_diff = (patched_logits[0, c_clean] - patched_logits[0, c_corr]).item()
    shift = patched_logit_diff - baseline_logit_diff
    pred = torch.argmax(patched_logits[0]).item()
    
    flipped = (pred == c_clean)
    clean_prob = patched_probs[c_clean].item()
    
    return shift, flipped, clean_prob

# EVALUATIONS
num_layers = hp['num_blocks']
num_heads = hp['num_heads']

print("Starting evaluations over 100 validation prompts...\n")

for mode in ['a', 'b']:
    print(f"EVALUATING CIRCUIT SPECIALIZATION FOR OPERAND: {mode.upper()}")
    print("-"*60)
    
    single_head_shifts = np.zeros((num_layers, num_heads))
    
    # Prepare datasets for 100 runs
    prompt_runs = []
    for row in eval_prompts:
        a_val, b_val = row[0].item(), row[1].item()
        
        if mode == 'a':
            # Change A, keep B same
            a_corr = a_val
            while a_corr == a_val:
                a_corr = torch.randint(0, p, (1,)).item()
            
            clean_str = f"{a_val} {b_val} ="
            corr_str = f"{a_corr} {b_val} ="
            c_clean = (a_val + b_val) % p
            c_corr = (a_corr + b_val) % p
        else:
            # Change B, keep A same
            b_corr = b_val
            while b_corr == b_val:
                b_corr = torch.randint(0, p, (1,)).item()
                
            clean_str = f"{a_val} {b_val} ="
            corr_str = f"{a_val} {b_corr} ="
            c_clean = (a_val + b_val) % p
            c_corr = (a_val + b_corr) % p
            
        clean_toks = torch.tensor([tokenizer.encode(clean_str)], dtype=torch.long, device=device)
        corr_toks = torch.tensor([tokenizer.encode(corr_str)], dtype=torch.long, device=device)
        
        prompt_runs.append((clean_toks, corr_toks, c_clean, c_corr))
        
    # Part 1: Single Head Sweep
    for l in range(num_layers):
        for h in range(num_heads):
            shifts = []
            for clean_toks, corr_toks, c_clean, c_corr in prompt_runs:
                shift, _, _ = evaluate_patching(clean_toks, corr_toks, c_clean, c_corr, target_head_single=(l, h))
                shifts.append(shift)
            single_head_shifts[l, h] = np.mean(shifts)
            
    # Print individual average shifts
    print("\nAverage Logit Difference Shift by Head:")
    print(f"{'Layer':<6} | {'Head':<5} | {'Avg Shift':<12}")
    print("-" * 30)
    for l in range(num_layers):
        for h in range(num_heads):
            print(f"Layer {l} | Head {h} | {single_head_shifts[l, h]:+.4f}")
            
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(single_head_shifts, cmap='RdYlBu_r', aspect='auto')
    for l in range(num_layers):
        for h in range(num_heads):
            val = single_head_shifts[l, h]
            ax.text(h, l, f"{val:+.2f}", ha="center", va="center", fontweight='bold')
    ax.set_xticks(range(num_heads))
    ax.set_yticks(range(num_layers))
    ax.set_xticklabels([f"Head {h}" for h in range(num_heads)])
    ax.set_yticklabels([f"Layer {l}" for l in range(num_layers)])
    ax.set_xlabel("Attention Heads")
    ax.set_ylabel("Layers")
    ax.set_title(f"Avg Logit Difference Shift (Operand {mode.upper()} Changed)")
    fig.colorbar(im, ax=ax, label="Logit Diff Shift")
    plt.tight_layout()
    plt.savefig(f'results/patching_heatmap_{mode}.png')
    plt.close()
    print(f"\nHeatmap saved to results/patching_heatmap_{mode}.png")
    
    # Part 2: Combination Evaluation (Searching for Global Circuits)
    print("\nEvaluating Candidate Global Circuits (Average over 100 runs):")
    if mode == 'a':
        candidates = [
            ("L0H1, L0H2, L1H1", [(0, 1), (0, 2), (1, 1)]),
            ("L0H1, L0H2, L1H0, L1H1", [(0, 1), (0, 2), (1, 0), (1, 1)]),
            ("All L0 Heads + L1H3 & L1H1", [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (1, 1)]),
            ("All L0 Heads + L1H0 & L1H1", [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]),
            ("All L0 Heads + L1H3, L1H1, L1H0", [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (1, 1), (1, 0)])
        ]
    else:
        candidates = [
            ("L0H1, L0H3, L1H1", [(0, 1), (0, 3), (1, 1)]),
            ("All L0 Heads + L1H1", [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1)]),
            ("All L0 Heads + L1H3 & L1H1", [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (1, 1)]),
            ("All L0 Heads + L1H3, L1H1, L1H0", [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (1, 1), (1, 0)])
        ]
        
    for name, combo in candidates:
        combo_shifts = []
        flips = 0
        probs = []
        for clean_toks, corr_toks, c_clean, c_corr in prompt_runs:
            shift, flipped, clean_prob = evaluate_patching(clean_toks, corr_toks, c_clean, c_corr, heads_to_patch=combo)
            combo_shifts.append(shift)
            if flipped:
                flips += 1
            probs.append(clean_prob)
            
        print(f"Combo: {name:<35} | Recovery Acc: {flips/100:.1%} | Avg Target Prob: {np.mean(probs):.2%} | Avg Shift: {np.mean(combo_shifts):+.4f}")
    print()
