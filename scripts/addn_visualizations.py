import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import GPT
from generate import ModularTokenizer

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

# Ensure results folder exists
os.makedirs('results', exist_ok=True)

# Representational Fourier Circles
W_E = model.word_embeddings.weight.detach() # shape: (vocab_size, model_dim)
W_E_numbers = W_E[:p, :] # shape: (p, model_dim)

# Compute 1D FFT of the embeddings along the vocab/token dimension
fft_WE = torch.fft.fft(W_E_numbers, dim=0) # shape: (p, model_dim)

# We will plot the circles for our key frequencies k=10, k=26, k=45
# and a control frequency (k=5) where the model is not active.
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

frequencies_to_plot = [10, 26, 45, 3]

for idx, k in enumerate(frequencies_to_plot):
    ax = axes[idx]
    
    # Get the complex Fourier vector for frequency k
    # This represents the coordinates of the 2D subspace representing frequency k
    F_k = fft_WE[k] # shape: (model_dim,)
    
    # Project W_E onto the real and imaginary parts of F_k
    proj_real = (W_E_numbers @ F_k.real).cpu().numpy()
    proj_imag = (W_E_numbers @ F_k.imag).cpu().numpy()
    
    ax.plot(proj_real, proj_imag, color='lightgray', linestyle='-', zorder=1)
    scatter = ax.scatter(proj_real, proj_imag, c=range(p), cmap='hsv', s=35, edgecolors='black', linewidth=0.3, zorder=2)
    
    is_control = (k == 3)
    title_suffix = " (Control - Unlearned)" if is_control else " (Learned Harmonic)"
    ax.set_title(f"Frequency k = {k}{title_suffix}", fontsize=11, fontweight='bold' if not is_control else 'normal')
    ax.set_xlabel("Re(Projection)")
    ax.set_ylabel("Im(Projection)")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.axis('equal')

plt.suptitle(f"Fourier Projections of Word Embeddings $W_E$ (Modulo p={p})", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/fourier_embeddings.png')
plt.close()

# Fourier Frequency Mapping
num_heads = hp['num_heads']
model_dim = hp['embed_dim']
head_dim = model_dim // num_heads

# Get Layer 0 attention parameters
q_proj_weight = model.blocks[0].attention.q_proj.weight.detach() # (num_heads*head_dim, model_dim)
k_proj_weight = model.blocks[0].attention.k_proj.weight.detach() # (num_kv_heads*head_dim, model_dim)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for h in range(num_heads):
    # Slice weights for head h
    W_Q = q_proj_weight[h * head_dim : (h + 1) * head_dim, :]
    W_K = k_proj_weight[h * head_dim : (h + 1) * head_dim, :]
    
    # Compute the QK bilinear matrix over the numbers: W_E @ W_Q^T @ W_K @ W_E^T
    M_QK = (W_E_numbers @ W_Q.T) @ (W_K @ W_E_numbers.T) # shape: (p, p)
    
    # Average diagonals to find attention vs. (a - b) mod p
    d_values = torch.zeros(p, device=device)
    for a in range(p):
        for b in range(p):
            diff = (a - b) % p
            d_values[diff] += M_QK[a, b]
    d_values = d_values / p
    
    # Perform 1D FFT to extract the frequency power spectrum
    fft_vals = torch.fft.fft(d_values)
    # Ignore DC component (index 0) and slice to Nyquist limit (frequencies 1 to p//2)
    power_spectrum = torch.abs(fft_vals)[1 : p//2 + 1].cpu().numpy()
    frequencies = np.arange(1, len(power_spectrum) + 1)
    
    # Finding dominant frequency
    dominant_idx = np.argmax(power_spectrum)
    dominant_freq = frequencies[dominant_idx]
    
    axes[h].bar(frequencies, power_spectrum, color='royalblue', edgecolor='none')
    axes[h].axvline(x=dominant_freq, color='crimson', linestyle='--', label=f"Dominant k={dominant_freq}")
    axes[h].set_title(f"Layer 0 Head {h} Power Spectrum\nDominant: k = {dominant_freq}")
    axes[h].set_xlabel("Fourier Frequency (k)")
    axes[h].set_ylabel("Power")
    axes[h].grid(True, linestyle=':', alpha=0.5)
    axes[h].legend()

plt.suptitle(f"Fourier Frequency Tuning of Layer 0 Attention Heads", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/fourier_frequencies.png')
plt.close()