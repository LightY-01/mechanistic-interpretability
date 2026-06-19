# Mechanistic Interpretability on a Custom 2-Layer Transformer

This repository contains a mechanistic interpretability study on a custom-built, attention-only 2-layer GPT model trained from scratch to perform modular addition (a + b (mod 97)). The goal of the project is to reproduce the delayed generalization phenomenon known as **grokking** and mathematically reverse-engineer the attention circuits the model learns to solve the task.

---

## Repository Structure

The project is structured into `scripts/` (containing source code) and `results/` (containing trained checkpoints and visualization plots) directories:

```
├── scripts/
│   ├── model.py                 # Transformer architecture (Grouped Query Attention, optional MLP)
│   ├── mod_data_gen.py          # Modular arithmetic dataset generator
│   ├── run.py                   # Self-contained training script implementing grokking
│   ├── generate.py              # Sequence generation utilities with ModularTokenizer
│   ├── infer.py                 # Interactive inference CLI for modular addition prompts
│   ├── logit_lens.py            # Logit lens analysis on a single prompt (distribution plotting)
│   └── logit_lens_batched.py    # Logit lens validation analysis over the entire validation split
├── results/
│   ├── gpt_checkpoint.pth       # Trained model checkpoint
│   ├── grokking_curves.png      # Training/validation loss & accuracy curves showing grokking phase
│   ├── logit_lens_single_dist.png # Probability distribution across layers for a single equation
│   └── logit_lens_batched_stats.png # Validation accuracy and confidence plots across layers
├── requirements.txt             # Python package dependencies
└── README.md                    # Project documentation
```

---

## Current Progress

### Step 1: Reproducing Grokking
* **Model Configuration:** A tiny 2-layer (`num_blocks=2`), 4-head (`num_heads=4`, `num_kv_heads=4`), embedding dimension 128 model. The feed-forward MLP layers were disabled (`use_mlp=False`) to keep the residual stream linear for interpretability.
* **Regularization & Training:** Trained using AdamW with high weight decay (`weight_decay=1.0`) on 35\% of the 9,409 possible addition pairs, holding out 65\% for validation.
* **Observation:** The model initially overfits and memorizes the training set (100\% train accuracy, ≈1\% validation accuracy). Over thousands of training steps, the high weight decay forces the memorized weights to collapse, transitioning the model to a generalizing circuit that achieves 100\% accuracy on the unseen validation set.

### Step 2: The Logit Lens
* **Mechanism:** Registered forward hooks on the residual stream at each layer: `layer_0` (embeddings), `layer_1` (Block 0 output), and `layer_2` (Block 1 output). At the `=` token position, intermediate activations are normalized (`model.ln1`) and projected directly to vocabulary space (`model.proj`).
* **Single-Equation Distribution:** Plots show how probability mass shifts from a flat uniform distribution (Layer 0) to periodic/structured frequency bands (Layer 1), before condensing into a sharp spike at the correct sum token (Layer 2).
* **Batched Validation Statistics:** Proved systematically over the 6,116 validation equations that Layer 0 (embeddings) and Layer 1 (Block 0) hold near 0\% prediction accuracy, demonstrating that the final modular sum is realized and mapped to the residual stream at Layer 2.

---

## Future Steps

### Step 3: Activation Patching
Pinpoint the specific attention heads responsible for transferring, processing, and outputting the modular sum. We will run clean and corrupted forward passes, patch activations head-by-head, and measure the logit difference to isolate the circuit's minimal nodes.

### Step 4: Circuit Diagram & DFT Analysis
Visualize the representation space. We will perform dimensionality reduction (PCA) on the input embeddings (W<sub>E</sub>) and unembeddings (W<sub>U</sub>) to reveal the circular/helical representations representing the Discrete Fourier Transform (DFT) rotation group, allowing us to map out a complete circuit diagram of information flow.

---

## Getting Started

### Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### Running the Code
1. **Train the Model:**
   ```bash
   python scripts/run.py
   ```
2. **Logit Lens (Single Prompt):**
   ```bash
   python scripts/logit_lens.py
   ```
3. **Logit Lens (Batched Validation):**
   ```bash
   python scripts/logit_lens_batched.py
   ```
4. **Interactive Inference:**
   ```bash
   python scripts/infer.py
   ```
