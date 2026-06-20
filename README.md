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
│   ├── logit_lens_batched.py    # Logit lens validation analysis over the entire validation split
│   ├── activation_patching.py   # Demo script for patching individual attention head activations
│   └── patching_batched.py      # Batch patching evaluations over 100 random validation prompts
├── results/
│   ├── gpt_checkpoint.pth       # Trained model checkpoint
│   ├── grokking_curves.png      # Training/validation loss & accuracy curves showing grokking phase
│   ├── logit_lens_dist.png      # Probability distribution across layers for a single equation
│   ├── logit_lens_batched.png   # Validation accuracy and confidence plots across layers
│   ├── patching_heatmap_a.png   # Logit difference shift heatmap for Operand A changes (batched)
│   └── patching_heatmap_b.png   # Logit difference shift heatmap for Operand B changes (batched)
├── requirements.txt             # Python package dependencies
└── README.md                    # Project documentation
```

---

## Current Progress

### Step 1: Reproducing Grokking
* **Model Configuration:** A tiny 2-layer (`num_blocks=2`), 4-head (`num_heads=4`, `num_kv_heads=4`), embedding dimension 128 model. The feed-forward MLP layers were disabled (`use_mlp=False`) to keep the residual stream linear for interpretability.
* **Regularization & Training:** Trained using AdamW with high weight decay (`weight_decay=1.0`) on 35% of the 9,409 possible addition pairs, holding out 65% for validation.
* **Observation:** The model overfitted the training set initially (100% train accuracy, ≈1% validation accuracy). Under continued weight decay constraint, the model transitioned to a generalizing circuit, grokking the task to reach 100% validation accuracy.

### Step 2: The Logit Lens
* **Mechanism:** Registered forward hooks on the residual stream at each layer: `layer_0` (embeddings), `layer_1` (Block 0 output), and `layer_2` (Block 1 output). Projected intermediate activations directly into vocabulary space.
* **Findings:** Proved systematically over the 6,116 validation equations that Layer 0 (embeddings) and Layer 1 (Block 0) hold near 0% prediction accuracy, demonstrating that the final modular sum is realized and mapped to the residual stream at Layer 2 (Block 1 output).

### Step 3: Activation Patching
* **Mechanism:** Intercepted the forward pass of corrupted prompts (e.g. changing operand a or b) and patched in the attention head outputs (`z` activations) from clean runs to measure the recovery in logit difference.
* **Specialization Findings (Single Prompt):**
  * Swapping operand b (`12 45 =` vs. `12 23 =`) identified `L0H3` and `L0H1` as the key movers of b. Patching `[(0, 1), (0, 3), (1, 1)]` restored target prediction to **92.11%**.
  * Swapping operand a (`12 45 =` vs. `23 45 =`) identified a different circuit `[(0, 1), (0, 2), (1, 0), (1, 1)]` (**94.90%** recovery), proving position specialization.
  * Changing b to a different delta (`12 45 =` vs. `12 33 =`) required patching all Layer 0 heads (**96.20%**), showing that circuits are Fourier-frequency dependent.
* **Global Circuit Findings (Batched 100 Prompts):**
  * **Layer 0 is globally distributed:** To recover operands consistently across all prompts, all 4 heads in Layer 0 must be patched, acting in parallel to route distinct Fourier components.
  * **Layer 1 Computes Positional Harmonics:** **`L1H1`** is the global compute center (averaging ≈+8.55 shift). **`L1H0`** specializes in computing a-corruptions, while **`L1H3`** specializes in computing b-corruptions. Patching `All L0 Heads + L1H3, L1H1, L1H0` provides **82%-85%** global mathematical recovery.

---

## Future Steps

### Step 4: Circuit Diagram & DFT Analysis
1. **Representational PCA Circles:** Extract the embedding weight matrix W<sub>E</sub> and project the row vectors (0 to 96) to 2D/3D spaces using PCA to visually demonstrate that the model maps modular numbers onto a geometric circle (phase rotations).
2. **Fourier Frequency Mapping:** Map the attention head query/key projections to identify the specific trigonometric frequencies **ω<sub>k</sub> = 2πk / p** each head is tuned to, showing how the model performs addition via trig identities:  
**cos(ω<sub>k</sub>(a + b)) = cos(ω<sub>k</sub>a)cos(ω<sub>k</sub>b) - sin(ω<sub>k</sub>a)sin(ω<sub>k</sub>b)**
3. **Circuit Flow Diagram:** Draw a concrete flow diagram illustrating how token inputs enter, decompose into Fourier components, compute rotations in Layer 1, and project back to logits.

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
2. **Logit Lens (Single & Batched):**
   ```bash
   python scripts/logit_lens.py
   python scripts/logit_lens_batched.py
   ```
3. **Activation Patching Sweep:**
   ```bash
   python scripts/activation_patching.py
   python scripts/patching_batched.py
   ```
