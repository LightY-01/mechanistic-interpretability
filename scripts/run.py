import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mod_data_gen import ModularArithmeticDataset
from model import GPT

# Hyperparameters
p = 97
vocab_size = p + 1  # 0 to p-1, plus '=' token (represented as p)
context_length = 3
embed_dim = 128
num_heads = 4
num_kv_heads = 4
num_blocks = 2
learning_rate = 1e-3
weight_decay = 1.0
max_iters = 15000
batch_size = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = 'gpt_checkpoint.pth'

print(f"Training modular addition model on {device}")

# Load dataset
dataset = ModularArithmeticDataset(p=p)
data = dataset.data  # shape (9409, 4)

# Split dataset: 35% train, 65% validation
torch.manual_seed(42)
train_size = int(0.35 * len(dataset))
val_size = len(dataset) - train_size

shuffled_indices = torch.randperm(len(data))
train_data = data[shuffled_indices[:train_size]]
val_data = data[shuffled_indices[train_size:]]

print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

# Model
model = GPT(
    vocab_size=vocab_size,
    model_dim=embed_dim,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    context_length=context_length,
    num_blocks=num_blocks,
    use_mlp=False
).to(device)

# Optimizer with high weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.98))

# Track statistics
steps = []
train_losses = []
val_losses = []
train_accs = []
val_accs = []

def evaluate(data_split):
    model.eval()
    with torch.no_grad():
        x = data_split[:, :-1].to(device)
        y = data_split[:, 1:].to(device)
        logits = model(x)
        # Loss and accuracy only on the last prediction (predicting c)
        loss = F.cross_entropy(logits[:, -1, :], y[:, -1]).item()
        preds = torch.argmax(logits[:, -1, :], dim=-1)
        acc = (preds == y[:, -1]).float().mean().item()
    model.train()
    return loss, acc

for step in range(max_iters):
    # Sample training batch
    indices = torch.randint(0, len(train_data), (batch_size,))
    batch = train_data[indices]
    x = batch[:, :-1].to(device)
    y = batch[:, 1:].to(device)
    
    # Forward pass
    logits = model(x)
    loss = F.cross_entropy(logits[:, -1, :], y[:, -1])
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    # Evaluate periodically
    if step % 100 == 0 or step == max_iters - 1:
        train_loss, train_acc = evaluate(train_data)
        val_loss, val_acc = evaluate(val_data)
        
        steps.append(step)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Step {step:5d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:6.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:6.2f}%")
        
    # Save checkpoint and plot curves only every 500 steps or at the end
    if step % 500 == 0 or step == max_iters - 1:
        torch.save({
            'model_state_dict': model.state_dict(),
            'hyperparams': {
                'vocab_size': vocab_size,
                'embed_dim': embed_dim,
                'num_heads': num_heads,
                'num_kv_heads': num_kv_heads,
                'context_length': context_length,
                'num_blocks': num_blocks,
                'use_mlp': False,
                'p': p
            }
        }, CHECKPOINT)
        
        # Save curves plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(steps, train_losses, label='Train')
        ax1.plot(steps, val_losses, label='Val')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_yscale('log')
        ax1.set_title('Cross Entropy Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(steps, train_accs, label='Train')
        ax2.plot(steps, val_accs, label='Val')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.suptitle('Grokking Modular Addition (Attention-Only 2-Layer Transformer)')
        plt.savefig('grokking_curves.png')
        plt.close()
        
print("Training complete! Checkpoint saved and curves plotted.")
