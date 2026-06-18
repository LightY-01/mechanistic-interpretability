import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

class ModularArithmeticDataset(Dataset):
    def __init__(self, p: int = 97):
        """
        Generates a dataset for the equation: (a + b) mod p = c
        The vocabulary consists of numbers 0 to p-1, plus an '=' token.
        """
        self.p = p
        self.vocab_size = p + 1  # Numbers 0 to p-1, plus the '=' symbol
        self.eq_token = p        # The '=' token is represented by the integer `p`
        
        # Generate every possible combination of (a, b)
        data = []
        for a in range(p):
            for b in range(p):
                c = (a + b) % p
                # Sequence format: [a, b, =, c]
                data.append([a, b, self.eq_token, c])
                
        self.data = torch.tensor(data, dtype=torch.long)
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For autoregressive training, X is the sequence without the last token,
        and Y is the sequence shifted by one.
        X: [a, b, =]
        Y: [b, =, c]
        """
        row = self.data[idx]
        x = row[:-1]
        y = row[1:]
        return x, y

if __name__ == "__main__":
    dataset = ModularArithmeticDataset(p=97)
    print(f"Dataset size: {len(dataset)} equations")
    print(f"Vocabulary size: {dataset.vocab_size} tokens")
    
    x, y = dataset[0]
    print(f"Sample X (Input): {x.tolist()} -> meaning: (0 + 0 = )")
    print(f"Sample Y (Target): {y.tolist()} -> meaning: (0 = 0)")