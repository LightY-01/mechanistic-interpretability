import torch
from model import GPT
from generate import generate_text, ModularTokenizer

checkpoint_path = 'gpt_checkpoint.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Loading checkpoint: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location=device)
hp = ckpt['hyperparams']

tokenizer = ModularTokenizer(p=hp.get('p', 97))

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

print("Modular Addition Inference (e.g. '12 45 ='). Type 'quit' to exit.")
while True:
    prompt = input("\nPrompt > ").strip()
    if prompt.lower() in ('quit', 'exit', 'q'):
        break
    if not prompt:
        continue
    if not prompt.endswith('='):
        prompt += " ="
        
    output = generate_text(
        model          = model,
        prompt         = prompt,
        tokenizer      = tokenizer,
        max_new_tokens = 1,
        context_length = hp['context_length'],
        temperature    = 0.0,
        device         = device,
    )
    print(f"Result: {output}")
