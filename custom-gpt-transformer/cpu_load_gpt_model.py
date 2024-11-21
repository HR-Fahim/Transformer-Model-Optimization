import os
import argparse
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F

# Force the model to use CPU
device = 'cpu'

# Argument parser
parser = argparse.ArgumentParser(description='Infinite in Modern Thought')
parser.add_argument('-batch_size', type=int, default=32, help='Please enter the batch size')
args = parser.parse_args()
batch_size = args.batch_size

# Hyperparameters
block_size = 64
max_iters = 100
learning_rate = 3e-3
eval_iters = 100
n_embd = 256
n_head = 8
n_layer = 4
dropout = 0.2

# Load vocabulary
vocab_file = 'C:/Users/Asus/Desktop/Research/Project/custom-gpt-transformer/vocab.txt'
if not os.path.exists(vocab_file):
    raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")

with open(vocab_file, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

# Create encoding and decoding mappings
string_to_index = {ch: i for i, ch in enumerate(chars)}
index_to_string = {i: ch for i, ch in enumerate(chars)}

endcode = lambda s: [string_to_index[c] for c in s]
decode = lambda l: ''.join([index_to_string[i] for i in l])

# Model components
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head)
        self.ffw = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x.transpose(0, 1)
        y, _ = self.sa(x, x, x)
        x = x + y
        x = self.ln1(x)
        x = x.transpose(0, 1)
        y = self.ffw(x)
        x = x + y
        x = self.ln2(x)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.lm_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(0, T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.lm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # Ensure slicing respects block size
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]  # Focus on the last token
            probs = F.softmax(logits, dim=-1)

            # Clamp probabilities to avoid invalid indices
            probs = probs[:, :vocab_size]

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# Load model
model_file = 'C:/Users/Asus/Desktop/Research/Project/custom-gpt-transformer/models/model-01.pkl'

# Load model
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Override vocab_size if mismatched
if model.lm_head.out_features != vocab_size:
    print(f"Updating model vocab size from {model.lm_head.out_features} to {vocab_size}")
    model.lm_head = nn.Linear(n_embd, vocab_size)
    model.token_embedding_table = nn.Embedding(vocab_size, n_embd)

model = model.to(device)

# Interactive prompt
while True:
    try:
        prompt = input("Model prompt >>> ").strip()
        if not prompt:
            print("Empty prompt. Please enter some text.")
            continue

        # Check if prompt contains invalid characters
        if any(c not in string_to_index for c in prompt):
            print("Error: Prompt contains characters not in the vocabulary.")
            continue

        context = torch.tensor(endcode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        print(f"Context shape: {context.shape}, Context: {context}")

        generated_chars = decode(model.generate(context, max_new_tokens=100)[0].tolist())
        print(generated_chars)

    except Exception as e:
        print(f"Error: {e}")
