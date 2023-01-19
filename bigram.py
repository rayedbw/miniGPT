from pathlib import Path
import requests

import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparams
BATCH_SIZE = 256
CONTEXT_LENGTH = 16
MAX_ITER = int(1e6)
EVAL_INTERVAL = int(1e5)
EVAL_ITER = int(1e4)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# For reproducibility
torch.manual_seed(1337)

# Download dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file_path = Path("./shakespeare.txt")
if not file_path.exists():
    print("Downloading...")
    with requests.get(url, stream=True) as r:
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
    print(f"Saving to {file_path}")
else:
    print("Dataset exists")

# Load dataset
with open(file_path) as f:
    text = f.read()

# Create vocabulary
chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)

# Integer to character mapping and vice-versa
itos = {i: c for i, c in enumerate(chars)}
ctoi = {c: i for i, c in itos.items()}

# Encode decode sentence
encode = lambda s: [ctoi[c] for c in s]
decode = lambda a: "".join([itos[i] for i in a])

# Train, test split
data = torch.tensor(encode(text))
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Get minibatch of training samples
def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - CONTEXT_LENGTH, (BATCH_SIZE,))
    X = torch.stack([data[i : i + CONTEXT_LENGTH] for i in idx])
    y = torch.stack([data[i + 1 : i + 1 + CONTEXT_LENGTH] for i in idx])
    X, y = X.to(DEVICE), y.to(DEVICE)
    return X, y


# Measure training performance
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITER)
        for iter in range(EVAL_ITER):
            X, y = get_batch(split)
            _, loss = model(X, y)
            losses[iter] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Define bigram model
class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, VOCAB_SIZE)

    def forward(self, input, target=None):
        loss = None
        logit = self.embed(input)

        if target is None:
            return logit, loss

        B, T, C = logit.shape
        logit = logit.view(B * T, C)
        target = target.view(B * T)
        loss = F.cross_entropy(logit, target)

        return logit, loss

    def generate(self, input, max_tokens=100):
        for _ in range(max_tokens):
            logit, _ = self(input)
            logit = logit[:, -1, :]
            prob = F.softmax(logit, dim=1)
            pred = torch.multinomial(prob, num_samples=1)
            input = torch.concatenate((input, pred), dim=1)
        return input


# Instantiate model
model = BigramLM()
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters())

# Train
for iter in range(MAX_ITER):
    X, y = get_batch("train")
    logit, loss = model(X, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

# Save model
torch.save(model.state_dict(), f"./models/bigram-{MAX_ITER}.pt")

# Generate shakespeare
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(model.generate(context, max_tokens=500).squeeze().tolist()))
