from pathlib import Path
import requests

import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparams
BATCH_SIZE = 64
SEQ_LEN = 256
MAX_ITER = 5000
EVAL_INTERVAL = 500
EVAL_ITER = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 512
NUM_HEADS = 8
NUM_BLOCKS = 6
FEED_FORWARD_MULTIPLIER = 4

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
    # print("Dataset exists")
    pass

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
    idx = torch.randint(len(data) - SEQ_LEN, (BATCH_SIZE,))
    X = torch.stack([data[i : i + SEQ_LEN] for i in idx])
    y = torch.stack([data[i + 1 : i + 1 + SEQ_LEN] for i in idx])
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


class MultiHeadAttention(nn.Module):
    """Multi headed self attention"""

    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert (
            self.embed_dim % self.num_heads == 0
        ), "embed_dim must be divisible by num_heads"
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout()
        self.register_buffer(
            "tril",
            torch.tril(torch.ones((SEQ_LEN, SEQ_LEN))).view(1, 1, SEQ_LEN, SEQ_LEN),
        )

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(self.embed_dim, dim=2)
        q = q.view(B, T, self.num_heads, self.embed_dim // self.num_heads).transpose(
            1, 2
        )  # B, H, T, Eh
        k = k.view(B, T, self.num_heads, self.embed_dim // self.num_heads).transpose(
            1, 2
        )  # B, H, T, Eh
        v = v.view(B, T, self.num_heads, self.embed_dim // self.num_heads).transpose(
            1, 2
        )  # B, H, T, Eh

        scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # B, H, T, T
        scores_masked = scores.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
        weights = self.dropout(F.softmax(scores_masked, dim=-1))  # B, H, T, T

        y = weights @ v  # B, H, T, Eh
        y = y.transpose(1, 2).contiguous().view(B, T, -1)  # B, T, E
        y = self.dropout(self.linear(y))

        return y


class FeedForwardNetwork(nn.Module):
    """Single layer MLP"""

    def __init__(self, in_features) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_features, in_features * FEED_FORWARD_MULTIPLIER),
            nn.ReLU(),
            nn.Linear(in_features * FEED_FORWARD_MULTIPLIER, in_features),
            nn.Dropout(),
        )

    def forward(self, x):
        return self.ffn(x)


class Block(nn.Module):
    """Transformer block"""

    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForwardNetwork(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.mha(self.ln1(x)) + x
        x = self.ffn(self.ln2(x)) + x
        return x


# Define bigram model
class TransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_embed = nn.Embedding(SEQ_LEN, EMBED_DIM)
        self.blocks = nn.Sequential(
            *[Block(EMBED_DIM, NUM_HEADS) for _ in range(NUM_BLOCKS)],
        )
        self.lnorm = nn.LayerNorm(EMBED_DIM)
        self.linear = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, input, target=None):
        loss = None
        B, T = input.shape

        token_emb = self.token_embed(input)  # B, T -> B, T, E
        pos_emb = self.pos_embed(torch.arange(T, device=DEVICE))  # T, E
        x = token_emb + pos_emb  # B, T, E
        x = self.blocks(x)
        x = self.lnorm(x)
        logit = self.linear(x)  # B, T, V

        if target is None:
            return logit, loss

        B, T, V = logit.shape
        logit = logit.view(B * T, V)
        target = target.view(B * T)
        loss = F.cross_entropy(logit, target)

        return logit, loss

    @torch.no_grad()
    def generate(self, input, max_tokens=1000):
        self.eval()
        for _ in range(max_tokens):
            input_cond = input[:, -SEQ_LEN:]
            logit, _ = self(input_cond)
            logit = logit[:, -1, :]
            prob = F.softmax(logit, dim=1)
            pred = torch.multinomial(prob, num_samples=1)
            input = torch.concatenate((input, pred), dim=1)
        self.train()
        return input


# Instantiate model
model = TransformerDecoder()
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters())

num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameter count: {num_parameters}")
print("Beginning training...")

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
model_path = (
    f"./models/bigram-{MAX_ITER}-mha-block-residual-lnorm-dropout-paper_params.pt"
)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# model = BigramLM()
# model.load_state_dict(torch.load("./models/bigram-100000-mha-with-ff.pt"))
# model.to(DEVICE)

# Generate shakespeare
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(model.generate(context).squeeze().tolist()))
