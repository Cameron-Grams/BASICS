import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32

#-----------------------

#  !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# create the blocks for encoding
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# create the character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

#encoder and decoder for tokens
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encodes an input string into a list of integers
decode = lambda l: ''.join(itos[i] for i in l) # decodes an input list into a string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # first 90% for training
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    #generalize a small batch of data into inputs x and y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i+1: i + block_size + 1] for i in ix])
    x, v = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C); batch, time, channel tensor
        # time is the block size (8) and channel is the vocab size (65)
        logits = self.lm_head(tok_emb)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # function expects (B, T)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        # logits = score for the next character in the sequence 
        return logits, loss



    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context

        for _ in range(max_new_tokens):

            # get predictions
            logits, loss = self(idx)

            # focus only on the last time step
            logits = logits[:, -1,:] # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # (B, C)

            # sample from the distribution 
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

#------------------------------------------------
model = BigramLanguageModel()

m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # evaluate the loss at given iterations
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
