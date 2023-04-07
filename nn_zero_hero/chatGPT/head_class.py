class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B, T, C)

        # compute attendion scores (affinities between nodes)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) --> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList((Head(head_size) for _ in range(num_heads)))

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

class FeedForward(nn.Module):
    """linear layer followed by non-linearity"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
