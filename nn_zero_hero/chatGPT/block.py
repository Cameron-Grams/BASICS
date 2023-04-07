class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimension, n_head: the number of heads
        super.__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x


