import torch
import torch.nn as nn

# Config class to hold all the hyperparameters
class Config:
    num_layers = 2
    num_heads = 4
    max_len = 5
    hidden_dim = 128
    dropout = 0.1
    num_updates = int(1e5)
    mod = 97
    split_size = 0.4

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_head, p_dropout):
        super(TransformerBlock, self).__init__()
        
        #TODO: Implement MultiHead Attention Yourself.
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.multiheadattn = nn.MultiheadAttention(hidden_dim, num_head, p_dropout)

        # FeedFoward Network.
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, x):

        seq_len = len(x)

        # define the causal attn mask.
        causal_mask = torch.triu(torch.full((seq_len, seq_len), True, device=x.device), diagonal=1)
        
        # this is the residual stream
        residual = x

        # Compute the attention and add it back to the residual stream.
        x = self.ln1(x)
        attn_out, _ = self.multiheadattn(x, x, x, attn_mask = causal_mask )
        residual = residual + attn_out

        # pass through the FFN
        x = self.ln2(residual)
        residual = residual + self.ffn(x)

        return residual

# GPT Style Decoder Only Transformer
class Transformer(nn.Module):    
    def __init__(self, cfg: Config):
        super(Transformer, self).__init__()

        # Define the hyperparameters for the model
        self.num_layers = cfg.num_layers
        self.num_heads = cfg.num_heads
        self.seq_length = cfg.max_len
        self.hidden_dim = cfg.hidden_dim
        self.p_dropout = cfg.dropout
        self.mod = cfg.mod

        # num tokens = mod + 2 (op, =)
        vocab_size = self.mod + 2

        # we need to create the embedding vectors (both position and token)
        self.token_emb = nn.Embedding(vocab_size, self.hidden_dim)
        self.position_emb = nn.Embedding(self.seq_length, self.hidden_dim)

        # dropout layer for regularization
        self.dropout = nn.Dropout(self.p_dropout)

        # define the transformer blocks.
        layers = [ TransformerBlock(self.hidden_dim, self.num_heads, self.p_dropout) for _ in range(self.num_layers) ]
        self.decoderblock = nn.Sequential(*layers)

        # Get the softmax scores
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        token_embeddings = self.token_emb(x)
        positions = torch.arange(seq_len, device=x.device).repeat(batch_size, 1)
        postn_embeddings = self.position_emb(positions)

        # This is the input to the transformer blocks.
        x = self.dropout(token_embeddings + postn_embeddings)
        
        # x has a shape of [512, 4, 128] -> [4, 512, 128]
        x = torch.permute(x, (1, 0, 2))
        x = self.decoderblock(x)

        x = self.ln1(x)
        x = self.fc1(x)

        return x