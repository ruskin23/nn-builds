import torch
import torch.nn as nn



class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention mechanism for transformer models.

    Args:
        d_in (int): Input feature dimension.
        d_out (int): Output feature dimension (typically same as d_in).
        context_length (int): Maximum sequence length for causal masking.
        dropout (float): Dropout probability for attention weights.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): If True, include bias in Q, K, V projections.

    Input:
        x (Tensor): Input tensor of shape (batch_size, seq_len, d_in)

    Output:
        Tensor: Output tensor of shape (batch_size, seq_len, d_out)
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        # Linear projections for queries, keys, and values
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Final output projection
        self.out_proj = nn.Linear(d_out, d_out)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Causal mask to prevent attention to future tokens
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (b, seq_len, d_in)

        Returns:
            Tensor of shape (b, seq_len, d_out)
        """
        b, seq_len, _ = x.shape

        # Linear projections
        queries = self.W_query(x)      # (b, seq_len, d_out)
        keys    = self.W_key(x)        # (b, seq_len, d_out)
        values  = self.W_value(x)      # (b, seq_len, d_out)

        # Reshape for multi-head: split d_out into (num_heads, head_dim)
        queries = queries.view(b, seq_len, self.num_heads, self.head_dim)  # (b, seq_len, n_heads, head_dim)
        keys    = keys.view(b, seq_len, self.num_heads, self.head_dim)     # (b, seq_len, n_heads, head_dim)
        values  = values.view(b, seq_len, self.num_heads, self.head_dim)   # (b, seq_len, n_heads, head_dim)

        # Transpose: move num_heads forward for multi-head attention
        queries = queries.transpose(1, 2)  # (b, n_heads, seq_len, head_dim)
        keys    = keys.transpose(1, 2)    # (b, n_heads, seq_len, head_dim)
        values  = values.transpose(1, 2)  # (b, n_heads, seq_len, head_dim)

        # Scaled dot-product attention
        attn_scores = queries @ keys.transpose(-2, -1)  # (b, n_heads, seq_len, seq_len)
        attn_scores /= self.head_dim ** 0.5

        # Apply causal mask (upper triangle)
        mask = self.mask[:seq_len, :seq_len].bool()     # (seq_len, seq_len)
        attn_scores.masked_fill_(mask, float('-inf'))   # (b, n_heads, seq_len, seq_len)

        # Softmax and dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)   # (b, n_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)           # (b, n_heads, seq_len, seq_len)

        # Weighted sum of values
        context = attn_weights @ values     # (b, n_heads, seq_len, head_dim)

        # Concatenate heads
        context = context.transpose(1, 2)   # (b, seq_len, n_heads, head_dim)
        context = context.contiguous().view(b, seq_len, self.d_out)  # (b, seq_len, d_out)

        # Final projection
        output = self.out_proj(context)    # (b, seq_len, d_out)

        return output

class LayerNorm(nn.Module):
    """
    Layer Normalization applied over the last dimension of the input.

    Args:
        emb_dim (int): Embedding dimension (last dimension of input)

    Input:
        x: Tensor of shape (batch_size, ..., emb_dim)

    Output:
        Tensor of shape (batch_size, ..., emb_dim)
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # x: (..., emb_dim)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Approximate GELU activation as described in the original Transformer paper.

    Input:
        x: Tensor of any shape

    Output:
        Tensor of same shape as input
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * x.pow(3))
        ))


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network used in Transformer blocks.

    Args:
        config (dict): Contains:
            - "emb_dim": embedding dimension

    Architecture:
        Linear (emb_dim → 4×emb_dim) → GELU → Linear (4×emb_dim → emb_dim)

    Input:
        x: Tensor of shape (batch_size, seq_len, emb_dim)

    Output:
        Tensor of shape (batch_size, seq_len, emb_dim)
    """
    def __init__(self, config):
        super().__init__()
        emb_dim = config["emb_dim"]
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),   # (b, seq, 4*emb)
            GELU(),                             # (b, seq, 4*emb)
            nn.Linear(4 * emb_dim, emb_dim),    # (b, seq, emb)
        )

    def forward(self, x):
        # x: (b, seq_len, emb_dim)
        return self.layers(x)  # (b, seq_len, emb_dim)


class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of:
    - Multi-Head Self-Attention
    - Layer Normalization
    - Feed Forward Network
    - Residual Connections
    - Dropout

    Args:
        config (dict): Configuration dictionary with keys:
            - emb_dim (int): Embedding dimension
            - contex_length (int): Maximum sequence length
            - drop_rate (float): Dropout rate
            - n_heads (int): Number of attention heads
            - qkv_bias (bool): Whether to use bias in QKV projections

    Input:
        x: Tensor of shape (batch_size, seq_len, emb_dim)

    Output:
        Tensor of shape (batch_size, seq_len, emb_dim)
    """
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadedAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["contex_length"],
            dropout=config["drop_rate"],
            num_heads=config["n_heads"],
            qkv_bias=config["qkv_bias"]
        )
        self.ff = FeedForward(config=config)
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        # x: (b, seq_len, emb_dim)

        # ---- Multi-Head Attention Block ----
        shortcut = x                                      # (b, seq, emb)
        x = self.norm1(x)                                 # (b, seq, emb)
        x = self.att(x)                                   # (b, seq, emb)
        x = self.dropout(x)                               # (b, seq, emb)
        x = x + shortcut                                  # Residual connection

        # ---- Feed Forward Block ----
        shortcut = x                                      # (b, seq, emb)
        x = self.norm2(x)                                 # (b, seq, emb)
        x = self.ff(x)                                    # (b, seq, emb)
        x = self.dropout(x)                               # (b, seq, emb)
        x = x + shortcut                                  # Residual connection

        return x                                          # (b, seq, emb)
    
    
class GPTModel(nn.Module):
    """
    GPT-style Transformer language model.

    Components:
        - Token embedding
        - Positional embedding
        - Dropout after embedding
        - Stack of Transformer blocks
        - Final LayerNorm
        - Output linear layer projecting to vocab logits

    Args:
        config (dict): A dictionary with:
            - vocab_size (int): Size of vocabulary
            - emb_dim (int): Embedding dimension
            - context_length (int): Max sequence length
            - drop_rate (float): Dropout probability
            - n_layers (int): Number of transformer blocks
            - n_heads (int): Number of attention heads
            - qkv_bias (bool): Whether to use bias in QKV projections

    Input:
        in_idx: Tensor of shape (batch_size, seq_len), token indices

    Output:
        logits: Tensor of shape (batch_size, seq_len, vocab_size)
    """
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )

        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        Forward pass.

        Args:
            in_idx (Tensor): (batch_size, seq_len), token indices

        Returns:
            logits (Tensor): (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = in_idx.shape

        # Token embeddings: (b, seq_len, emb_dim)
        token_embeds = self.tok_emb(in_idx)

        # Positional embeddings: (seq_len, emb_dim) → (1, seq_len, emb_dim)
        pos_ids = torch.arange(seq_len, device=in_idx.device).unsqueeze(0)
        pos_embeds = self.pos_emb(pos_ids)  # (1, seq_len, emb_dim)

        # Add & dropout embeddings
        x = token_embeds + pos_embeds       # (b, seq_len, emb_dim)
        x = self.drop_emb(x)                # (b, seq_len, emb_dim)

        # Transformer blocks
        x = self.trf_blocks(x)              # (b, seq_len, emb_dim)

        # Final layer norm
        x = self.final_norm(x)              # (b, seq_len, emb_dim)

        # Project to vocab logits
        logits = self.out_head(x)           # (b, seq_len, vocab_size)

        return logits
