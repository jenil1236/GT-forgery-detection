from __future__ import annotations
import torch
import torch.nn as nn


class QFormerLayer(nn.Module):
    """
    Single Q-Former layer:
      1. Self-attention over queries
      2. Cross-attention: queries attend to token pool
      3. FFN
    All with pre-norm (LayerNorm before attention).
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 ffn_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.self_attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,   # (B, n_q, d_model)
        pool:    torch.Tensor,   # (B, n_tokens, d_model)
    ):
        # 1. Self-attention over queries
        q_norm = self.norm1(queries)
        sa_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + self.dropout(sa_out)
        # 2. Cross-attention: queries -> pool
        q_norm = self.norm2(queries)
        ca_out, attn_w = self.cross_attn(q_norm, pool, pool)
        queries = queries + self.dropout(ca_out)
        # 3. FFN
        q_norm = self.norm3(queries)
        queries = queries + self.ffn(q_norm)
        return queries, attn_w  # attn_w: (B, n_q, n_tokens)


class QFormer(nn.Module):
    """
    Q-Former: n_layers of QFormerLayer.
    Compresses token pool (B, 294, 256) into (B, n_queries, 256).
    """

    def __init__(self, n_queries: int = 16, d_model: int = 256,
                 n_layers: int = 2, n_heads: int = 8,
                 ffn_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.n_queries = n_queries
        self.d_model   = d_model
        # Learned query embeddings
        self.query_emb = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.layers = nn.ModuleList([
            QFormerLayer(d_model=d_model, n_heads=n_heads,
                         ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, pool: torch.Tensor):
        """
        Args:
            pool: (B, 294, 256) unified token pool
        Returns:
            queries: (B, n_queries, d_model)
            last_attn: (B, n_queries, 294) cross-attention weights from last layer
        """
        B = pool.size(0)
        queries = self.query_emb.expand(B, -1, -1)  # (B, n_queries, d_model)
        last_attn = None
        for layer in self.layers:
            queries, last_attn = layer(queries, pool)
        queries = self.norm_out(queries)
        return queries, last_attn
