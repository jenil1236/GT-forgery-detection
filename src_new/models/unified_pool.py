from __future__ import annotations
import math
import torch
import torch.nn as nn


def _sinusoidal_pe_2d(G: int, d_model: int) -> torch.Tensor:
    """
    2D sinusoidal positional encoding.
    Returns (G*G, d_model) tensor, not learned.
    """
    half = d_model // 2
    pe_1d = torch.zeros(G, half)
    pos = torch.arange(G).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, half, 2).float() * (-math.log(10000.0) / half))
    pe_1d[:, 0::2] = torch.sin(pos * div)
    pe_1d[:, 1::2] = torch.cos(pos * div[:half // 2])
    # Combine row and col PE
    pe_row = pe_1d.unsqueeze(1).expand(G, G, half)  # (G, G, half)
    pe_col = pe_1d.unsqueeze(0).expand(G, G, half)  # (G, G, half)
    pe_2d = torch.cat([pe_row, pe_col], dim=-1)      # (G, G, d_model)
    return pe_2d.reshape(G * G, d_model)              # (G*G, d_model)


class UnifiedTokenPool(nn.Module):
    """
    Takes 6 aligned streams (each B x 49 x 256) and builds a single
    token pool (B, 294, 256) with positional, modality, and type embeddings.

    Stream order: visual_ref, visual_test, text_ref, text_test, signal_ref, signal_test
    """

    STREAM_ORDER = [
        ("visual_ref",  0, 0),
        ("visual_test", 0, 1),
        ("text_ref",    1, 0),
        ("text_test",   1, 1),
        ("signal_ref",  2, 0),
        ("signal_test", 2, 1),
    ]  # (key, modality_idx, type_idx)  modality: 0=visual,1=text,2=signal  type: 0=ref,1=test

    def __init__(self, d_model: int = 256, G: int = 7):
        super().__init__()
        self.d_model = d_model
        self.G = G
        self.n_tokens = G * G  # 49
        # Fixed 2D sinusoidal PE (not learned)
        pe = _sinusoidal_pe_2d(G, d_model)   # (49, 256)
        self.register_buffer("pe", pe)
        # Learned modality embeddings: visual=0, text=1, signal=2
        self.modality_emb = nn.Embedding(3, d_model)
        # Learned type embeddings: ref=0, test=1
        self.type_emb = nn.Embedding(2, d_model)

    def forward(self, aligned: dict) -> torch.Tensor:
        """
        Args:
            aligned: dict from SpatialAlignLayer, each value (B, 49, 256)
        Returns:
            pool: (B, 294, 256)
        """
        device = next(iter(aligned.values())).device
        streams = []
        for key, mod_idx, type_idx in self.STREAM_ORDER:
            tokens = aligned[key]  # (B, 49, 256)
            # Add positional encoding (broadcast across batch)
            tokens = tokens + self.pe.unsqueeze(0)  # (B, 49, 256)
            # Add modality embedding
            mod_e = self.modality_emb(torch.tensor(mod_idx, device=device))
            tokens = tokens + mod_e.unsqueeze(0).unsqueeze(0)  # (B, 49, 256)
            # Add type embedding
            type_e = self.type_emb(torch.tensor(type_idx, device=device))
            tokens = tokens + type_e.unsqueeze(0).unsqueeze(0)  # (B, 49, 256)
            streams.append(tokens)
        pool = torch.cat(streams, dim=1)  # (B, 294, 256)
        return pool
