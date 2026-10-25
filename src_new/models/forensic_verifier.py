from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

from src_new.models.feature_extractor import FeatureExtractor
from src_new.models.align             import SpatialAlignLayer
from src_new.models.unified_pool      import UnifiedTokenPool
from src_new.models.qformer           import QFormer
from src_new.models.belief_head       import InterpretableBeliefHead


class ForensicVerifier(nn.Module):
    """
    Full Q-Former forensic verifier.

    Frozen:    FeatureExtractor (Stage-1 encoders)
    Trainable: SpatialAlignLayer + UnifiedTokenPool + QFormer + BeliefHead

    Forward input:  batch dict with ref_visual, ref_text, ref_signal,
                                      test_visual, test_text, test_signal
    Forward output: dict with belief, per_query_logits, query_weights,
                    final_logit, last_attn
    """

    def __init__(
        self,
        visual_enc, text_enc, signal_enc,
        d_model: int    = 256,
        G: int          = 7,
        n_queries: int  = 16,
        n_layers: int   = 2,
        n_heads: int    = 8,
        ffn_dim: int    = 1024,
        dropout: float  = 0.1,
        belief_hidden: int = 128,
    ):
        super().__init__()
        self.extractor   = FeatureExtractor(visual_enc, text_enc, signal_enc)
        self.align       = SpatialAlignLayer(d_model=d_model, G=G)
        self.pool_layer  = UnifiedTokenPool(d_model=d_model, G=G)
        self.qformer     = QFormer(
            n_queries=n_queries, d_model=d_model,
            n_layers=n_layers,   n_heads=n_heads,
            ffn_dim=ffn_dim,     dropout=dropout,
        )
        self.belief_head = InterpretableBeliefHead(
            d_model=d_model, hidden=belief_hidden)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = next(self.align.parameters()).device
        # Move image tensors to device
        keys = ["ref_visual","ref_text","ref_signal",
                "test_visual","test_text","test_signal"]
        imgs = {k: batch[k].to(device) for k in keys if k in batch}
        # 1. Extract spatial features (frozen, no grad)
        feats   = self.extractor(**imgs)
        # 2. Align to (B, 49, 256) per stream
        aligned = self.align(feats)
        # 3. Build unified token pool (B, 294, 256)
        pool    = self.pool_layer(aligned)
        # 4. Q-Former -> (B, 16, 256)
        queries, last_attn = self.qformer(pool)
        # 5. Belief head
        out = self.belief_head(queries)
        out["last_attn"] = last_attn
        return out

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_counts(self):
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"frozen": frozen, "trainable": trainable, "total": frozen + trainable}
