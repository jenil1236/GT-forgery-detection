"""
pretrain.py - Stage 1: metric learning pretraining of encoders.

Trains visual, text, and signal encoders with InfoNCE + contrastive alignment.
Encoders learn to produce embeddings where:
  - genuine (ref, test) pairs are close
  - forged (ref, test) pairs are far apart

This stage is the key to zero-shot generalisation:
the model learns HOW to compare, not WHAT a specific document looks like.
"""

from __future__ import annotations
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.visual_encoder import VisualEncoder
from src.models.text_encoder import TextEncoder
from src.models.signal_encoder import SignalEncoder
from src.training.losses import InfoNCELoss, ContrastiveAlignmentLoss, TripletLoss
from src.utils.checkpoint import (
    save_checkpoint, find_latest_checkpoint,
    checkpoint_filename, find_best_checkpoint
)
from src.utils.logger import Logger


class Stage1Trainer:
    """
    Trains visual + signal encoders with metric learning.
    Text encoder projector is also trained here (TrOCR backbone frozen).

    Pretraining strategy:
      - Triplets: anchor=reference, positive=genuine_test, negative=forged_test
      - Loss = InfoNCE + alpha * ContrastiveAlignment + beta * Triplet
      - AdamW, cosine LR decay
      - Save best checkpoint based on validation separation ratio
    """

    def __init__(
        self,
        visual_encoder: VisualEncoder,
        text_encoder: TextEncoder,
        signal_encoder: SignalEncoder,
        cfg,
        logger: Logger,
        device: str = "cuda",
    ):
        self.visual_enc = visual_encoder.to(device)
        self.text_enc   = text_encoder.to(device)
        self.signal_enc = signal_encoder.to(device)
        self.cfg = cfg
        self.logger = logger
        self.device = device

        self.infonce = InfoNCELoss(temperature=cfg.infonce_temperature)
        self.alignment = ContrastiveAlignmentLoss(margin=1.0)
        self.triplet = TripletLoss(margin=cfg.triplet_margin)

        # Optimiser: only train projectors + signal encoder (visual backbone mostly frozen)
        params = (
            list(self.visual_enc.projector.parameters()) +
            list(self.text_enc.projector.parameters()) +
            list(self.signal_enc.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            params,
            lr=cfg.stage1_lr,
            weight_decay=cfg.stage1_weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.stage1_epochs,
            eta_min=cfg.stage1_lr * 0.01,
        )

        # FIX: use torch.amp instead of deprecated torch.cuda.amp
        use_amp = (device == "cuda") and cfg.mixed_precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self._use_amp = use_amp

        self.start_epoch = 0
        self.best_sep_ratio = 0.0

    def _compute_separation_ratio(
        self,
        eR: torch.Tensor,
        eT_genuine: torch.Tensor,
        eT_forged: torch.Tensor,
    ) -> float:
        """inter-class distance / intra-class distance — higher is better."""
        n = min(eR.size(0), eT_genuine.size(0), eT_forged.size(0))
        if n == 0:
            return 0.0
        intra = torch.nn.functional.pairwise_distance(eR[:n], eT_genuine[:n]).mean()
        inter = torch.nn.functional.pairwise_distance(eR[:n], eT_forged[:n]).mean()
        if intra.item() < 1e-8:
            return 0.0
        return (inter / intra).item()

    def _forward_encoders(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict, Dict, Dict]:
        """Run all three encoders on a batch. Returns (visual_out, text_out, signal_out)."""
        def enc_pair(encoder, ref_key, test_key):
            ref = batch[ref_key].to(self.device)
            test = batch[test_key].to(self.device)
            eR = encoder(ref)
            eT = encoder(test)
            return eR, eT

        # FIX: torch.amp.autocast requires explicit device_type argument
        with torch.amp.autocast("cuda", enabled=self._use_amp):
            eR_v, eT_v = enc_pair(self.visual_enc, "ref_visual", "test_visual")
            eR_t, eT_t = enc_pair(self.text_enc, "ref_text", "test_text")
            eR_s, eT_s = enc_pair(self.signal_enc, "ref_signal", "test_signal")

        return (eR_v, eT_v), (eR_t, eT_t), (eR_s, eT_s)

    def _compute_loss(
        self,
        eR: torch.Tensor,
        eT: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute combined loss for one encoder."""
        labels = labels.to(self.device)

        gen_mask = (labels == 0)
        frg_mask = (labels == 1)

        if gen_mask.sum() == 0 or frg_mask.sum() == 0:
            dist = torch.nn.functional.pairwise_distance(eR, eT)
            loss = dist[gen_mask].mean() if gen_mask.sum() > 0 else torch.tensor(0.0, device=self.device)
            return loss, {"loss": loss.item()}

        eR_gen = eR[gen_mask]
        eT_gen = eT[gen_mask]
        eT_frg = eT[frg_mask]

        n = min(len(eR_gen), len(eT_frg))
        if n > 0:
            l_infonce = self.infonce(eR_gen[:n], eT_gen[:n], eT_frg[:n])
        else:
            l_infonce = torch.tensor(0.0, device=self.device)

        l_align = self.alignment(eR, eT, labels)

        n_triplet = min(len(eR_gen), len(eT_frg))
        if n_triplet > 0:
            l_triplet = self.triplet(eR_gen[:n_triplet], eT_gen[:n_triplet], eT_frg[:n_triplet])
        else:
            l_triplet = torch.tensor(0.0, device=self.device)

        loss = l_infonce + self.cfg.contrastive_alpha * l_align + 0.1 * l_triplet

        metrics = {
            "loss_infonce": l_infonce.item(),
            "loss_align": l_align.item(),
            "loss_triplet": l_triplet.item(),
            "loss_total": loss.item(),
        }
        return loss, metrics

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict:
        self.visual_enc.train()
        self.text_enc.train()
        self.signal_enc.train()

        total_loss = 0.0
        total_sep = 0.0
        n_batches = 0

        for step, batch in enumerate(
            self.logger.progress(loader, desc=f"  Stage1 Epoch {epoch}", leave=False)
        ):
            self.optimizer.zero_grad(set_to_none=True)
            labels = batch["label"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self._use_amp):
                (eR_v, eT_v), (eR_t, eT_t), (eR_s, eT_s) = self._forward_encoders(batch)
                l_v, m_v = self._compute_loss(eR_v, eT_v, labels)
                l_t, m_t = self._compute_loss(eR_t, eT_t, labels)
                l_s, m_s = self._compute_loss(eR_s, eT_s, labels)
                loss = l_v + l_t + l_s

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                list(self.visual_enc.parameters()) +
                list(self.text_enc.parameters()) +
                list(self.signal_enc.parameters()),
                max_norm=1.0,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            gen_mask = (labels == 0)
            frg_mask = (labels == 1)
            if gen_mask.sum() > 0 and frg_mask.sum() > 0:
                sep = self._compute_separation_ratio(
                    eR_v[gen_mask], eT_v[gen_mask], eT_v[frg_mask]
                )
                total_sep += sep

            total_loss += loss.item()
            n_batches += 1

            if step % self.cfg.log_every_n_steps == 0:
                self.logger.log_step(
                    epoch=epoch, step=step,
                    loss_visual=m_v["loss_total"],
                    loss_text=m_t["loss_total"],
                    loss_signal=m_s["loss_total"],
                    loss_total=loss.item(),
                )

        self.scheduler.step()

        return {
            "train_loss": total_loss / max(n_batches, 1),
            "separation_ratio": total_sep / max(n_batches, 1),
            "lr": self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> Dict:
        self.visual_enc.eval()
        self.text_enc.eval()
        self.signal_enc.eval()

        total_loss = 0.0
        total_sep = 0.0
        n_batches = 0

        for batch in self.logger.progress(loader, desc=f"  Stage1 Val {epoch}", leave=False):
            labels = batch["label"].to(self.device)
            (eR_v, eT_v), (eR_t, eT_t), (eR_s, eT_s) = self._forward_encoders(batch)
            l_v, _ = self._compute_loss(eR_v, eT_v, labels)
            l_t, _ = self._compute_loss(eR_t, eT_t, labels)
            l_s, _ = self._compute_loss(eR_s, eT_s, labels)
            loss = l_v + l_t + l_s
            total_loss += loss.item()

            gen_mask = (labels == 0)
            frg_mask = (labels == 1)
            if gen_mask.sum() > 0 and frg_mask.sum() > 0:
                sep = self._compute_separation_ratio(
                    eR_v[gen_mask], eT_v[gen_mask], eT_v[frg_mask]
                )
                total_sep += sep
            n_batches += 1

        return {
            "val_loss": total_loss / max(n_batches, 1),
            "val_separation_ratio": total_sep / max(n_batches, 1),
        }

    def save_encoders(self, epoch: int, metrics: dict, is_best: bool = False) -> None:
        """Save all three encoder checkpoints."""
        for name, model in [
            ("stage1_visual", self.visual_enc),
            ("stage1_text",   self.text_enc),
            ("stage1_signal", self.signal_enc),
        ]:
            state = {
                "epoch": epoch,
                "stage": name,
                "model_state": model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_metric": self.best_sep_ratio,
                "metric_name": "val_separation_ratio",
                "config": self.cfg.to_dict(),
                "run_id": self.cfg.run_id,
            }
            fname = checkpoint_filename(name, epoch)
            save_checkpoint(
                state, self.cfg.checkpoint_dir, fname, is_best=is_best
            )

    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume: bool = False,
    ) -> None:
        """Full Stage 1 training loop."""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 1: Encoder Pretraining (Metric Learning)")
        self.logger.info(f"  Epochs:     {self.cfg.stage1_epochs}")
        self.logger.info(f"  Batch size: {self.cfg.stage1_batch_size}")
        self.logger.info(f"  LR:         {self.cfg.stage1_lr}")
        self.logger.info(f"  Device:     {self.device}")
        self.logger.info("=" * 60)

        if resume:
            ckpt_path = find_latest_checkpoint(self.cfg.checkpoint_dir, "stage1_visual")
            if ckpt_path:
                self.logger.info(f"Resuming Stage 1 from {ckpt_path}")
                state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                self.visual_enc.load_state_dict(state["model_state"])
                self.start_epoch = state["epoch"] + 1
                self.best_sep_ratio = state.get("best_metric", 0.0)
                for enc_name, enc in [("stage1_text", self.text_enc), ("stage1_signal", self.signal_enc)]:
                    best = find_best_checkpoint(self.cfg.checkpoint_dir, enc_name)
                    if best:
                        s = torch.load(best, map_location=self.device, weights_only=True)
                        enc.load_state_dict(s["model_state"])
            else:
                self.logger.info("No Stage 1 checkpoint found, starting fresh.")

        if self.start_epoch >= self.cfg.stage1_epochs:
            self.logger.info("Stage 1 already complete, skipping.")
            return

        for epoch in range(self.start_epoch, self.cfg.stage1_epochs):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)

            epoch_metrics = {**train_metrics, **val_metrics}
            self.logger.log_epoch(epoch, phase="stage1", **epoch_metrics)

            is_best = val_metrics["val_separation_ratio"] > self.best_sep_ratio
            if is_best:
                self.best_sep_ratio = val_metrics["val_separation_ratio"]

            if (epoch + 1) % self.cfg.checkpoint_every_n_epochs == 0 or is_best:
                self.save_encoders(epoch, epoch_metrics, is_best=is_best)

            elapsed = time.time() - t0
            self.logger.info(
                f"Stage1 Epoch {epoch:03d}/{self.cfg.stage1_epochs-1} | "
                f"loss={train_metrics['train_loss']:.4f} | "
                f"sep_ratio={val_metrics['val_separation_ratio']:.3f} | "
                f"{'BEST ' if is_best else ''}"
                f"{elapsed:.1f}s"
            )

        self.logger.info("Stage 1 complete.")
