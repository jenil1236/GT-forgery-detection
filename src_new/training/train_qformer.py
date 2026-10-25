from __future__ import annotations
import os, time, glob, traceback
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src_new.utils.checkpoint import (
    save_checkpoint, find_latest_checkpoint,
    find_best_checkpoint, checkpoint_filename,
)
from src_new.utils.logger import Logger


def _next_err(error_dir):
    existing = glob.glob(os.path.join(error_dir, "[0-9]*.txt"))
    if not existing: return 0
    idxs = []
    for p in existing:
        try: idxs.append(int(os.path.basename(p).split("_")[0]))
        except: pass
    return max(idxs) + 1 if idxs else 0

def _write_err(error_dir, tag):
    if not error_dir: return
    os.makedirs(error_dir, exist_ok=True)
    idx = _next_err(error_dir)
    p = os.path.join(error_dir, f"{idx}_{tag}.txt")
    with open(p, "w") as f: traceback.print_exc(file=f)
    print(f"[ERROR] Traceback -> {p}")


class QFormerTrainer:
    """
    Stage-2 trainer for the ForensicVerifier (Q-Former architecture).

    Loss: Binary Cross-Entropy on final belief.
    Metrics: accuracy, AUC, loss.
    Optimiser: AdamW with cosine LR decay.
    """

    def __init__(self, model, cfg, logger: Logger,
                 device: str = "cuda", error_log_dir: str = ""):
        self.model         = model.to(device)
        self.cfg           = cfg
        self.logger        = logger
        self.device        = device
        self.error_log_dir = error_log_dir
        self.best_auc      = 0.0
        self.start_epoch   = 0

        self.criterion = nn.BCELoss()

        self.optimizer = torch.optim.AdamW(
            model.trainable_params(),
            lr=cfg.qf_lr,
            weight_decay=cfg.qf_weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.qf_epochs,
            eta_min=cfg.qf_lr * 0.01,
        )

    # ── train epoch ──────────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict:
        try:
            return self._train_impl(loader, epoch)
        except Exception:
            _write_err(self.error_log_dir, f"qf_train_ep{epoch}")
            raise

    def _train_impl(self, loader: DataLoader, epoch: int) -> Dict:
        self.model.train()
        total_loss = n_correct = n_total = 0
        n_batches = len(loader)
        log_interval = max(1, n_batches // 5)

        for step, batch in enumerate(loader):
            labels = batch["label"].float().to(self.device)  # (B,)
            self.optimizer.zero_grad(set_to_none=True)

            out    = self.model(batch)
            belief = out["belief"]  # (B,) in [0,1]
            loss   = self.criterion(belief, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.trainable_params(), 1.0)
            self.optimizer.step()

            preds     = (belief >= 0.5).long()
            n_correct += (preds == labels.long()).sum().item()
            n_total   += labels.size(0)
            total_loss += loss.item()

            if step % log_interval == 0 or step == n_batches - 1:
                pct = 100.0 * (step + 1) / n_batches
                acc = n_correct / max(n_total, 1)
                self.logger.info(
                    f"  [QF Ep {epoch:02d}] Batch {step+1:04d}/{n_batches} ({pct:5.1f}%) | "
                    f"loss={loss.item():.4f} (avg={total_loss/(step+1):.4f}) | "
                    f"acc={acc:.4f}"
                )

        self.scheduler.step()
        return {
            "train_loss": total_loss / max(n_batches, 1),
            "train_acc":  n_correct / max(n_total, 1),
            "lr":         self.scheduler.get_last_lr()[0],
        }

    # ── validate ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> Dict:
        try:
            return self._val_impl(loader, epoch)
        except Exception:
            _write_err(self.error_log_dir, f"qf_val_ep{epoch}")
            raise

    def _val_impl(self, loader: DataLoader, epoch: int) -> Dict:
        self.model.eval()
        total_loss = n_correct = n_total = 0
        all_scores: List[float] = []
        all_labels: List[int]   = []
        n_batches = len(loader)

        for val_step, batch in enumerate(loader):
            labels = batch["label"].float().to(self.device)
            out    = self.model(batch)
            belief = out["belief"]
            loss   = self.criterion(belief, labels)
            preds  = (belief >= 0.5).long()
            n_correct += (preds == labels.long()).sum().item()
            n_total   += labels.size(0)
            total_loss += loss.item()
            all_scores.extend(belief.cpu().tolist())
            all_labels.extend(labels.cpu().long().tolist())
            if val_step == n_batches - 1:
                self.logger.info(
                    f"  [QF Ep {epoch:02d}] Val done | "
                    f"val_loss={total_loss/max(n_batches,1):.4f} | "
                    f"val_acc={n_correct/max(n_total,1):.4f}"
                )

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.5
        except Exception:
            auc = 0.5

        return {
            "val_loss": total_loss / max(n_batches, 1),
            "val_acc":  n_correct / max(n_total, 1),
            "val_auc":  float(auc),
        }

    # ── run ──────────────────────────────────────────────────────────────

    def run(self, train_loader: DataLoader, val_loader: DataLoader,
            resume: bool = False) -> None:
        self.logger.info("=" * 60)
        self.logger.info("Q-FORMER FORENSIC VERIFIER TRAINING")
        self.logger.info(f"  Epochs        : {self.cfg.qf_epochs}")
        self.logger.info(f"  Batch size    : {self.cfg.stage1_batch_size}")
        self.logger.info(f"  LR            : {self.cfg.qf_lr}")
        pc = self.model.param_counts()
        self.logger.info(f"  Trainable     : {pc['trainable']:,}")
        self.logger.info(f"  Frozen        : {pc['frozen']:,}")
        self.logger.info("=" * 60)

        if resume:
            ck = find_latest_checkpoint(self.cfg.checkpoint_dir, "qformer")
            if ck:
                self.logger.info(f"Resuming from {ck}")
                try:
                    import numpy as np
                    torch.serialization.add_safe_globals([np.core.multiarray.scalar,np.dtype])
                except Exception: pass
                try:
                    state = torch.load(ck, map_location=self.device, weights_only=True)
                except Exception:
                    state = torch.load(ck, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state["model_state"])
                self.start_epoch = state["epoch"] + 1
                self.best_auc    = state.get("best_metric", 0.0)

        if self.start_epoch >= self.cfg.qf_epochs:
            self.logger.info("Q-Former training already complete, skipping.")
            return

        epoch_bar = tqdm(
            range(self.start_epoch, self.cfg.qf_epochs),
            desc="QFormer", unit="epoch", dynamic_ncols=True,
            initial=self.start_epoch, total=self.cfg.qf_epochs,
        )
        for epoch in epoch_bar:
            t0 = time.time()
            tm = self.train_epoch(train_loader, epoch)
            vm = self.validate(val_loader, epoch)
            self.logger.log_epoch(epoch, phase="qformer", **{**tm, **vm})

            is_best = vm["val_auc"] > self.best_auc
            if is_best: self.best_auc = vm["val_auc"]

            if (epoch + 1) % self.cfg.checkpoint_every_n_epochs == 0 or is_best:
                tag = "BEST MODEL" if is_best else "checkpoint"
                self.logger.info(
                    f"  >>> [{tag}] Epoch {epoch:03d} | "
                    f"val_auc={vm['val_auc']:.4f} (best={self.best_auc:.4f})"
                )
                save_checkpoint(
                    {
                        "epoch":        epoch,
                        "stage":        "qformer",
                        "model_state":  self.model.state_dict(),
                        "best_metric":  self.best_auc,
                        "metric_name":  "val_auc",
                        "config":       self.cfg.to_dict(),
                    },
                    self.cfg.checkpoint_dir,
                    checkpoint_filename("qformer", epoch),
                    is_best=is_best,
                )

            elapsed = time.time() - t0
            epoch_bar.set_postfix(
                loss=f"{tm['train_loss']:.4f}",
                auc=f"{vm['val_auc']:.4f}",
                best="*" if is_best else "",
            )
            self.logger.info(
                f"QF Epoch {epoch:03d}/{self.cfg.qf_epochs-1} | "
                f"loss={tm['train_loss']:.4f} | acc={tm['train_acc']:.4f} | "
                f"val_auc={vm['val_auc']:.4f} | "
                f"{'BEST ' if is_best else ''}{elapsed:.1f}s"
            )

        self.logger.info("Q-Former training complete.")
