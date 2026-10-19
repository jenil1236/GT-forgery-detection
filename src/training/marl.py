"""
marl.py - Stage 2: Multi-Agent PPO (MAPPO) cooperative training.

Alternating schedule (prevents private code collapse):
  Phase A (judge):  freeze experts, train judge 5 epochs
  Phase B (expert): freeze judge,  train experts 5 epochs
  Repeat for 5 cycles (50 total epochs).

Expert credit: Market Scoring Rule
Judge credit:  Proper Scoring Rule with query cost
"""

from __future__ import annotations
import os
import time
import glob
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.judge import LogOddsJudge, DECLARE_ACTION
from src.training.losses import (
    ProperScoringRuleLoss, DecorrelationPenalty, ContrastiveAlignmentLoss,
)
from src.utils.checkpoint import save_checkpoint, find_latest_checkpoint, checkpoint_filename
from src.utils.logger import Logger


def _next_error_index(error_log_dir: str) -> int:
    existing = glob.glob(os.path.join(error_log_dir, "[0-9]*.txt"))
    if not existing:
        return 0
    indices = []
    for p in existing:
        try:
            indices.append(int(os.path.basename(p).split('_')[0]))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def _write_error_file(error_log_dir: str, tag: str) -> None:
    """Write traceback to error_log_dir/<N>_<tag>.txt (N auto-increments)."""
    if not error_log_dir:
        return
    os.makedirs(error_log_dir, exist_ok=True)
    idx = _next_error_index(error_log_dir)
    fpath = os.path.join(error_log_dir, f"{idx}_{tag}.txt")
    try:
        with open(fpath, "w") as fh:
            traceback.print_exc(file=fh)
        print(f"[ERROR] Traceback saved -> {fpath}")
    except Exception:
        pass


@dataclass
class Transition:
    obs_lambda: torch.Tensor
    obs_beliefs: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    reward: torch.Tensor
    value: torch.Tensor
    done: torch.Tensor
    step: int
    available_mask: torch.Tensor = None  # stored for PPO recomputation


@dataclass
class EpisodeBuffer:
    transitions: List[Transition] = field(default_factory=list)
    final_belief: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    n_queries: Optional[torch.Tensor] = None


class MAPPOTrainer:
    """
    Multi-Agent PPO trainer with centralised critic (CTDE).
    """

    def __init__(
        self,
        expert_system,
        judge,
        cfg,
        logger: Logger,
        device: str = "cuda",
        error_log_dir: str = "",
    ):
        self.expert_system = expert_system.to(device)
        self.judge = judge.to(device)
        self.cfg = cfg
        self.logger = logger
        self.device = device
        self.error_log_dir = error_log_dir
        self.best_metric = 0.0
        self.start_epoch = 0
        self.modalities = ["visual", "text", "signal"]
        self.decorr_penalty = DecorrelationPenalty()
        self.alignment_loss = ContrastiveAlignmentLoss(margin=1.0)
        self._rebuild_opts()

    def _rebuild_opts(self) -> None:
        ep = [p for p in self.expert_system.parameters() if p.requires_grad]
        jp = [p for p in self.judge.parameters() if p.requires_grad]
        if not ep:
            ep = [nn.Parameter(torch.zeros(1))]
        if not jp:
            jp = [nn.Parameter(torch.zeros(1))]
        self.opt_experts = torch.optim.Adam(ep, lr=self.cfg.stage2_lr_actor)
        self.opt_judge   = torch.optim.Adam(jp, lr=self.cfg.stage2_lr_critic)

    def _set_phase(self, phase: str) -> None:
        if phase == "judge":
            for p in self.expert_system.parameters():
                p.requires_grad = False
            for p in self.judge.parameters():
                p.requires_grad = True
        else:
            for p in self.expert_system.parameters():
                p.requires_grad = True
            for p in self.judge.parameters():
                p.requires_grad = False
        self._rebuild_opts()

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout_episode(self, batch: Dict, epoch: int) -> EpisodeBuffer:
        try:
            return self._rollout_episode_impl(batch, epoch)
        except Exception:
            tag = f"error_stage2_rollout_ep{epoch}"
            _write_error_file(self.error_log_dir, tag)
            raise

    def _rollout_episode_impl(self, batch: Dict, epoch: int) -> EpisodeBuffer:
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        B = batch["label"].size(0)
        d = self.device
        labels = batch["label"].to(d)
        lambda_t = torch.zeros(B, device=d)
        eb_slot = torch.full((B, self.cfg.n_experts), 0.5, device=d)
        avail = torch.ones(B, self.cfg.n_experts + 1, dtype=torch.bool, device=d)
        buf = EpisodeBuffer(labels=labels)
        done = torch.zeros(B, dtype=torch.bool, device=d)
        n_q = torch.zeros(B, device=d)

        eo = self.expert_system(batch, deterministic=False)
        # Sanitise expert outputs: replace any NaN/Inf with zeros
        for nm in eo:
            for key in ("eR", "eT", "belief"):
                if key in eo[nm] and torch.is_tensor(eo[nm][key]):
                    if torch.isnan(eo[nm][key]).any() or torch.isinf(eo[nm][key]).any():
                        eo[nm][key] = torch.nan_to_num(eo[nm][key], nan=0.0, posinf=1.0, neginf=-1.0)
        jout = None

        for step in range(self.cfg.max_episode_steps):
            jout = self.judge(
                lambda_t=lambda_t, step=step,
                max_steps=self.cfg.max_episode_steps,
                expert_beliefs=eb_slot,
                available_mask=avail,
            )
            action = jout["action"]
            rew = torch.zeros(B, device=d)
            nd = done.clone()

            for b in range(B):
                if done[b]:
                    continue
                a = action[b].item()
                is_last = (step == self.cfg.max_episode_steps - 1)
                if a == DECLARE_ACTION or is_last:
                    fb = jout["belief"][b].clamp(1e-7, 1 - 1e-7)
                    y = labels[b].float()
                    ls = y * torch.log(fb) + (1 - y) * torch.log(1 - fb)
                    rew[b] = ls - self.cfg.query_cost * n_q[b]
                    nd[b] = True
                elif a < self.cfg.n_experts:
                    nm = self.modalities[a]
                    if nm in eo:
                        q = eo[nm]["belief"][b].squeeze().clamp(1e-7, 1 - 1e-7)
                    else:
                        q = torch.tensor(0.5, device=d)
                    ob = torch.sigmoid(lambda_t[b]).clamp(1e-7, 1 - 1e-7)
                    lambda_t[b] = LogOddsJudge.update_lambda(
                        lambda_t[b:b+1], q.unsqueeze(0)
                    ).squeeze()
                    nb = torch.sigmoid(lambda_t[b]).clamp(1e-7, 1 - 1e-7)
                    yb = labels[b].float()
                    msr = (yb * torch.log(nb) + (1 - yb) * torch.log(1 - nb)
                           - yb * torch.log(ob) - (1 - yb) * torch.log(1 - ob))
                    rew[b] = msr - self.cfg.query_cost
                    eb_slot[b, a] = q
                    avail[b, a] = False
                    n_q[b] += 1

            buf.transitions.append(Transition(
                obs_lambda=lambda_t.clone().detach(),
                obs_beliefs=eb_slot.clone().detach(),
                action=action.detach(),
                log_prob=jout["log_prob"].detach(),
                reward=rew.detach(),
                value=jout["value"].detach(),
                done=nd.clone(),
                step=step,
                available_mask=avail.clone().detach(),
            ))
            done = nd
            if done.all():
                break

        buf.final_belief = jout["belief"].detach() if jout is not None else None
        buf.n_queries = n_q.detach()
        return buf

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def _gae(self, buf: EpisodeBuffer):
        T = len(buf.transitions)
        B = buf.labels.size(0)
        d = self.device
        rew = torch.stack([t.reward for t in buf.transitions])
        val = torch.stack([t.value  for t in buf.transitions])
        don = torch.stack([t.done.float() for t in buf.transitions])
        ret = torch.zeros(T, B, device=d)
        adv = torch.zeros(T, B, device=d)
        lg  = torch.zeros(B, device=d)
        for t in reversed(range(T)):
            nv = torch.zeros(B, device=d) if t == T - 1 else val[t + 1]
            delta = rew[t] + self.cfg.gamma * nv * (1 - don[t]) - val[t]
            lg = delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - don[t]) * lg
            adv[t] = lg
            ret[t] = lg + val[t]
        af = adv.reshape(-1)
        adv = (adv - af.mean()) / (af.std() + 1e-8)
        return ret, adv

    # ------------------------------------------------------------------
    # PPO judge update
    # ------------------------------------------------------------------

    def _judge_ppo(self, buf: EpisodeBuffer, ret, adv) -> Dict:
        tp = tv = te = 0.0
        clips = []
        for _ in range(self.cfg.ppo_epochs):
            for t, tr in enumerate(buf.transitions):
                mask = tr.available_mask.to(self.device) if tr.available_mask is not None else None
                lp, ent, v = self.judge.compute_log_prob_entropy(
                    tr.obs_lambda.to(self.device),
                    tr.step,
                    self.cfg.max_episode_steps,
                    tr.obs_beliefs.to(self.device),
                    tr.action.to(self.device),
                    available_mask=mask,
                )
                olp = tr.log_prob.to(self.device)
                a = adv[t].to(self.device)
                r = ret[t].to(self.device)
                ratio = torch.exp(lp - olp)
                s1 = ratio * a
                s2 = torch.clamp(ratio, 1 - self.cfg.ppo_clip, 1 + self.cfg.ppo_clip) * a
                pl = -torch.min(s1, s2).mean()
                vl = 0.5 * (v - r).pow(2).mean()
                el = -ent.mean()
                loss = pl + self.cfg.value_loss_coef * vl + self.cfg.entropy_bonus * el
                self.opt_judge.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.judge.parameters(), 0.5)
                self.opt_judge.step()
                clips.append(((ratio - 1.0).abs() > self.cfg.ppo_clip).float().mean().item())
                tp += pl.item()
                tv += vl.item()
                te += ent.mean().item()
        n = max(len(buf.transitions) * self.cfg.ppo_epochs, 1)
        return {
            "judge_policy_loss": tp / n,
            "judge_value_loss": tv / n,
            "judge_entropy": te / n,
            "ppo_clip_fraction": sum(clips) / max(len(clips), 1),
        }

    # ------------------------------------------------------------------
    # Expert update
    # ------------------------------------------------------------------

    def _expert_update(self, bd: Dict, labels: torch.Tensor) -> Dict:
        aln = self.alignment_loss
        labels = labels.to(self.device)
        eo = self.expert_system(bd, deterministic=False)
        eRs, eTs = self.expert_system.get_embeddings(eo)
        # start with differentiable zero anchored to model params
        total = (list(self.expert_system.parameters())[0] * 0).sum()
        for nm in self.modalities:
            if nm not in eo:
                continue
            total = total + aln(eo[nm]["eR"], eo[nm]["eT"], labels)
        decorr = (list(self.expert_system.parameters())[0] * 0).sum()  # zero but differentiable
        if eRs.size(1) > 1:
            decorr = self.decorr_penalty(eRs) + self.decorr_penalty(eTs)
            total = total + self.cfg.decorr_penalty * decorr
        self.opt_experts.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(self.expert_system.parameters(), 0.5)
        self.opt_experts.step()
        return {
            "expert_align_loss": total.item(),
            "decorr_penalty": float(decorr.item() if torch.is_tensor(decorr) else decorr),
        }

    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict:
        try:
            return self._train_epoch_impl(loader, epoch)
        except Exception:
            tag = f"error_stage2_epoch{epoch}"
            _write_error_file(self.error_log_dir, tag)
            raise

    def _train_epoch_impl(self, loader: DataLoader, epoch: int) -> Dict:
        phase = "judge" if (epoch % 10) < 5 else "expert"
        self._set_phase(phase)
        self.expert_system.train()
        self.judge.train()
        tr_tot = acc_tot = 0.0
        extra = {}
        n = 0
        n_batches_total = len(loader)
        log_interval = max(1, n_batches_total // 5)
        for batch_idx, batch in enumerate(loader):
            buf = self._rollout_episode(batch, epoch)
            ret, adv = self._gae(buf)
            bd = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            if phase == "judge":
                extra.update(self._judge_ppo(buf, ret, adv))
            else:
                extra.update(self._expert_update(bd, batch["label"]))
            if buf.final_belief is not None:
                pred = (buf.final_belief >= 0.5).long().cpu()
                acc_tot += (pred == buf.labels.cpu()).float().mean().item()
            tr_tot += sum(t.reward.mean().item() for t in buf.transitions)
            n += 1
            if batch_idx % log_interval == 0 or batch_idx == n_batches_total - 1:
                pct = 100.0 * (batch_idx + 1) / n_batches_total
                avg_rew = tr_tot / max(n, 1)
                avg_acc = acc_tot / max(n, 1)
                self.logger.info(
                    f"  [S2 Ep {epoch:02d} {phase}] "
                    f"Batch {batch_idx+1:04d}/{n_batches_total} ({pct:5.1f}%) | "
                    f"avg_reward={avg_rew:.4f} | acc={avg_acc:.4f}"
                )
        m = {
            "mean_episode_reward": tr_tot / max(n, 1),
            "judge_accuracy": acc_tot / max(n, 1),
            "phase": phase,
        }
        m.update(extra)
        return m

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> Dict:
        try:
            return self._validate_impl(loader, epoch)
        except Exception:
            tag = f"error_stage2_val_epoch{epoch}"
            _write_error_file(self.error_log_dir, tag)
            raise

    def _validate_impl(self, loader: DataLoader, epoch: int) -> Dict:
        self.expert_system.eval()
        self.judge.eval()
        tacc = 0.0
        ab: List[float] = []
        al: List[int] = []
        n = 0
        n_val_total = len(loader)
        for val_idx, batch in enumerate(loader):
            buf = self._rollout_episode(batch, epoch)
            if buf.final_belief is not None:
                pred = (buf.final_belief >= 0.5).long().cpu()
                tacc += (pred == buf.labels.cpu()).float().mean().item()
                ab.extend(buf.final_belief.cpu().tolist())
                al.extend(buf.labels.cpu().tolist())
            n += 1
            if val_idx == n_val_total - 1:
                self.logger.info(
                    f"  [S2 Ep {epoch:02d}] Val done | "
                    f"acc={tacc/max(n,1):.4f}"
                )
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(al, ab) if len(set(al)) > 1 else 0.5
        except Exception:
            auc = 0.5
        return {"val_accuracy": tacc / max(n, 1), "val_auc": auc}

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume: bool = False,
    ) -> None:
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2: MAPPO Cooperative Multi-Agent Training")
        self.logger.info(f"  Total epochs: {self.cfg.stage2_total_epochs}")
        self.logger.info(f"  PPO clip:     {self.cfg.ppo_clip}")
        self.logger.info(f"  Query cost:   {self.cfg.query_cost}")
        self.logger.info("=" * 60)

        if resume:
            ck = find_latest_checkpoint(self.cfg.checkpoint_dir, "stage2")
            if ck:
                self.logger.info(f"Resuming Stage 2 from {ck}")
                state = torch.load(ck, map_location=self.device, weights_only=True)
                self.start_epoch = state["epoch"] + 1
                self.best_metric = state.get("best_metric", 0.0)

        if self.start_epoch >= self.cfg.stage2_total_epochs:
            self.logger.info("Stage 2 already complete, skipping.")
            return

        from tqdm import tqdm as _tqdm
        epoch_bar = _tqdm(
            range(self.start_epoch, self.cfg.stage2_total_epochs),
            desc="Stage2", unit="epoch", dynamic_ncols=True,
            initial=self.start_epoch,
            total=self.cfg.stage2_total_epochs,
        )
        for epoch in epoch_bar:
            t0 = time.time()
            tm = self.train_epoch(train_loader, epoch)
            vm = self.validate(val_loader, epoch)
            # Remove "phase" from tm to avoid duplicate kwarg with log_epoch
            tm_log = {k: v for k, v in tm.items() if k != "phase"}
            self.logger.log_epoch(epoch, phase="stage2", **{**tm_log, **vm})
            best = vm["val_auc"] > self.best_metric
            if best:
                self.best_metric = vm["val_auc"]
            if (epoch + 1) % self.cfg.checkpoint_every_n_epochs == 0 or best:
                _ckpt_reason = "BEST MODEL" if best else "checkpoint"
                self.logger.info(
                    f"  >>> [{_ckpt_reason}] Epoch {epoch:03d} saving... | "
                    f"val_auc={vm['val_auc']:.4f} (best={self.best_metric:.4f})"
                )
                st = {
                    "epoch": epoch,
                    "stage": "stage2",
                    "model_state": {
                        "experts": self.expert_system.state_dict(),
                        "judge": self.judge.state_dict(),
                    },
                    "best_metric": self.best_metric,
                    "metric_name": "val_auc",
                    "config": self.cfg.to_dict(),
                    "run_id": self.cfg.run_id,
                }
                save_checkpoint(
                    st, self.cfg.checkpoint_dir,
                    checkpoint_filename("stage2", epoch),
                    is_best=best,
                )
                self.logger.info(
                    f"  >>> Checkpoint saved -> "
                    f"{self.cfg.checkpoint_dir}/{checkpoint_filename('stage2', epoch)}"
                )
            tag = "BEST " if best else ""
            epoch_bar.set_postfix(
                reward=f"{tm.get('mean_episode_reward',0):.4f}",
                auc=f"{vm['val_auc']:.4f}",
                best="*" if best else "",
            )
            self.logger.info(
                f"Stage2 Ep {epoch:03d} | "
                f"reward={tm.get('mean_episode_reward', 0):.4f} | "
                f"acc={tm.get('judge_accuracy', 0):.4f} | "
                f"auc={vm['val_auc']:.4f} | "
                f"{tag}{time.time() - t0:.1f}s"
            )

        self.logger.info("Stage 2 complete.")
