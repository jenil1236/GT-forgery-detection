"""
logger.py — Structured logging: console (tqdm-aware) + per-run JSON metrics file.

Usage:
    logger = Logger(metrics_dir="output/metrics", run_id="run_20240101_120000")
    logger.log_step(epoch=1, step=10, loss=0.42, lr=1e-4)
    logger.log_epoch(epoch=1, train_loss=0.38, val_auc=0.87)
    logger.log_eval(results_dict)
    logger.close()
"""

from __future__ import annotations
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

from tqdm import tqdm


class Logger:
    """
    Dual-channel logger:
      - Console: Python logging with tqdm-safe output
      - File: newline-delimited JSON to output/metrics/<run_id>.jsonl
    """

    def __init__(
        self,
        metrics_dir: str,
        run_id: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
    ) -> None:
        self.run_id = run_id
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)

        # ── console handler (tqdm-safe) ──────────────────────────────────────
        self._console = logging.getLogger(f"gt_forgery.{run_id}")
        self._console.setLevel(level)
        self._console.handlers.clear()
        ch = _TqdmHandler(stream=sys.stdout)
        ch.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        self._console.addHandler(ch)
        self._console.propagate = False

        # ── metrics JSONL file ───────────────────────────────────────────────
        metrics_path = os.path.join(metrics_dir, f"{run_id}.jsonl")
        self._metrics_fh = open(metrics_path, "a", buffering=1)

        # ── optional plain-text log file ────────────────────────────────────
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            self._console.addHandler(fh)

        self.info(f"Logger initialised — run_id={run_id}")
        self.info(f"Metrics → {metrics_path}")

    # ── public convenience wrappers ──────────────────────────────────────────

    def info(self, msg: str) -> None:
        self._console.info(msg)

    def warning(self, msg: str) -> None:
        self._console.warning(msg)

    def error(self, msg: str) -> None:
        self._console.error(msg)

    def log_step(self, epoch: int, step: int, **metrics) -> None:
        """Log a training step (scalar metrics as kwargs)."""
        record = {"type": "step", "epoch": epoch, "step": step,
                  "ts": time.time(), **metrics}
        self._write(record)

    def log_epoch(self, epoch: int, phase: str = "train", **metrics) -> None:
        """Log end-of-epoch metrics."""
        record = {"type": "epoch", "epoch": epoch, "phase": phase,
                  "ts": time.time(), **metrics}
        self._write(record)
        # also print to console
        metric_str = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in metrics.items())
        self.info(f"[Epoch {epoch:03d}] {phase.upper()} | {metric_str}")

    def log_eval(self, results: Dict[str, Any], split: str = "test") -> None:
        """Log final evaluation results."""
        record = {"type": "eval", "split": split, "ts": time.time(), **results}
        self._write(record)
        self.info("=" * 60)
        self.info(f"EVALUATION RESULTS [{split.upper()}]")
        for k, v in results.items():
            if isinstance(v, float):
                self.info(f"  {k:<30s}: {v:.4f}")
            else:
                self.info(f"  {k:<30s}: {v}")
        self.info("=" * 60)

    def log_config(self, cfg_dict: dict) -> None:
        record = {"type": "config", "ts": time.time(), "config": cfg_dict}
        self._write(record)

    def progress(self, iterable, desc: str = "", total: Optional[int] = None, **kwargs):
        """Wrap an iterable in a tqdm progress bar."""
        return tqdm(iterable, desc=desc, total=total,
                    dynamic_ncols=True, leave=True, **kwargs)

    def close(self) -> None:
        self._metrics_fh.flush()
        self._metrics_fh.close()
        self.info("Logger closed.")

    # ── internal ─────────────────────────────────────────────────────────────

    def _write(self, record: dict) -> None:
        self._metrics_fh.write(json.dumps(record) + "\n")


class _TqdmHandler(logging.StreamHandler):
    """StreamHandler that uses tqdm.write to avoid breaking progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


def make_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")
