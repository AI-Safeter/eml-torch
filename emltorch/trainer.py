"""
Three-phase GPU-batched EML trainer.

Phase 1 — Fitting:    Adam on complex MSE, soft softmax selections.
Phase 2 — Hardening:  Ramp temp_inv + entropy penalty to push toward one-hot.
Phase 3 — Snapping:   argmax all logits, verify MSE stays low.

All phases run batched over F features × R restarts in a single GPU pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from .tree import BatchedEMLTree
from .symbolic import extract_expressions


# ------------------------------------------------------------------
# Config & result types
# ------------------------------------------------------------------

@dataclass
class EMLConfig:
    depth: int = 3
    num_restarts: int = 12
    num_vars: int = 1
    dtype: str = "complex64"       # "complex64" | "complex128" | "float32"
    device: str = "cuda:7"

    # Phase 1: Fitting
    fit_steps: int = 3000
    fit_lr: float = 3e-2
    grad_clip: float = 1.0
    # Selection mode during Phase 1:
    #   "softmax"     - original (depth-3+ collapses)
    #   "gumbel_soft" - Gumbel noise, continuous forward (Fix 1, default)
    fit_selection: str = "gumbel_soft"
    # Init strategy — Fix 4 (diverse restarts)
    #   init_scale: 0.1 = near-uniform softmax; 3-5 = sharply peaked
    #   init_mode:  "uniform" | "peaked"  (peaked → each restart random-tree)
    init_scale: float = 0.1
    init_mode: str = "uniform"

    # Training strategy — Fix 5 (bypass lethal topology-space gradient):
    #   "full"       - Phase 1 Adam on topology + Phase 2 hardening (original)
    #   "random"     - peaked init + evaluate only (pure random search)
    #   "hybrid"     - random search for topology discovery; gradient only
    #                  over affine leaf params once topology is fixed
    strategy: str = "full"
    # Hit threshold for declaring recovery at Phase 0 (random search)
    random_r2_target: float = 0.99

    # Phase 2: Hardening
    harden_steps: int = 1500
    harden_lr: float = 1e-2
    entropy_weight_final: float = 1e-2
    # Selection mode during Phase 2:
    #   "softmax"     - softmax with ramped temperature (original)
    #   "gumbel_hard" - hard straight-through (commits early)
    harden_selection: str = "softmax"

    # Phase 3: Snapping
    snap_mse_threshold: float = 1e-4

    # Preprocessing
    safe_shift: float = 0.0  # Add this offset to all data to avoid log(0). Set > 0 for SAE features.

    # Logging
    log_every: int = 500

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "complex64": torch.complex64,
            "complex128": torch.complex128,
            "float32": torch.float32,
            "float64": torch.float64,
        }[self.dtype]


@dataclass
class EMLResult:
    expressions: list[str]            # symbolic expression per feature
    mse_values: torch.Tensor          # (F,) best MSE per feature
    snapped_ok: torch.Tensor          # (F,) bool — MSE < threshold
    best_restart: torch.Tensor        # (F,) which restart won
    tree: BatchedEMLTree              # the fitted (snapped) tree
    total_time_s: float


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

class EMLTrainer:
    """Fit batched EML trees with 3-phase training."""

    def __init__(self, config: EMLConfig):
        self.cfg = config

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        var_names: list[str] | None = None,
    ) -> EMLResult:
        """
        Discover symbolic EML formulas: y ≈ f(x) for each feature.

        Args:
            x: Source activations — (F, N) or (F, V, N).
            y: Target activations — (F, N).
            var_names: Optional variable names for expression output.

        Returns:
            EMLResult with expressions, MSE, and the fitted tree.
        """
        cfg = self.cfg
        t0 = time.time()
        R = cfg.num_restarts

        # ---- Shape handling ----
        if x.dim() == 2:
            x = x.unsqueeze(1)           # (F, 1, N)
        F, V, N = x.shape
        assert y.shape == (F, N), f"y shape {y.shape} != expected ({F}, {N})"

        # ---- Preprocess: shift data to EML-safe range ----
        # SAE features are ReLU-gated (non-negative, many zeros).
        # EML's log(right) hits -inf on zeros. Shift by safe_shift to make
        # all values safely positive for log.
        if cfg.safe_shift > 0:
            x = x + cfg.safe_shift
            y = y + cfg.safe_shift

        # ---- Expand for restarts ----
        x_exp = x.repeat_interleave(R, dim=0).to(cfg.device)   # (F*R, V, N)
        y_exp = y.repeat_interleave(R, dim=0).to(cfg.device)   # (F*R, N)
        if cfg.torch_dtype.is_complex:
            y_exp = y_exp.to(cfg.torch_dtype)

        B = F * R

        # ---- Create tree ----
        # Strategy-specific init overrides
        if cfg.strategy == "random":
            init_scale = max(cfg.init_scale, 50.0)   # force hard one-hot
            init_mode = "peaked"
        else:
            init_scale = cfg.init_scale
            init_mode = cfg.init_mode

        tree = BatchedEMLTree(
            num_trees=B,
            depth=cfg.depth,
            num_vars=V,
            dtype=cfg.torch_dtype,
            device=cfg.device,
            init_scale=init_scale,
            init_mode=init_mode,
        )
        self._log(f"Tree: depth={cfg.depth}, {B} trees ({F} features × "
                  f"{R} restarts), {tree.total_params_per_tree} params/tree, "
                  f"strategy={cfg.strategy}")

        # ---- Strategy "random": skip training, snap, evaluate ----
        if cfg.strategy == "random":
            tree.temp_inv.fill_(1.0)
            tree.snap()
            tree.selection_mode = "softmax"
            tree.training = False
            return self._finalize(tree, x_exp, y_exp, F, R, V, var_names,
                                  cfg, t0)

        # ---- Phase 1: Fitting ----
        tree.selection_mode = cfg.fit_selection
        tree.train()
        self._log(f"Phase 1  Fitting  ({cfg.fit_steps} steps, lr={cfg.fit_lr}, "
                  f"selection={cfg.fit_selection})")
        opt = torch.optim.Adam(tree.parameters(), lr=cfg.fit_lr)
        nan_ct = 0

        for step in range(cfg.fit_steps):
            opt.zero_grad()
            pred = tree(x_exp)
            loss = self._mse(pred, y_exp)

            if not torch.isfinite(loss):
                nan_ct += 1
                self._nan_recover(tree)
                if nan_ct > 200:
                    self._log("  aborting Phase 1 — too many NaNs")
                    break
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(tree.parameters(), cfg.grad_clip)
            opt.step()

            if (step + 1) % cfg.log_every == 0:
                self._log(f"  step {step+1:>5}/{cfg.fit_steps}  "
                          f"loss={loss.item():.4e}  nan_recover={nan_ct}")

        # ---- Phase 2: Hardening ----
        tree.selection_mode = cfg.harden_selection
        self._log(f"Phase 2  Hardening  ({cfg.harden_steps} steps, "
                  f"selection={cfg.harden_selection})")
        opt = torch.optim.Adam(tree.parameters(), lr=cfg.harden_lr)

        for step in range(cfg.harden_steps):
            progress = step / max(cfg.harden_steps - 1, 1)
            tree.temp_inv.fill_(1.0 + progress * 20.0)

            opt.zero_grad()
            pred = tree(x_exp)
            recon = self._mse(pred, y_exp)

            ent_w = cfg.entropy_weight_final * progress
            ent = tree.entropy() * ent_w

            loss = recon + ent

            if not torch.isfinite(loss):
                self._nan_recover(tree)
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(tree.parameters(), cfg.grad_clip)
            opt.step()

            if (step + 1) % cfg.log_every == 0:
                self._log(f"  step {step+1:>5}/{cfg.harden_steps}  "
                          f"recon={recon.item():.4e}  ent={ent.item():.4e}  "
                          f"temp_inv={tree.temp_inv.item():.1f}")

        # ---- Phase 3: Snap & verify ----
        self._log("Phase 3  Snap to argmax")
        tree.temp_inv.fill_(1.0)
        tree.snap()
        tree.selection_mode = "softmax"   # deterministic for verification
        tree.training = False

        return self._finalize(tree, x_exp, y_exp, F, R, V, var_names, cfg, t0)

    def _finalize(self, tree, x_exp, y_exp, F, R, V, var_names, cfg, t0):
        """Evaluate snapped tree, pick best restart per feature, return result."""
        with torch.no_grad():
            pred = tree(x_exp)
            per_tree = self._per_tree_mse(pred, y_exp)  # (B,)

        mse_mat = per_tree.reshape(F, R)
        best_r = mse_mat.argmin(dim=1)                          # (F,)
        arange_f = torch.arange(F, device=cfg.device)
        best_idx = arange_f * R + best_r                        # (F,)
        best_mse = mse_mat[arange_f, best_r]                    # (F,)

        snapped_ok = best_mse < cfg.snap_mse_threshold

        self._log(f"  {snapped_ok.sum().item()}/{F} features below "
                  f"MSE threshold ({cfg.snap_mse_threshold:.0e})")
        self._log(f"  MSE range: [{best_mse.min().item():.2e}, "
                  f"{best_mse.max().item():.2e}]")

        if var_names is None:
            var_names = [f"x{i}" for i in range(V)] if V > 1 else ["x"]

        expressions = extract_expressions(tree, best_idx.tolist(), var_names)

        total_t = time.time() - t0
        self._log(f"Done in {total_t:.1f}s")

        return EMLResult(
            expressions=expressions,
            mse_values=best_mse.cpu(),
            snapped_ok=snapped_ok.cpu(),
            best_restart=best_r.cpu(),
            tree=tree,
            total_time_s=total_t,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        if diff.is_complex():
            return diff.abs().pow(2).mean()
        return diff.pow(2).mean()

    @staticmethod
    def _per_tree_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        if diff.is_complex():
            mse = diff.abs().pow(2).mean(dim=-1)
        else:
            mse = diff.pow(2).mean(dim=-1)
        # NaN trees get a large MSE so they're never selected as best
        return torch.where(torch.isfinite(mse), mse,
                           torch.tensor(1e20, device=mse.device, dtype=mse.dtype))

    @staticmethod
    def _nan_recover(tree: BatchedEMLTree):
        """Scale all parameters by 0.5 to escape NaN basin."""
        with torch.no_grad():
            for p in tree.parameters():
                p.mul_(0.5)

    @staticmethod
    def _log(msg: str):
        print(f"[EML] {msg}")
