"""
Hybrid EML trainer: linear + symbolic nonlinear decomposition.

Pure EML cannot represent even the identity function at shallow depth
(x itself requires going through exp(ln(x))), so fitting linear data with
EML alone is pathological. Real symbolic regression tools (PySR, Eureqa)
always include a linear term.

This module wraps BatchedEMLTree in an affine hull:
    y = bias + linear_coeffs @ x + eml_scale * EML_tree(x)

Interpretation of the learned expression:
    - (bias, linear_coeffs) is the best linear predictor
    - eml_scale * EML_tree captures the nonlinear residual
    - If eml_scale -> 0, the signal is linear
    - If nonlinear_R2 > 0.3 on residuals, EML found real structure
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from .tree import BatchedEMLTree
from .symbolic import extract_expressions


@dataclass
class HybridConfig:
    depth: int = 3
    num_restarts: int = 24
    num_vars: int = 1
    dtype: str = "float32"
    device: str = "cuda:7"

    fit_steps: int = 5000
    fit_lr: float = 2e-2
    grad_clip: float = 1.0

    harden_steps: int = 2000
    harden_lr: float = 5e-3
    entropy_weight_final: float = 1e-2

    # Encourages eml_scale -> 0 when the signal is linear
    # so we don't report spurious symbolic structure
    eml_l1_weight: float = 1e-3

    log_every: int = 1000

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "float64": torch.float64}[self.dtype]


@dataclass
class HybridResult:
    expressions: list[str]       # "a + b*x1 + c*(eml_tree)" per feature
    linear_r2: torch.Tensor      # (F,) R2 explained by linear alone
    hybrid_r2: torch.Tensor      # (F,) R2 from full hybrid fit
    nonlinear_r2: torch.Tensor   # (F,) R2 on residuals that EML captured
    bias: torch.Tensor           # (F,)
    linear_coeffs: torch.Tensor  # (F, V)
    eml_scale: torch.Tensor      # (F,)
    eml_expressions: list[str]   # just the EML sub-tree expression
    best_restart: torch.Tensor   # (F,)
    tree: BatchedEMLTree
    total_time_s: float


class HybridEMLTrainer:
    """Hybrid linear + EML fit. One-shot three-phase like EMLTrainer."""

    def __init__(self, cfg: HybridConfig):
        self.cfg = cfg

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        var_names: list[str] | None = None,
    ) -> HybridResult:
        cfg = self.cfg
        t0 = time.time()
        R = cfg.num_restarts

        if x.dim() == 2:
            x = x.unsqueeze(1)
        F, V, N = x.shape
        assert y.shape == (F, N)

        x_exp = x.repeat_interleave(R, dim=0).to(cfg.device).to(cfg.torch_dtype)
        y_exp = y.repeat_interleave(R, dim=0).to(cfg.device).to(cfg.torch_dtype)
        B = F * R

        # ---- Linear baseline: closed form for comparison AND warm start ----
        linear_r2, lin_init, bias_init = self._closed_form_linear(x, y)

        # ---- Build hybrid model ----
        tree = BatchedEMLTree(
            num_trees=B, depth=cfg.depth, num_vars=V,
            dtype=cfg.torch_dtype, device=cfg.device,
        )
        # Warm-start bias/lin_coef from closed-form — same across restarts
        bias_tensor = bias_init.to(cfg.device, cfg.torch_dtype).repeat_interleave(R)
        lin_tensor = lin_init.to(cfg.device, cfg.torch_dtype).repeat_interleave(R, dim=0)
        bias = nn.Parameter(bias_tensor.clone())
        lin_coef = nn.Parameter(lin_tensor.clone())
        eml_scale = nn.Parameter(torch.full((B,), 0.05, device=cfg.device, dtype=cfg.torch_dtype))

        params = list(tree.parameters()) + [bias, lin_coef, eml_scale]

        self._log(f"Hybrid tree: depth={cfg.depth}, {B} trees, {V} vars")
        self._log(f"Linear-only R2: mean={linear_r2.mean():.3f}")

        # ---- Phase 1: Fit ----
        self._log(f"Phase 1 Fitting ({cfg.fit_steps} steps)")
        opt = torch.optim.Adam(params, lr=cfg.fit_lr)
        nan_ct = 0
        for step in range(cfg.fit_steps):
            opt.zero_grad()
            pred = self._forward(tree, bias, lin_coef, eml_scale, x_exp)
            recon = (pred - y_exp).pow(2).mean()
            l1 = cfg.eml_l1_weight * eml_scale.abs().mean()
            loss = recon + l1

            if not torch.isfinite(loss):
                nan_ct += 1
                self._nan_recover(tree, bias, lin_coef, eml_scale)
                if nan_ct > 200:
                    self._log("  aborting — too many NaNs")
                    break
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

            if (step + 1) % cfg.log_every == 0:
                self._log(f"  step {step+1}/{cfg.fit_steps} recon={recon.item():.4e} "
                          f"|eml_scale|={eml_scale.abs().mean().item():.3f} nan={nan_ct}")

        # ---- Phase 2: Harden ----
        self._log(f"Phase 2 Hardening ({cfg.harden_steps} steps)")
        opt = torch.optim.Adam(params, lr=cfg.harden_lr)
        for step in range(cfg.harden_steps):
            prog = step / max(cfg.harden_steps - 1, 1)
            tree.temp_inv.fill_(1.0 + prog * 20.0)

            opt.zero_grad()
            pred = self._forward(tree, bias, lin_coef, eml_scale, x_exp)
            recon = (pred - y_exp).pow(2).mean()
            ent = tree.entropy() * (cfg.entropy_weight_final * prog)
            l1 = cfg.eml_l1_weight * eml_scale.abs().mean()
            loss = recon + ent + l1

            if not torch.isfinite(loss):
                self._nan_recover(tree, bias, lin_coef, eml_scale)
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

        # ---- Phase 3: Snap ----
        tree.temp_inv.fill_(1.0)
        tree.snap()
        with torch.no_grad():
            pred = self._forward(tree, bias, lin_coef, eml_scale, x_exp)
            per_tree = (pred - y_exp).pow(2).mean(dim=-1)
            per_tree = torch.where(torch.isfinite(per_tree), per_tree,
                                   torch.tensor(1e20, device=per_tree.device,
                                                dtype=per_tree.dtype))

        # Best restart per feature
        mse_mat = per_tree.reshape(F, R)
        best_r = mse_mat.argmin(dim=1)
        arange_f = torch.arange(F, device=cfg.device)
        best_idx = arange_f * R + best_r

        # Hybrid R2
        ss_tot = (y.to(cfg.device) - y.to(cfg.device).mean(dim=-1, keepdim=True)).pow(2).sum(dim=-1)
        hybrid_mse = mse_mat[arange_f, best_r]
        hybrid_ss_res = hybrid_mse * N
        hybrid_r2 = 1 - hybrid_ss_res / ss_tot.clamp(min=1e-8)

        # Nonlinear R2: how much of the residual did EML explain?
        nonlinear_r2 = (hybrid_r2 - linear_r2.to(cfg.device)).clamp(min=0.0)

        # Extract EML sub-expression
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(V)]
        eml_exprs = extract_expressions(tree, best_idx.tolist(), var_names)

        # Build full hybrid expression
        full_exprs = []
        for f in range(F):
            idx = best_idx[f].item()
            b_val = bias[idx].item()
            coefs = lin_coef[idx].tolist()
            s_val = eml_scale[idx].item()
            lin_str = " ".join(
                f"{c:+.3f}*{var_names[i]}" for i, c in enumerate(coefs)
            )
            parts = [f"{b_val:+.3f}"]
            if lin_str:
                parts.append(lin_str)
            parts.append(f"({s_val:+.3f}) * [{eml_exprs[f]}]")
            full_exprs.append(" ".join(parts).lstrip("+").strip())

        return HybridResult(
            expressions=full_exprs,
            linear_r2=linear_r2.cpu(),
            hybrid_r2=hybrid_r2.cpu(),
            nonlinear_r2=nonlinear_r2.cpu(),
            bias=bias[best_idx].detach().cpu(),
            linear_coeffs=lin_coef[best_idx].detach().cpu(),
            eml_scale=eml_scale[best_idx].detach().cpu(),
            eml_expressions=eml_exprs,
            best_restart=best_r.cpu(),
            tree=tree,
            total_time_s=time.time() - t0,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _forward(tree, bias, lin_coef, eml_scale, x):
        # x: (B, V, N)
        linear = bias[:, None] + (lin_coef[:, :, None] * x).sum(dim=1)   # (B, N)
        eml_out = tree(x)                                                # (B, N)
        return linear + eml_scale[:, None] * eml_out

    @staticmethod
    def _closed_form_linear(x, y):
        """Closed-form best affine fit per feature.

        Returns:
            r2:        (F,) linear-only R2
            lin_init:  (F, V) linear coefficients
            bias_init: (F,) intercepts
        """
        x_cpu = x.detach().cpu().double()
        y_cpu = y.detach().cpu().double()
        F_, V_, N_ = x_cpu.shape
        r2 = torch.zeros(F_)
        lin_init = torch.zeros(F_, V_)
        bias_init = torch.zeros(F_)
        for f in range(F_):
            Xf = x_cpu[f].T                                    # (N, V)
            Xf_aug = torch.cat([Xf, torch.ones(N_, 1, dtype=torch.float64)], dim=1)
            yf = y_cpu[f]
            sol = torch.linalg.lstsq(Xf_aug, yf.unsqueeze(1)).solution.squeeze(1)
            lin_init[f] = sol[:V_].float()
            bias_init[f] = sol[V_].float()
            pred = (Xf_aug @ sol.unsqueeze(1)).squeeze(1)
            ss_res = (yf - pred).pow(2).sum()
            ss_tot = (yf - yf.mean()).pow(2).sum().clamp(min=1e-12)
            r2[f] = float(1 - ss_res / ss_tot)
        return r2, lin_init, bias_init

    @staticmethod
    def _nan_recover(tree, bias, lin_coef, eml_scale):
        """Reset only NaN-prone parameters; preserve the linear fit."""
        with torch.no_grad():
            for p in tree.parameters():
                p.mul_(0.5)
            # Zero the EML contribution while keeping the linear predictor intact
            eml_scale.mul_(0.1)
            # Replace any NaN in linear params with zero (shouldn't happen but be safe)
            for p in (bias, lin_coef):
                torch.nan_to_num_(p, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _log(msg):
        print(f"[Hybrid] {msg}")
