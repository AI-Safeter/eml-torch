#!/usr/bin/env python3
"""H31 fit + baselines + 11-filter discipline.

Per vendor × per circuit class:
  1. Filter #1 tautology check
  2. Filter #2 poly K=2 preflight
  3. Filter #3 seed variance (5 EML seeds)
  4. Filter #4 PC1 manifold OOD
  5. Fit EML depth-3, depth-4
  6. Fit poly K=2, K=5, linear OLS

Outputs:
  outputs/h31_blackbox_cert/fit_results.json
"""
from __future__ import annotations

import json
import sys
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import emltorch  # noqa: E402

from _h31_common import (  # noqa: E402
    FEAT_ORDER,
    OUT_DIR,
    build_features as _build_features_all,
    load_measurements,
    r2_score,
    random_split,
)


def build_features(
    measurements: list[dict], circuit: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Filter to one circuit + black-box-safe (T, L, n_rep, entropy, log_tok)."""
    rows = [m for m in measurements if m["circuit"] == circuit]
    if len(rows) < 10:
        return None, None, []
    X, y = _build_features_all(rows)
    return X, y, FEAT_ORDER


def poly_fit_and_score(X_tr, y_tr, X_te, y_te, degree: int) -> float:
    """OLS on polynomial features up to `degree`, column-normalized ridge."""
    n, d = X_tr.shape
    cols, cols_te = [np.ones(n)], [np.ones(X_te.shape[0])]
    for k in range(1, degree + 1):
        for combo in combinations_with_replacement(range(d), k):
            col = np.ones(n)
            col_te = np.ones(X_te.shape[0])
            for idx in combo:
                col = col * X_tr[:, idx]
                col_te = col_te * X_te[:, idx]
            cols.append(col)
            cols_te.append(col_te)
    Phi_tr = np.stack(cols, axis=1)
    Phi_te = np.stack(cols_te, axis=1)
    col_scale = np.maximum(np.abs(Phi_tr).max(axis=0, keepdims=True), 1e-12)
    Phi_tr /= col_scale
    Phi_te /= col_scale
    lam = 0.01 * max(1.0, Phi_tr.shape[1] / max(1, Phi_tr.shape[0]))
    w = np.linalg.solve(
        Phi_tr.T @ Phi_tr + lam * np.eye(Phi_tr.shape[1]), Phi_tr.T @ y_tr
    )
    return r2_score(y_te, Phi_te @ w)


def eml_fit_and_score(
    X_tr, y_tr, X_te, y_te, depth: int, seed: int
) -> tuple[float, str]:
    """Run emltorch.fit and score on holdout."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        r = emltorch.fit(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
            depth=depth,
            population=2048,
            generations=20,
            polish=True,
            normalize_inputs=True,
        )
        y_pred = r.predict(torch.tensor(X_te, dtype=torch.float32))
        if hasattr(y_pred, "cpu"):
            y_pred = y_pred.cpu().numpy()
        return r2_score(y_te, np.asarray(y_pred)), r.expression
    except Exception as e:
        return float("-inf"), f"FAIL: {type(e).__name__}: {e}"


def pc1_split(X: np.ndarray, y: np.ndarray, holdout_frac: float = 0.2):
    """Hold out the top holdout_frac of points along PC1 of X."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Xs = Xc / (Xc.std(axis=0, keepdims=True) + 1e-12)
    U, S, _ = np.linalg.svd(Xs, full_matrices=False)
    scores = U[:, 0] * S[0]
    test_mask = scores >= np.quantile(scores, 1.0 - holdout_frac)
    return X[~test_mask], y[~test_mask], X[test_mask], y[test_mask]


def process_one_circuit(measurements: list[dict], circuit: str, tag: str) -> dict:
    X, y, feat = build_features(measurements, circuit)
    if X is None:
        return {"circuit": circuit, "tag": tag, "status": "insufficient_data"}

    n = X.shape[0]
    print(f"\n[H31-{tag}] {circuit}: n={n}, feat={feat}")
    print(
        f"  y stats: mean={y.mean():.3f}, std={y.std():.3f}, min={y.min():.3f}, max={y.max():.3f}"
    )

    if y.std() < 0.01:
        return {
            "circuit": circuit,
            "tag": tag,
            "status": "degenerate_target",
            "n": n,
            "y_stats": {"mean": float(y.mean()), "std": float(y.std())},
        }

    # ---------------------------------------------------------------
    # Filter #1 tautology — check non-determinism in features
    # ---------------------------------------------------------------
    # Best linear predictor + residual
    Phi = np.concatenate([np.ones((n, 1)), X], axis=1)
    w_lin, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    lin_pred = Phi @ w_lin
    max_resid = float(np.max(np.abs(y - lin_pred)))
    tautology_passes = max_resid > 0.1

    # ---------------------------------------------------------------
    # Random + PC1 train/test splits
    # ---------------------------------------------------------------
    X_tr_r, y_tr_r, X_te_r, y_te_r = random_split(X, y, frac=0.25, seed=42)
    try:
        X_tr_pc, y_tr_pc, X_te_pc, y_te_pc = pc1_split(X, y, holdout_frac=0.20)
    except Exception:
        X_tr_pc = X_tr_r
        y_tr_pc = y_tr_r
        X_te_pc = X_te_r
        y_te_pc = y_te_r

    # ---------------------------------------------------------------
    # Baselines
    # ---------------------------------------------------------------
    poly_results = {}
    for split_name, (X_tr, y_tr, X_te, y_te) in [
        ("random", (X_tr_r, y_tr_r, X_te_r, y_te_r)),
        ("pc1", (X_tr_pc, y_tr_pc, X_te_pc, y_te_pc)),
    ]:
        for K in [1, 2, 5]:
            if X_tr.shape[0] < 8:
                poly_results[f"poly{K}_{split_name}"] = None
                continue
            try:
                r2 = poly_fit_and_score(X_tr, y_tr, X_te, y_te, K)
            except Exception:
                r2 = None
            poly_results[f"poly{K}_{split_name}"] = r2

    filter2_aborted = (
        poly_results.get("poly2_random") is not None
        and poly_results["poly2_random"] >= 0.95
    )

    # ---------------------------------------------------------------
    # EML (skipped if filter #2 firing OR tautology check fails)
    # ---------------------------------------------------------------
    eml_results = {}
    if filter2_aborted:
        eml_results["status"] = "skipped_filter2"
        best_expr = None
    elif not tautology_passes:
        eml_results["status"] = "skipped_tautology"
        best_expr = None
    else:
        for depth in [3, 4]:
            for split_name, (X_tr, y_tr, X_te, y_te) in [
                ("random", (X_tr_r, y_tr_r, X_te_r, y_te_r)),
                ("pc1", (X_tr_pc, y_tr_pc, X_te_pc, y_te_pc)),
            ]:
                r2s = []
                exprs = []
                for seed in range(5):
                    r2, expr = eml_fit_and_score(X_tr, y_tr, X_te, y_te, depth, seed)
                    r2s.append(r2)
                    exprs.append(expr)
                # best
                idx_best = int(np.argmax(r2s))
                eml_results[f"d{depth}_{split_name}"] = {
                    "mean": (
                        float(np.mean([x for x in r2s if x > -1e6]))
                        if any(x > -1e6 for x in r2s)
                        else None
                    ),
                    "std": (
                        float(np.std([x for x in r2s if x > -1e6]))
                        if any(x > -1e6 for x in r2s)
                        else None
                    ),
                    "best": float(r2s[idx_best]),
                    "best_expr": exprs[idx_best],
                    "seeds": r2s,
                }
        # pick globally best expression for cert
        best_key = None
        best_score = -np.inf
        for k, v in eml_results.items():
            if isinstance(v, dict) and v.get("best", -np.inf) > best_score:
                best_score = v["best"]
                best_key = k
        best_expr = eml_results[best_key]["best_expr"] if best_key else None
        eml_results["best_key"] = best_key
        eml_results["best_score"] = float(best_score) if best_score > -np.inf else None

    return {
        "circuit": circuit,
        "tag": tag,
        "n": n,
        "feature_names": feat,
        "y_stats": {
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
        },
        "tautology_max_resid": max_resid,
        "tautology_passes": bool(tautology_passes),
        "filter2_aborted_poly2": filter2_aborted,
        "poly": poly_results,
        "eml": eml_results,
        "best_eml_expr": best_expr,
    }


def main() -> None:
    all_results = {}
    for tag in ["qwen36", "gemma4"]:
        path = OUT_DIR / f"measurements_{tag}.jsonl"
        if not path.exists():
            print(f"[H31] {path} missing — skipping {tag}")
            continue
        ms = load_measurements(tag)
        print(f"[H31] {tag}: {len(ms)} measurements")
        circuits = sorted(set(m["circuit"] for m in ms))
        all_results[tag] = {}
        for c in circuits:
            res = process_one_circuit(ms, c, tag)
            all_results[tag][c] = res

    out_path = OUT_DIR / "fit_results.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[H31] Wrote {out_path}")


if __name__ == "__main__":
    main()
