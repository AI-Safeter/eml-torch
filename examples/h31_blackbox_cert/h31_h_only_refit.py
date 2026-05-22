#!/usr/bin/env python3
"""H31 single-feature refit: EML on entropy_top50 alone (drop L).

Walk-back driver: original formula uses (L, H) features but L=0 across
all factual prompts, making the formula collapse to affine-of-linear-in-H.
Linear regression P = a + b·H achieves the same HELDOUT R² (Δ=0.0000).

Question: does EML find a nonlinear function of H alone that beats
linear regression at HELDOUT R²?

Output: outputs/h31_blackbox_cert/h_only_refit_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import emltorch  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs"


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def linear_r2(H_tr, y_tr, H_te, y_te) -> tuple[float, tuple[float, float]]:
    A_tr = np.stack([np.ones(len(H_tr)), H_tr], axis=1)
    A_te = np.stack([np.ones(len(H_te)), H_te], axis=1)
    w, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
    return r2_score(y_te, A_te @ w), (float(w[0]), float(w[1]))


def poly_r2(H_tr, y_tr, H_te, y_te, degree: int) -> float:
    Phi_tr = np.stack([H_tr**k for k in range(degree + 1)], axis=1)
    Phi_te = np.stack([H_te**k for k in range(degree + 1)], axis=1)
    cs = np.maximum(np.abs(Phi_tr).max(axis=0, keepdims=True), 1e-12)
    Phi_tr_n = Phi_tr / cs
    Phi_te_n = Phi_te / cs
    lam = 0.01 * max(1.0, Phi_tr.shape[1] / max(1, Phi_tr.shape[0]))
    reg = lam * np.eye(Phi_tr_n.shape[1])
    w = np.linalg.solve(Phi_tr_n.T @ Phi_tr_n + reg, Phi_tr_n.T @ y_tr)
    return r2_score(y_te, Phi_te_n @ w)


def eml_r2(H_tr, y_tr, H_te, y_te, depth: int, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        r = emltorch.fit(
            torch.tensor(H_tr.reshape(-1, 1), dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
            depth=depth,
            population=4096,
            generations=30,
            polish=True,
            normalize_inputs=True,
        )
        yp = r.predict(torch.tensor(H_te.reshape(-1, 1), dtype=torch.float32))
        if hasattr(yp, "cpu"):
            yp = yp.cpu().numpy()
        else:
            yp = np.asarray(yp)
        return r2_score(y_te, yp), r.expression
    except Exception as e:
        return float("-inf"), f"FAIL: {e}"


def main():
    with (OUT_DIR / "measurements_qwen36.jsonl").open() as f:
        rows = [json.loads(l) for l in f if json.loads(l)["circuit"] == "factual"]
    H = np.array([r["entropy_top50"] for r in rows])
    y = np.array([r["p_target"] for r in rows])

    results = {"n": len(rows), "feature": "entropy_top50 only", "splits": {}}

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(rows))
    n_te = max(2, int(len(rows) * 0.25))
    te = idx[:n_te]
    tr = idx[n_te:]

    H_tr, H_te = H[tr], H[te]
    y_tr, y_te = y[tr], y[te]

    split_results = {"linear": {}, "poly": {}, "eml": {}}

    r2_lin, (a, b) = linear_r2(H_tr, y_tr, H_te, y_te)
    split_results["linear"] = {"r2": r2_lin, "intercept": a, "slope": b}
    print(f"Linear  P = {a:.4f} + ({b:+.4f})·H: HELDOUT R² = {r2_lin:.4f}")

    for k in [2, 3, 5]:
        r2 = poly_r2(H_tr, y_tr, H_te, y_te, k)
        split_results["poly"][f"K={k}"] = r2
        print(f"Poly K={k}: HELDOUT R² = {r2:.4f}")

    for depth in [3, 4]:
        seeds_r2 = []
        seeds_expr = []
        for seed in range(10):
            r2, expr = eml_r2(H_tr, y_tr, H_te, y_te, depth, seed)
            seeds_r2.append(r2)
            seeds_expr.append(expr)
        valid = [x for x in seeds_r2 if x > -1e6]
        if valid:
            best_idx = int(np.argmax(seeds_r2))
            split_results["eml"][f"d{depth}"] = {
                "best": float(seeds_r2[best_idx]),
                "median": float(sorted(valid)[len(valid) // 2]),
                "mean": float(np.mean(valid)),
                "n_seeds": len(seeds_r2),
                "best_expr": seeds_expr[best_idx],
            }
            print(
                f"EML d={depth}: best HELDOUT R² = {seeds_r2[best_idx]:.4f} "
                f"(median over {len(valid)}/10 seeds = {sorted(valid)[len(valid) // 2]:.4f})"
            )
            print(f"  best_expr: {seeds_expr[best_idx]}")
    results["splits"]["random_seed42"] = split_results

    # Verdict
    print()
    print("=== VERDICT ===")
    lin_r2 = r2_lin
    eml_best = max(
        (split_results["eml"][k]["best"] for k in split_results["eml"]),
        default=float("-inf"),
    )
    eml_median = max(
        (split_results["eml"][k]["median"] for k in split_results["eml"]),
        default=float("-inf"),
    )
    print(f"Linear-in-H HELDOUT R²: {lin_r2:.4f}")
    print(f"EML best HELDOUT R² (10 seeds × depths 3,4): {eml_best:.4f}")
    print(f"EML median HELDOUT R²: {eml_median:.4f}")
    print(f"Δ (EML best − linear) = {eml_best - lin_r2:+.4f}")
    print(f"Δ (EML median − linear) = {eml_median - lin_r2:+.4f}")
    if eml_best > lin_r2 + 0.02 and eml_median > lin_r2:
        print(">>> EML wins nontrivially. Re-headline the formula.")
    elif eml_median <= lin_r2:
        print(">>> EML does not beat linear regression at the median. WALK BACK.")
    else:
        print(">>> EML wins at best-seed but not median. Inconclusive.")

    out_path = OUT_DIR / "h_only_refit_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
