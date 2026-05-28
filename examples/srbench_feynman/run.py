"""
H33 — EML vs Polynomial vs PySR on Feynman equations subset
============================================================
Benchmark comparing emltorch symbolic regression against polynomial baselines
and PySR on a subset of AI-Feynman / SRBench equations with 1–3 variables.

Usage (from this directory, with emltorch pip-installed or available on
PYTHONPATH; PySR must be installed for the PySR row — see pip extras `pysr`):

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        python3 -u run.py

Output: results.json next to this script.
"""

import sys
import os
import time
import json
import traceback
import warnings
import numpy as np

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "results.json")

# Allow running from a fresh git clone without `pip install -e .` by adding
# the repo root (two levels up) to sys.path.
_REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import emltorch as eml

print(f"emltorch version: {eml.__version__}", flush=True)

from pysr import PySRRegressor
import pysr

print(f"PySR version: {pysr.__version__}", flush=True)

# ─── Equation definitions ─────────────────────────────────────────────────────
# (name, variables, sample_fn, gt_fn)
# sample_fn(rng) -> dict of variable arrays (N=300)
# gt_fn(**vars) -> y array
N_SAMPLES = 300


def make_equations():
    equations = []

    # I.6.20a: exp(-theta**2/2), theta∈[1,3]
    equations.append(
        {
            "name": "I.6.20a",
            "desc": "exp(-theta^2/2)",
            "n_vars": 1,
            "sample": lambda rng: {"theta": rng.uniform(1, 3, N_SAMPLES)},
            "gt": lambda theta: np.exp(-(theta**2) / 2),
        }
    )

    # I.6.20: exp(-(theta/sigma)^2/2), theta∈[1,3], sigma∈[1,3]
    equations.append(
        {
            "name": "I.6.20",
            "desc": "exp(-(theta/sigma)^2/2)",
            "n_vars": 2,
            "sample": lambda rng: {
                "theta": rng.uniform(1, 3, N_SAMPLES),
                "sigma": rng.uniform(1, 3, N_SAMPLES),
            },
            "gt": lambda theta, sigma: np.exp(-((theta / sigma) ** 2) / 2),
        }
    )

    # I.12.5: q2*Ef (product), q2∈[1,5], Ef∈[1,5]
    equations.append(
        {
            "name": "I.12.5",
            "desc": "q2*Ef",
            "n_vars": 2,
            "sample": lambda rng: {
                "q2": rng.uniform(1, 5, N_SAMPLES),
                "Ef": rng.uniform(1, 5, N_SAMPLES),
            },
            "gt": lambda q2, Ef: q2 * Ef,
        }
    )

    # I.25.13: q/C (ratio), q∈[1,5], C∈[1,5]
    equations.append(
        {
            "name": "I.25.13",
            "desc": "q/C",
            "n_vars": 2,
            "sample": lambda rng: {
                "q": rng.uniform(1, 5, N_SAMPLES),
                "C": rng.uniform(1, 5, N_SAMPLES),
            },
            "gt": lambda q, C: q / C,
        }
    )

    # I.14.3: m*g*z (3 var product), m,g,z∈[1,5]
    equations.append(
        {
            "name": "I.14.3",
            "desc": "m*g*z",
            "n_vars": 3,
            "sample": lambda rng: {
                "m": rng.uniform(1, 5, N_SAMPLES),
                "g": rng.uniform(1, 5, N_SAMPLES),
                "z": rng.uniform(1, 5, N_SAMPLES),
            },
            "gt": lambda m, g, z: m * g * z,
        }
    )

    # II.3.24: Pwr/(4*pi*r^2), Pwr∈[1,5], r∈[1,5]
    equations.append(
        {
            "name": "II.3.24",
            "desc": "Pwr/(4*pi*r^2)",
            "n_vars": 2,
            "sample": lambda rng: {
                "Pwr": rng.uniform(1, 5, N_SAMPLES),
                "r": rng.uniform(1, 5, N_SAMPLES),
            },
            "gt": lambda Pwr, r: Pwr / (4 * np.pi * r**2),
        }
    )

    # I.27.6: 1/(1/d1 + n/d2), d1,d2∈[1,5], n∈[1,5]
    equations.append(
        {
            "name": "I.27.6",
            "desc": "1/(1/d1 + n/d2)",
            "n_vars": 3,
            "sample": lambda rng: {
                "d1": rng.uniform(1, 5, N_SAMPLES),
                "d2": rng.uniform(1, 5, N_SAMPLES),
                "n": rng.uniform(1, 5, N_SAMPLES),
            },
            "gt": lambda d1, d2, n: 1.0 / (1.0 / d1 + n / d2),
        }
    )

    # I.26.2: arcsin(n*sin(theta2)), n∈[0,1], theta2∈[1,5]
    # n*sin(theta2) can exceed 1 → need to guard domain
    # Only include if domain check passes on sampled data
    equations.append(
        {
            "name": "I.26.2",
            "desc": "arcsin(n*sin(theta2))",
            "n_vars": 2,
            "sample": lambda rng: {
                "n": rng.uniform(0, 1, N_SAMPLES),
                "theta2": rng.uniform(1, 5, N_SAMPLES),
            },
            "gt": lambda n, theta2: np.arcsin(
                np.clip(n * np.sin(theta2), -1 + 1e-7, 1 - 1e-7)
            ),
            "domain_check": lambda vars_: np.all(
                np.abs(vars_["n"] * np.sin(vars_["theta2"])) <= 1.0
            ),
        }
    )

    return equations


# ─── Polynomial feature builders ──────────────────────────────────────────────
def poly_k2_features(X):
    """
    Degree-2 polynomial features: constant + {x_i} + {x_i^2} + {x_i*x_j for i<j}.
    For V variables: 1 + V + V + C(V,2) = 1 + 2V + V*(V-1)/2 features.
    """
    n, v = X.shape
    cols = [np.ones((n, 1))]
    # linear terms
    cols.append(X)
    # squared terms
    cols.append(X**2)
    # pairwise products
    for i in range(v):
        for j in range(i + 1, v):
            cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.hstack(cols)


def poly_k5_features(X):
    """
    Degree-5-ish polynomial features: constant + per-feature powers {x_i^k, k=1..5}
    + pairwise products {x_i*x_j for i<j} + degree-2 cross terms only.
    This is bounded (avoids exponential blow-up with V=3 full degree-5 cross terms).
    Specifically: 1 + 5*V + C(V,2) features.
    """
    n, v = X.shape
    cols = [np.ones((n, 1))]
    # per-feature powers k=1..5
    for i in range(v):
        for k in range(1, 6):
            cols.append((X[:, i] ** k).reshape(-1, 1))
    # pairwise products (degree-2 cross terms)
    for i in range(v):
        for j in range(i + 1, v):
            cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.hstack(cols)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 1.0 if ss_res < 1e-15 else 0.0
    return float(1 - ss_res / ss_tot)


def fit_poly(X_train, y_train, X_test, y_test, feat_fn, name):
    """Fit polynomial baseline via OLS, return (heldout_r2, n_terms, fit_time_s)."""
    t0 = time.time()
    Phi_train = feat_fn(X_train)
    Phi_test = feat_fn(X_test)
    # Clip extreme values to avoid overflow / numerical blow-up
    np.clip(Phi_train, -1e6, 1e6, out=Phi_train)
    np.clip(Phi_test, -1e6, 1e6, out=Phi_test)
    coeffs, _, _, _ = np.linalg.lstsq(Phi_train, y_train, rcond=None)
    y_pred = Phi_test @ coeffs
    t1 = time.time()
    r2 = r2_score(y_test, y_pred)
    n_terms = Phi_train.shape[1]
    return r2, n_terms, t1 - t0


def fit_eml(X_train, y_train, X_test, y_test, depth, label):
    """Fit emltorch EML at given depth, return (heldout_r2, complexity, fit_time_s, expression)."""
    t0 = time.time()
    res = eml.fit(X_train, y_train, depth=depth, device="cpu")
    t1 = time.time()
    # predict on test
    import torch

    if X_test.shape[1] == 1:
        x_input = torch.tensor(X_test[:, 0], dtype=torch.float32)
    else:
        x_input = torch.tensor(X_test, dtype=torch.float32)
    preds = res.predict(x_input)
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
    else:
        preds = np.array(preds)
    r2 = r2_score(y_test, preds)
    complexity = res.expression.count("eml(")
    return r2, complexity, t1 - t0, res.expression


# ─── PySR setup ───────────────────────────────────────────────────────────────
PYSR_KWARGS = dict(
    niterations=30,
    populations=8,
    population_size=30,
    maxsize=20,
    deterministic=True,
    parallelism="serial",
    random_state=0,
    progress=False,
    verbosity=0,
    temp_equation_file=True,
)


def fit_pysr(X_train, y_train, X_test, y_test):
    """Fit PySR, return (heldout_r2, complexity, fit_time_s)."""
    t0 = time.time()
    m = PySRRegressor(**PYSR_KWARGS)
    m.fit(X_train, y_train)
    t1 = time.time()
    preds = m.predict(X_test)
    r2 = r2_score(y_test, preds)
    best = m.get_best()
    complexity = int(best["complexity"])
    return r2, complexity, t1 - t0


# ─── PySR warmup (exclude Julia precompile from per-equation timing) ──────────
def pysr_warmup():
    print("PySR warmup (Julia precompile)...", flush=True)
    t0 = time.time()
    rng_w = np.random.default_rng(999)
    X_w = rng_w.uniform(1, 3, (50, 1)).astype(np.float32)
    y_w = np.exp(-X_w[:, 0] ** 2 / 2)
    m_w = PySRRegressor(**{**PYSR_KWARGS, "niterations": 3, "populations": 2})
    m_w.fit(X_w, y_w)
    elapsed = time.time() - t0
    print(f"  Warmup complete in {elapsed:.1f}s", flush=True)


# ─── Main benchmark loop ──────────────────────────────────────────────────────
def run_benchmark():
    results = {}

    # PySR warmup
    pysr_warmup()

    equations = make_equations()

    for eq in equations:
        name = eq["name"]
        print(f"\n{'='*60}", flush=True)
        print(f"Equation: {name} — {eq['desc']}", flush=True)

        # Sample data
        rng = np.random.default_rng(0)
        var_dict = eq["sample"](rng)

        # Domain check if present
        if "domain_check" in eq:
            if not eq["domain_check"](var_dict):
                print(
                    f"  WARNING: domain check FAILED for {name}, skipping", flush=True
                )
                results[name] = {"skipped": "domain_check_failed"}
                _save_results(results)
                continue

        # Build X, y
        var_names = list(var_dict.keys())
        X_all = np.column_stack([var_dict[v] for v in var_names]).astype(np.float64)
        y_all = eq["gt"](**var_dict).astype(np.float64)

        if not np.all(np.isfinite(y_all)):
            print(f"  WARNING: non-finite y values for {name}, skipping", flush=True)
            results[name] = {"skipped": "non_finite_y"}
            _save_results(results)
            continue

        # 75-25 train/test split
        rng_split = np.random.default_rng(42)
        idx = rng_split.permutation(N_SAMPLES)
        n_train = int(0.75 * N_SAMPLES)
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

        X_train_raw = X_all[train_idx]
        X_test_raw = X_all[test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        # Standardize X by TRAIN mean/std (applied to test too)
        X_mean = X_train_raw.mean(axis=0)
        X_std = X_train_raw.std(axis=0)
        X_std[X_std < 1e-10] = 1.0  # guard zero std
        X_train = (X_train_raw - X_mean) / X_std
        X_test = (X_test_raw - X_mean) / X_std

        eq_result = {
            "desc": eq["desc"],
            "n_vars": eq["n_vars"],
            "n_train": n_train,
            "n_test": len(test_idx),
            "methods": {},
        }

        # ── EML depth=3 ──────────────────────────────────────────────────────
        print("  Fitting eml_d3...", end=" ", flush=True)
        try:
            r2, cpx, t, expr = fit_eml(
                X_train, y_train, X_test, y_test, depth=3, label="eml_d3"
            )
            eq_result["methods"]["eml_d3"] = {
                "heldout_r2": r2,
                "complexity": cpx,
                "complexity_unit": "eml_operator_count",
                "fit_time_s": t,
                "expression": expr,
            }
            print(f"R²={r2:.4f}, time={t:.2f}s, eml_ops={cpx}", flush=True)
        except Exception as exc:
            eq_result["methods"]["eml_d3"] = {"error": str(exc)}
            print(f"ERROR: {exc}", flush=True)

        # ── EML depth=4 ──────────────────────────────────────────────────────
        print("  Fitting eml_d4...", end=" ", flush=True)
        try:
            r2, cpx, t, expr = fit_eml(
                X_train, y_train, X_test, y_test, depth=4, label="eml_d4"
            )
            eq_result["methods"]["eml_d4"] = {
                "heldout_r2": r2,
                "complexity": cpx,
                "complexity_unit": "eml_operator_count",
                "fit_time_s": t,
                "expression": expr,
            }
            print(f"R²={r2:.4f}, time={t:.2f}s, eml_ops={cpx}", flush=True)
        except Exception as exc:
            eq_result["methods"]["eml_d4"] = {"error": str(exc)}
            print(f"ERROR: {exc}", flush=True)

        # ── poly_K2 ───────────────────────────────────────────────────────────
        print("  Fitting poly_K2...", end=" ", flush=True)
        try:
            r2, n_terms, t = fit_poly(
                X_train, y_train, X_test, y_test, poly_k2_features, "poly_K2"
            )
            eq_result["methods"]["poly_K2"] = {
                "heldout_r2": r2,
                "complexity": n_terms,
                "complexity_unit": "ols_coefficient_count",
                "fit_time_s": t,
            }
            print(f"R²={r2:.4f}, time={t:.4f}s, terms={n_terms}", flush=True)
        except Exception as exc:
            eq_result["methods"]["poly_K2"] = {"error": str(exc)}
            print(f"ERROR: {exc}", flush=True)

        # ── poly_K5 ───────────────────────────────────────────────────────────
        # poly_K5: constant + per-feature powers k=1..5 + pairwise products (degree-2 cross terms)
        # 1 + 5*V + C(V,2) features — bounded, no degree-5 cross terms
        print("  Fitting poly_K5...", end=" ", flush=True)
        try:
            r2, n_terms, t = fit_poly(
                X_train, y_train, X_test, y_test, poly_k5_features, "poly_K5"
            )
            eq_result["methods"]["poly_K5"] = {
                "heldout_r2": r2,
                "complexity": n_terms,
                "complexity_unit": "ols_coefficient_count",
                "fit_time_s": t,
            }
            print(f"R²={r2:.4f}, time={t:.4f}s, terms={n_terms}", flush=True)
        except Exception as exc:
            eq_result["methods"]["poly_K5"] = {"error": str(exc)}
            print(f"ERROR: {exc}", flush=True)

        # ── PySR ──────────────────────────────────────────────────────────────
        # NOTE: PySR numbers are LIVE from this run (no cache from h32_* outputs)
        print("  Fitting PySR (niterations=30)...", end=" ", flush=True)
        try:
            r2, cpx, t = fit_pysr(X_train, y_train, X_test, y_test)
            eq_result["methods"]["pysr"] = {
                "heldout_r2": r2,
                "complexity": cpx,
                "complexity_unit": "pysr_node_count",
                "fit_time_s": t,
                "source": "live_run_this_script",
            }
            print(f"R²={r2:.4f}, time={t:.2f}s, nodes={cpx}", flush=True)
        except Exception as exc:
            eq_result["methods"]["pysr"] = {
                "heldout_r2": f"error: {exc}",
                "complexity": None,
                "fit_time_s": None,
                "source": "live_run_this_script",
            }
            print(f"ERROR: {exc}", flush=True)
            traceback.print_exc()

        results[name] = eq_result
        _save_results(results)  # incremental save after each equation
        print(f"  Saved to {OUTPUT_JSON}", flush=True)

    # ── Aggregate statistics ──────────────────────────────────────────────────
    methods_list = ["eml_d3", "eml_d4", "poly_K2", "poly_K5", "pysr"]
    agg = {}

    for method in methods_list:
        r2_vals = []
        time_vals = []
        for eq_name, eq_data in results.items():
            if "methods" not in eq_data:
                continue
            m_data = eq_data["methods"].get(method, {})
            if "error" in m_data:
                continue
            r2 = m_data.get("heldout_r2")
            t = m_data.get("fit_time_s")
            if isinstance(r2, (int, float)) and np.isfinite(r2):
                r2_vals.append(r2)
            if isinstance(t, (int, float)) and np.isfinite(t):
                time_vals.append(t)
        agg[method] = {
            "mean_heldout_r2": float(np.mean(r2_vals)) if r2_vals else None,
            "median_heldout_r2": float(np.median(r2_vals)) if r2_vals else None,
            "median_fit_time_s": float(np.median(time_vals)) if time_vals else None,
            "n_equations": len(r2_vals),
            "clean_recovery_count_r2gt0999": int(sum(1 for r in r2_vals if r > 0.999)),
        }

    # Win counts: per equation, which method has strictly best R²
    win_count = {m: 0 for m in methods_list}
    for eq_name, eq_data in results.items():
        if "methods" not in eq_data:
            continue
        best_r2 = -np.inf
        best_methods = []
        for method in methods_list:
            m_data = eq_data["methods"].get(method, {})
            r2 = m_data.get("heldout_r2")
            if isinstance(r2, (int, float)) and np.isfinite(r2):
                if r2 > best_r2 + 1e-6:
                    best_r2 = r2
                    best_methods = [method]
                elif abs(r2 - best_r2) <= 1e-6:
                    best_methods.append(method)
        for m in best_methods:
            win_count[m] += 1
    for method in methods_list:
        agg[method]["win_count"] = win_count[method]

    results["aggregate"] = agg
    _save_results(results)

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("AGGREGATE SUMMARY TABLE", flush=True)
    print("=" * 80, flush=True)
    print(
        f"{'Method':<12}  {'Mean R²':>10}  {'Median R²':>11}  "
        f"{'Med Time(s)':>12}  {'Wins':>6}  {'Clean(>0.999)':>14}",
        flush=True,
    )
    print("-" * 80, flush=True)
    for method in methods_list:
        a = agg[method]
        mean_r2 = (
            f"{a['mean_heldout_r2']:.4f}" if a["mean_heldout_r2"] is not None else "N/A"
        )
        med_r2 = (
            f"{a['median_heldout_r2']:.4f}"
            if a["median_heldout_r2"] is not None
            else "N/A"
        )
        med_t = (
            f"{a['median_fit_time_s']:.3f}"
            if a["median_fit_time_s"] is not None
            else "N/A"
        )
        wins = a["win_count"]
        clean = a["clean_recovery_count_r2gt0999"]
        print(
            f"{method:<12}  {mean_r2:>10}  {med_r2:>11}  {med_t:>12}  {wins:>6}  {clean:>14}",
            flush=True,
        )
    print("=" * 80, flush=True)
    print(
        "\nNOTE: Complexity units are NOT directly comparable across methods:\n"
        "  eml_d3/eml_d4 = count of eml() operators in expression\n"
        "  poly_K2/poly_K5 = number of OLS coefficient terms [1 + 2V + C(V,2)] or [1 + 5V + C(V,2)]\n"
        "  pysr = node count in selected best equation (all AST nodes)\n",
        flush=True,
    )
    print(f"Results saved to: {OUTPUT_JSON}", flush=True)


def _save_results(results):
    """Write results dict to JSON (full re-dump for crash safety)."""

    def _convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=_convert)


if __name__ == "__main__":
    run_benchmark()
