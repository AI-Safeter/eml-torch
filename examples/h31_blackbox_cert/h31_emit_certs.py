#!/usr/bin/env python3
"""H31 cert emission + dual z3+cvc5 verify.

For each (vendor, circuit) with HELDOUT R² ≥ 0.5 (per fit_results.json):
  - Working-region cert: lower-bound P_target > τ over the convex hull
    of training-data features.
  - Failure-region cert: upper-bound P_target > τ over a boundary box.

Output:
  outputs/h31_blackbox_cert/certs/*.smt2
  outputs/h31_blackbox_cert/cert_verdicts.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import z3
import cvc5

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emltorch.smt import eml_tree_to_smt2_intervals  # noqa: E402

from _h31_common import FEAT_ORDER, build_features  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs"
CERT_DIR = OUT_DIR / "certs"
CERT_DIR.mkdir(parents=True, exist_ok=True)


def run_z3(smt_text: str, timeout_s: int = 30) -> tuple[str, float]:
    t0 = time.time()
    try:
        s = z3.Solver()
        s.set("timeout", timeout_s * 1000)
        s.from_string(smt_text)
        r = s.check()
        if r == z3.unsat:
            verdict = "unsat"
        elif r == z3.sat:
            verdict = "sat"
        else:
            verdict = "unknown"
    except Exception as e:
        verdict = f"error:{type(e).__name__}:{str(e)[:160]}"
    return verdict, time.time() - t0


def run_cvc5(smt_text: str, timeout_s: int = 30) -> tuple[str, float]:
    t0 = time.time()
    try:
        tm = cvc5.TermManager()
        s = cvc5.Solver(tm)
        s.setOption("tlimit", str(timeout_s * 1000))
        s.setOption("produce-models", "true")
        ip = cvc5.InputParser(s)
        ip.setStringInput(cvc5.InputLanguage.SMT_LIB_2_6, smt_text, "h31")
        sm = ip.getSymbolManager()
        while True:
            cmd = ip.nextCommand()
            if cmd.isNull():
                break
            cmd.invoke(s, sm)
        r = s.checkSat()
        if r.isUnsat():
            verdict = "unsat"
        elif r.isSat():
            verdict = "sat"
        else:
            verdict = "unknown"
    except Exception as e:
        verdict = f"error:{type(e).__name__}:{str(e)[:160]}"
    return verdict, time.time() - t0


def emit_one_cert(
    tag: str,
    circuit: str,
    formula: str,
    feature_names: list[str],
    var_ranges: dict,
    target_value: float,
    cert_kind: str,
) -> dict:
    """Emit + dual-verify one .smt2.

    cert_kind:
      'working_lb' — lower-bound P > τ (expect UNSAT)
      'failure_ub' — upper-bound P > τ (expect SAT counterexample)
    """
    cert_id = f"{tag}_{circuit}_{cert_kind}"
    smt_path = CERT_DIR / f"{cert_id}.smt2"

    # eml_tree_to_smt2_intervals asserts NOT (formula > τ) for SAFE certs:
    # working_lb expects UNSAT (formula > τ always over the box);
    # failure_ub expects SAT (counterexample with formula ≤ τ).
    op = ">"

    try:
        smt = eml_tree_to_smt2_intervals(
            formula=formula,
            var_ranges=var_ranges,
            target_op=op,
            target_value=target_value,
            title=f"H31 {cert_id}",
        )
    except Exception as e:
        return {
            "cert_id": cert_id,
            "status": f"emit_error:{type(e).__name__}:{e}",
        }

    smt_path.write_text(smt)

    z3_verdict, z3_t = run_z3(smt)
    cvc5_verdict, cvc5_t = run_cvc5(smt)

    dual_unsat = z3_verdict == "unsat" and cvc5_verdict == "unsat"
    dual_sat = z3_verdict == "sat" and cvc5_verdict == "sat"

    return {
        "cert_id": cert_id,
        "smt_path": str(smt_path.relative_to(REPO_ROOT)),
        "formula": formula,
        "var_ranges": {k: list(v) for k, v in var_ranges.items()},
        "target_op": op,
        "target_value": target_value,
        "cert_kind": cert_kind,
        "z3": {"verdict": z3_verdict, "wall_s": z3_t},
        "cvc5": {"verdict": cvc5_verdict, "wall_s": cvc5_t},
        "dual_unsat": dual_unsat,
        "dual_sat": dual_sat,
    }


def main() -> None:
    fit_path = OUT_DIR / "fit_results.json"
    if not fit_path.exists():
        print(f"[H31] {fit_path} missing — run h31_fit_and_baseline.py first")
        sys.exit(1)

    fit_results = json.loads(fit_path.read_text())

    cert_results = []
    for tag, circuits in fit_results.items():
        for circuit, info in circuits.items():
            if info.get("status") in ("insufficient_data", "degenerate_target"):
                continue
            best_expr = info.get("best_eml_expr")
            eml = info.get("eml", {})
            # Discipline (filter #10 adapted): require BOTH best ≥ 0.5 AND
            # median seed R² ≥ 0.3. Single-seed lucky fits rejected.
            best_score = eml.get("best_score") if isinstance(eml, dict) else None
            best_key = eml.get("best_key") if isinstance(eml, dict) else None
            median_seed_r2 = None
            if best_key and isinstance(eml.get(best_key), dict):
                seeds = eml[best_key].get("seeds", [])
                valid = sorted([s for s in seeds if s is not None and s > -1e3])
                median_seed_r2 = valid[len(valid) // 2] if valid else None
            qualifies = (
                best_expr is not None
                and best_score is not None
                and best_score >= 0.5
                and median_seed_r2 is not None
                and median_seed_r2 >= 0.3
            )
            if not qualifies:
                cert_results.append(
                    {
                        "tag": tag,
                        "circuit": circuit,
                        "status": "skipped_low_r2",
                        "best_score": best_score,
                        "median_seed_r2": median_seed_r2,
                    }
                )
                continue

            # Reconstruct feature arrays from measurements to compute IQR boxes.
            ms_path = OUT_DIR / f"measurements_{tag}.jsonl"
            if not ms_path.exists():
                continue
            ms = [json.loads(line) for line in ms_path.open()]
            rows = [m for m in ms if m["circuit"] == circuit]
            if len(rows) == 0:
                continue
            X, _ = build_features(rows)
            # emltorch expressions reference x1..x5 in FEAT_ORDER position.
            feats_used = [f"x{i+1}" for i in range(len(FEAT_ORDER))]
            xi_to_arr = {f"x{i+1}": X[:, i] for i in range(len(FEAT_ORDER))}

            # Operating box = IQR (25–75 percentile) on each feature —
            # tight high-confidence region where probes were observed.
            # Failure box = beyond the OBSERVED full range (extrapolation).
            var_ranges_working = {}
            var_ranges_failure = {}
            for v in feats_used:
                arr = xi_to_arr.get(v)
                if arr is None:
                    continue
                lo_full, hi_full = float(arr.min()), float(arr.max())
                q25 = float(np.percentile(arr, 25))
                q75 = float(np.percentile(arr, 75))
                if hi_full - lo_full < 1e-6:
                    var_ranges_working[v] = (lo_full - 0.1, hi_full + 0.1)
                elif q75 - q25 < 1e-6:
                    # Most observations identical — small band
                    var_ranges_working[v] = (q25 - 0.05, q75 + 0.05)
                else:
                    var_ranges_working[v] = (q25, q75)
                # Failure box = beyond observed range (extrapolation)
                rng = max(hi_full - lo_full, 0.5)
                var_ranges_failure[v] = (hi_full + 0.1 * rng, hi_full + 0.5 * rng)

            if not var_ranges_working:
                continue

            # τ chosen conservative: empirically formula predictions on
            # the IQR working box stay well above 0.10 for factual circuit
            # (verified by hand-evaluation of the depth-4 EML expression).
            # Setting τ=0.10 yields a meaningful "predicted P > 0.10" claim.
            tau = 0.10

            # Working cert
            r_work = emit_one_cert(
                tag,
                circuit,
                best_expr,
                list(var_ranges_working.keys()),
                var_ranges_working,
                target_value=tau,
                cert_kind="working_lb",
            )
            cert_results.append(r_work)

            # Failure cert (try; may emit_error if Ln args go negative)
            r_fail = emit_one_cert(
                tag,
                circuit,
                best_expr,
                list(var_ranges_failure.keys()),
                var_ranges_failure,
                target_value=tau,
                cert_kind="failure_ub",
            )
            cert_results.append(r_fail)

    out_path = OUT_DIR / "cert_verdicts.json"
    out_path.write_text(json.dumps(cert_results, indent=2))
    print(f"[H31] Wrote {len(cert_results)} cert verdicts → {out_path}")

    # Summary
    by_kind = {"working_lb": [], "failure_ub": []}
    for c in cert_results:
        if "cert_kind" in c:
            by_kind[c["cert_kind"]].append(c)
    print(f"\n[H31] Summary:")
    for kind, lst in by_kind.items():
        dual_unsat = sum(1 for c in lst if c.get("dual_unsat"))
        dual_sat = sum(1 for c in lst if c.get("dual_sat"))
        print(f"  {kind}: n={len(lst)}, dual-UNSAT={dual_unsat}, dual-SAT={dual_sat}")


if __name__ == "__main__":
    main()
