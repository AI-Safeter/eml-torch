"""``eml-vacuity-audit`` console entry point.

Runs the four-check vacuity audit (shift-invariance, non-vacuity control,
relative-vs-absolute mass floor, numerical-precision floor) on a concentration
cert claim and prints the structured verdict. Two input modes:

- ``--scores-json FILE``: a JSON file with ``{"scores": [...], "target_idx": N}``
  (optionally ``exclude_from_sum``, ``layer``, ``head``; or a list of such rows
  to audit a whole atlas).
- ``--model M --prompt P --layer L --head H``: extract the head's log-prob
  attention row from a HuggingFace causal LM and audit it.

Examples:
    eml-vacuity-audit --scores-json head.json --tau 0.95 --form v3
    eml-vacuity-audit --model gpt2 --prompt "..." --layer 5 --head 5 --tau 0.5 \\
        --form softmax_interval --precision-floor 0.0039
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional, Sequence

from .vacuity_audit import vacuity_audit, audit_attention_atlas, SOUND


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eml-vacuity-audit",
        description="Test whether an attention-concentration / softmax-mass / "
        "routing cert claim is SOUND or VACUOUS / RELATIVE-ONLY / UNDER-PRECISION.",
    )
    src = p.add_argument_group("input (one of)")
    src.add_argument(
        "--scores-json",
        default=None,
        help="JSON file: {scores, target_idx[, exclude_from_sum, layer, head]} "
        "or a list of such rows",
    )
    src.add_argument("--model", default=None, help="HF model name (with --prompt etc.)")
    src.add_argument("--prompt", default=None, help="prompt to run the model on")
    src.add_argument("--layer", type=int, default=None, help="layer (model mode)")
    src.add_argument("--head", type=int, default=None, help="head (model mode)")
    src.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    src.add_argument(
        "--query-pos", type=int, default=-1, help="query position (default: -1)"
    )

    p.add_argument("--tau", type=float, default=0.95, help="concentration threshold")
    p.add_argument(
        "--form",
        default="softmax_interval",
        choices=["softmax_interval", "v3", "v2", "interval", "raw_weight"],
        help="cert form to audit (default: softmax_interval)",
    )
    p.add_argument(
        "--exclude-from-sum",
        default=None,
        help="comma-separated key indices the cert drops from its Sum (e.g. 0,last)",
    )
    p.add_argument(
        "--precision-floor",
        type=float,
        default=None,
        help="require certified radius above this (e.g. 0.0039 for bf16)",
    )
    p.add_argument(
        "--control",
        default="uniform",
        choices=["uniform", "lowmass"],
        help="non-vacuity control row (default: uniform)",
    )
    p.add_argument("--rho", type=float, default=0.005, help="L_inf box for checks 1-3")
    p.add_argument("--json", action="store_true", help="emit JSON instead of text")
    return p


def _parse_exclude(s: Optional[str], n: int) -> Optional[list[int]]:
    if not s:
        return None
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(n - 1 if tok == "last" else int(tok))
    return out or None


def _load_rows_from_json(path: str) -> list[dict]:
    with open(path) as f:
        blob = json.load(f)
    if isinstance(blob, dict):
        return [blob]
    if isinstance(blob, list):
        return blob
    raise ValueError(f"{path}: expected an object or a list of objects")


def _extract_model_row(args) -> dict:
    from .extract import extract_head_logprob_scores

    import transformers
    import torch

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager", dtype=torch.float32
    ).eval()
    if args.device != "cpu":
        model = model.to(args.device)
    hs = extract_head_logprob_scores(
        model,
        tok,
        args.prompt,
        args.layer,
        args.head,
        device=args.device,
        query_pos=args.query_pos,
    )
    return {
        "scores": list(hs.scores),
        "target_idx": hs.argmax_idx,
        "layer": args.layer,
        "head": args.head,
    }


def _audit_rows(rows: Sequence[dict], args) -> list:
    reports = []
    for row in rows:
        n = len(row["scores"])
        exclude = row.get("exclude_from_sum")
        if exclude is None:
            exclude = _parse_exclude(args.exclude_from_sum, n)
        report = vacuity_audit(
            row["scores"],
            int(row["target_idx"]),
            args.tau,
            form=args.form,
            exclude_from_sum=exclude,
            precision_floor=args.precision_floor,
            control=args.control,
            rho=args.rho,
        )
        report.layer = row.get("layer")
        report.head = row.get("head")
        reports.append(report)
    return reports


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.scores_json:
        rows = _load_rows_from_json(args.scores_json)
    elif (
        args.model
        and args.prompt is not None
        and args.layer is not None
        and args.head is not None
    ):
        rows = [_extract_model_row(args)]
    else:
        print(
            "error: supply --scores-json, OR --model/--prompt/--layer/--head",
            file=sys.stderr,
        )
        return 2

    reports = _audit_rows(rows, args)

    if args.json:
        print(json.dumps([r.to_json() for r in reports], indent=2))
        return 0 if all(r.verdict == SOUND for r in reports) else 1

    n_sound = 0
    for r in reports:
        head_tag = ""
        if r.layer is not None and r.head is not None:
            head_tag = f"L{r.layer}.H{r.head}  "
        print(f"{head_tag}tau={args.tau:g} form={args.form}")
        print(r.explanation)
        print("-" * 72)
        if r.verdict == SOUND:
            n_sound += 1
    if len(reports) > 1:
        print(f"summary: {n_sound}/{len(reports)} heads SOUND")
    return 0 if n_sound == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
