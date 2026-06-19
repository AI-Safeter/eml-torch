"""``eml-certify`` console entry point.

Two modes:

- Single-head (``--layer L --head H``): extract the head's log-prob attention
  row, print which token it attends to + its softmax mass, and report the
  certified concentration radius and per-rho dual-solver verdicts.
- Sweep (no ``--layer``/``--head``): run ``AttentionCertAtlas`` over all
  (layer, head), print the top-k heads by certified radius, and optionally
  write the full per-head atlas JSON to ``--out``.

All certs use the honest, shift-invariant ``softmax_interval`` form (a
non-concentrated head correctly gets radius 0.0).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from .atlas import AttentionCertAtlas, certified_radius
from .extract import extract_head_logprob_scores


def _parse_rhos(s: Optional[str]) -> Optional[tuple[float, ...]]:
    if not s:
        return None
    return tuple(float(x) for x in s.split(",") if x.strip())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eml-certify",
        description="Portable dual-solver certificates of attention-head "
        "concentration for any HuggingFace causal LM.",
    )
    p.add_argument("--model", default="gpt2", help="HF model name (default: gpt2)")
    p.add_argument("--prompt", required=True, help="prompt to run the model on")
    p.add_argument("--tau", type=float, default=0.5, help="concentration threshold")
    p.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    p.add_argument("--layer", type=int, default=None, help="single-head mode: layer")
    p.add_argument("--head", type=int, default=None, help="single-head mode: head")
    p.add_argument(
        "--query-pos", type=int, default=-1, help="query position (default: -1)"
    )
    p.add_argument("--out", default=None, help="sweep mode: write atlas JSON here")
    p.add_argument("--rhos", default=None, help="comma-separated rho ladder, high->low")
    p.add_argument("--topk", type=int, default=10, help="sweep mode: heads to print")
    return p


def _load_model(model_name: str, device: str):
    import transformers
    import torch

    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager", dtype=torch.float32
    ).eval()
    if device != "cpu":
        model = model.to(device)
    return model, tok


def _run_single_head(args, rhos) -> int:
    model, tok = _load_model(args.model, args.device)
    hs = extract_head_logprob_scores(
        model,
        tok,
        args.prompt,
        args.layer,
        args.head,
        device=args.device,
        query_pos=args.query_pos,
    )
    kwargs = {"timeout_ms": 8000}
    if rhos is not None:
        kwargs["rhos"] = rhos
    cr = certified_radius(hs.scores, hs.argmax_idx, tau=args.tau, **kwargs)

    print(f"L{args.layer}.H{args.head}  (tau={args.tau})")
    print(f"  attends_to : {hs.tokens[hs.argmax_idx]!r} (key index {hs.argmax_idx})")
    print(f"  target_prob: {hs.target_prob:.4f}")
    print(f"  certified_radius: {cr.radius:.4g}")
    print("  per-rho verdicts:")
    for rho in sorted(cr.verdicts, reverse=True):
        print(f"    rho={rho:<7g} -> {cr.verdicts[rho]}")
    return 0


def _run_sweep(args, rhos) -> int:
    atlas = AttentionCertAtlas(
        args.model,
        device=args.device,
        prompt=args.prompt,
        tau=args.tau,
        rhos=rhos,
        query_pos=args.query_pos,
    )
    res = atlas.run()
    blob = res.to_json()

    ranked = sorted(blob["records"], key=lambda r: r["certified_radius"], reverse=True)
    n_cert = sum(1 for r in blob["records"] if r["certified_radius"] > 0.0)
    print(
        f"{args.model}: {len(blob['records'])} heads swept, "
        f"{n_cert} certify (radius>0) at tau={args.tau}"
    )
    print(f"  top {args.topk} by certified radius:")
    for r in ranked[: args.topk]:
        print(
            f"    L{r['layer']:>2}.H{r['head']:<2}  radius={r['certified_radius']:<7g}"
            f"  prob={r['target_prob']:.3f}  -> {r['attends_to_token']!r}"
        )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(blob, f, indent=2)
        print(f"  wrote atlas JSON -> {args.out}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    rhos = _parse_rhos(args.rhos)

    single = args.layer is not None or args.head is not None
    if single:
        if args.layer is None or args.head is None:
            print("error: --layer and --head must be given together", file=sys.stderr)
            return 2
        return _run_single_head(args, rhos)
    return _run_sweep(args, rhos)


if __name__ == "__main__":
    raise SystemExit(main())
