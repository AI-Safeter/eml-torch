"""Replay causal replacement with the shallow search winner as an additional control."""

import argparse
import json
import time

import torch
from analyze import intervention
from analyze import ordinary as ordinary_metrics
from common import OUT, load, setup
from evaluation_helpers import KINDS, interventions, ordinary
from replacement import E, Replacement, emit

from emltorch._ast import _parse_inner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates every frozen fresh pair")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    setup()
    dest = OUT / "replay"
    dest.mkdir(exist_ok=True)
    rep = Replacement()
    choices = json.loads((OUT / "stress-selection.json").read_text())["by_depth"]
    for label, depth in [("eml_deeper", "7"), ("eml_shallow_search", "3")]:
        scope = {"torch": torch, "E": E}
        source = (
            "def f(z):\n    z0,z1,z2=z.unbind(1)\n    return "
            + emit(_parse_inner(choices[depth]["expression"]))
            + "\n"
        )
        exec(compile(source, "<validated-eml>", "exec"), scope)
        rep.formulas[label] = scope["f"]
    if "eml_shallow_search" not in KINDS:
        KINDS.append("eml_shallow_search")
    model, tok = load()
    pairs = json.loads((OUT / "fresh-problems.json").read_text())
    summary = {}
    start = time.perf_counter()
    with torch.inference_mode():
        for group, style in [
            ("pairs", "prose"),
            ("range", "prose"),
            ("pairs", "symbolic"),
            ("pairs", "code"),
        ]:
            rows = pairs[group][: args.limit or None]
            name = f"{group}-{style}"
            normal = ordinary(model, tok, rep, rows, style)
            causal = interventions(model, tok, rep, rows, style)
            for prefix, data in [("ordinary", normal), ("interventions", causal)]:
                (dest / f"{prefix}-{name}.json").write_text(json.dumps(data, indent=2))
            summary[name] = {
                "ordinary": ordinary_metrics(normal),
                "interventions": intervention(causal),
            }
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))
    (dest / "metadata.json").write_text(
        json.dumps(
            {
                "seconds": time.perf_counter() - start,
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "limit": args.limit,
                "status": "replication of frozen tests; these are not newly unseen problems",
                "new_control": "depth-3 four-node candidate selected using validation only",
                "checks": "finite logits, exact restoration, alpha-zero equivalence",
            },
            indent=2,
        )
    )
    print("LLM replay complete", flush=True)


if __name__ == "__main__":
    main()
