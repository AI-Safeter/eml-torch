"""Reproduce cache equality against the native fitted-head validation cohorts."""

import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma import adapter


def main():
    adapter.install()
    import evaluate_heads as evaluate
    from data import STYLES
    from gemma.prefix_cache import GemmaPrefixCache
    from model_io import expand, load, logits, setup
    from runtime import HERE, RUNS

    setup()
    out = RUNS / "gemma/add"
    selection = json.loads((out / "selection.json").read_text())
    rep = evaluate.Replacements(out, selection)
    rows = expand(json.loads((out / "problems.json").read_text())["validation"][:4], STYLES)
    model, tok = load()
    checked = {}
    with torch.inference_mode(), GemmaPrefixCache(model, rep.layer, logits) as cache:
        for name, fn, kwargs in [
            ("ordinary-validation-audit.json", evaluate.ordinary, {}),
            ("ordinary-all-tokens-validation.json", evaluate.ordinary, {"all_tokens": True}),
            ("interventions-validation-audit.json", evaluate.interventions, {}),
            ("coordinate-validation.json", evaluate.coordinate_interventions, {}),
        ]:
            expected = json.loads((out / name).read_text())
            evaluate.logits = cache.logits
            try:
                actual = fn(model, tok, rep, rows, **kwargs)
            finally:
                evaluate.logits = logits
            assert actual == expected, name
            checked[name] = {
                "methods": len(actual),
                "records_per_method": len(actual["original"]),
                "native_source_sha256": hashlib.sha256((out / name).read_bytes()).hexdigest(),
                "all_raw_records_identical": True,
            }
    report = {
        "scope": "Actual fitted Gemma heads and native GELU controls, validation prompts only; cached results exactly equal to the original uncached evaluation files",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "files": checked,
        "cache_source_sha256": hashlib.sha256(
            (HERE / "gemma/prefix_cache.py").read_bytes()
        ).hexdigest(),
    }
    (HERE / "gemma/prefix-head-validation-reproduction.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
