# Gemma execution amendment

This append-only snapshot completes the user-requested model substitution described in `../GEMMA_AMENDMENT.md`. Gemma training/validation feature collection has finished; Gemma fitting and held-out evaluation have not started. Partial Qwen and superseded SmolLM2 held-out outputs already exist.

The only changes to previously frozen Python sources are a Gemma branch and subprocess dispatch helper in `runtime.py`, and adding the Gemma command-line choice in `seed_evaluate.py`, `geometry_evaluate.py`, `analyze.py`, and `per_style.py`. The original implementations remain recoverable from Git and their original hashes remain in the earlier snapshots. Learning functions, hyperparameters, data, intervention definitions, selection rules, and acceptance criteria are unchanged. The common fit/evaluation launchers now route Gemma stages through the separately frozen adapter. Qwen stages retain their original direct execution path.

`execution-freeze.json` records the exact superseded hashes and pins the dispatchers, amended roster, and source verifier before Gemma fitting. `source-freeze.json` separately pins the native adapter, collection, validation, and sparse GELU control. Final reporting must keep the 270 primary scalar candidates distinct from the 90 superseded SmolLM2 candidates and 27 whole-block candidates.
