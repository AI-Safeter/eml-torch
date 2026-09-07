# Arithmetic component analysis

This application locates an arithmetic-relevant MLP contribution in Qwen3-1.7B,
fits equations from upstream features, replaces the contribution, and checks
responses to upstream interventions. EML did not meet the causal-fidelity target.

Run the frozen experiment from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python applications/llm/run.py
CUDA_VISIBLE_DEVICES=0 python applications/llm/validate_pipeline.py
```

The pinned model downloads automatically. For offline use, set
`EMLTORCH_MODEL_PATH` to its cached snapshot. `run.py --limit 8` runs a smaller
smoke evaluation; it overwrites the replay directory, so use a separate clone
to retain the full records.

`replay/` holds this turn's complete GPU replication and the extra four-node
shallow-search control. The top-level ordinary/intervention records and
`summary.json` retain the preceding experiment, including held-out carry cases.
These cohorts and candidate equations are already evaluated; replays are not
new untouched tests. The component occupies one output direction in block 22;
the rest of the dense MLP still executes.

To rebuild selection and fitting from model activations, in a separate clone:

```bash
python applications/llm/design.py
CUDA_VISIBLE_DEVICES=0 python applications/llm/localize.py
CUDA_VISIBLE_DEVICES=0 python applications/llm/refine_features.py
CUDA_VISIBLE_DEVICES=0 python applications/llm/fit_equations.py
CUDA_VISIBLE_DEVICES=0 python applications/llm/evaluate.py
CUDA_VISIBLE_DEVICES=0 python applications/llm/analyze.py
CUDA_VISIBLE_DEVICES=0 python applications/llm/validate_pipeline.py
```

Rebuilding may choose a different component or feature map. The extra depth
expressions in `stress-selection.json` belong to the bundled frozen component;
do not combine them with a newly selected component by running `run.py` afterward.

The projection uses `d / (dᵀd)`. The pipeline audit checks that replacements do
not change the orthogonal output directions. Selection uses training and
validation partitions, and fitting inputs contain no numerical answer or
downstream logits. The three supervised latent coordinates are not established
arithmetic concepts. This experiment tests addition only.

This workflow is informed by causal interpretability research, including
[Goodfire's parameter-interpretation work](https://www.goodfire.com/research/interpreting-lm-parameters),
[arithmetic heuristics](https://arxiv.org/abs/2410.21272), and
[the addition helix study](https://arxiv.org/abs/2502.00873).
It does not integrate Goodfire software or constitute a collaboration.
