# Selected-token Jacobian lens on Gemma

This is a GPU implementation and feasibility study for connecting named internal
directions to EML equations. The initial semantic-intervention gate **failed**;
no EML equation was trained and no EML advantage is claimed. See the
[results](RESULTS.md) and [interactive measurements](explorer.html).

The reusable part computes selected rows of a corpus-averaged Jacobian directly
with vector–Jacobian products. Twelve selected tokens need twelve cotangents per
document instead of the 1,536 coordinate cotangents required for the full map.
That is a reduction in derivative queries, not a measured 128× runtime speedup.
The result is a limited token dictionary, not a full-vocabulary lens or a sparse
J-space decomposition.

The estimator follows the target-sum/source-mean definition in
[Anthropic's reference implementation](https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e)
for [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html).
We fold Gemma's final RMSNorm diagonal weights into the selected unembedding
rows, while excluding the input-dependent RMS denominator and logit softcap
from the direction calculation. Causal outcomes use the actual normalized,
softcapped model logits.

## Reproduce

Use CUDA, PyTorch `2.9.0+cu128`, Transformers `5.16.1`, and the locally cached
`google/gemma-4-E2B-it` revision in [protocol.json](protocol.json). The recorded
runs used an H100. Native calibration allocated at most 10.31 GiB; the FP32
suffix audit allocated 12.66 GiB after moving unused multimodal towers to CPU.
Allow additional memory for CUDA and shared GPU workloads.

Run from the repository root with a fresh output directory:

```bash
export EML_JSPACE_RUN="$PWD/.artifacts/jspace"
export CUDA_VISIBLE_DEVICES=3
mkdir -p "$EML_JSPACE_RUN"
gzip -dc jspace/evidence/documents.json.gz > "$EML_JSPACE_RUN/documents.json"
cp jspace/evidence/freeze.json "$EML_JSPACE_RUN/freeze.json"
python jspace/lens.py
python jspace/audit.py
python jspace/audit.py --fp32-suffix
python jspace/probe.py
```

`lens.py --limit 2` performs a short preflight; running again without the limit
resumes the same frozen corpus. Completed discovery results cannot be silently
overwritten. GPU backward reductions are not bitwise deterministic; checksums
identify the recorded artifacts, not a guarantee that a new calibration will
have identical bytes.

To draw the original corpus again instead of using the supplied frozen inputs,
set `EML_SQUAD_TRAIN` to the SQuAD training JSON named in the protocol. Its hash
is checked before selecting one 128-token paragraph from each of 64 articles.

## Evidence and scope

[evidence](evidence/) contains the frozen inputs, all 336 discovery measurements,
both numerical audits, split-half stability diagnostics, and averaged lens rows.
The larger per-document gradient tensors can be regenerated with `lens.py`.
The manifest distinguishes included files from externally stored artifacts.

The prospective gate examines eight ordered animal swaps and two formats at
three fixed layers. Animals and question formats reserved for a future equation
evaluation remain unevaluated. A failed interface gate does not refute the
J-space paper or EML: this is one small dictionary, model, corpus and intervention
design. It does prevent treating this setup as a demonstrated semantic equation
interface.

The experiment code in this directory is distributed under [Apache-2.0](LICENSE),
with [reference attribution](NOTICE). The EML library retains its MIT license.
The selected [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) Wikipedia excerpts
retain [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) attribution
to the dataset authors and original articles, identified by title in the inputs.
Model weights remain subject to Google's model terms and are not redistributed.
