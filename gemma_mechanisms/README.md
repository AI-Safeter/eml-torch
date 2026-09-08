# Gemma complete-MLP replacement study

This is a new prospective experiment. Its integration checks demonstrate actual MLP removal, **not retained model quality or a recovered arithmetic algorithm**. The earlier scalar study remains independently frozen under `robustness/`.

Read [PROTOCOL.md](PROTOCOL.md) for the hypotheses, stopping gates, accounting rules, and protected final evaluation. [model-identity.json](model-identity.json) verifies the exact local checkpoint. The 3M/6M coefficient grid compares EML and SiLU at nonlinear depths 1/2/4 and includes factorized linear controls, with three seeds. Training targets include the full MLP output and its native normalized residual contribution.

From this checkout, use the pinned Gemma environment. Commands write to a separate run directory; pretrained weights and corpus text are not committed. Check GPU availability before assigning a device. Collection and full-model validation need about 6–8 GiB free when the PLE table is on CPU. Offline fitting uses CPU-resident targets and a much smaller CUDA working set.

```bash
PY=/home/ubuntu/samuel/emltorch-gemma-env/bin/python
$PY -m gemma_mechanisms.identity
$PY -m gemma_mechanisms.prepare
CUDA_VISIBLE_DEVICES=1 $PY -m gemma_mechanisms.validate --full-model
CUDA_VISIBLE_DEVICES=1 $PY -m gemma_mechanisms.collect
CUDA_VISIBLE_DEVICES=2 $PY -m gemma_mechanisms.train
```

Default results: `/home/ubuntu/samuel/emltorch-gemma-replacement-runs`. Override with `--output` or `EML_GEMMA_MECHANISMS_RUNS`. Training has no teacher-model dependency. `train --candidate eml-b3000000-d1-s1103` runs one unchanged grid member, which the full-grid command later preserves. A fit failure is recorded and cannot silently disappear from selection.

The loaded model has 5,104,297,504 unique parameters; the target MLP has 56,623,104. A 3M replacement removes about 1.05% of total parameters. The 20% deployment target requires many blocks to pass fresh gates. CPU offload is memory placement, not compression, and will be applied equally to timing baselines.

The causal branch starts with addition unit carry and digit hypotheses. Collection includes candidate layer states at last-prompt and answer-prefix positions. Feature predictability alone will not be reported as mediation. No multiplication/division mechanism claim follows from an addition experiment.
