# Full-width replacement architecture screen

This prospective stage starts from commit `4a9ad6b` and the corrected Gemma
study. It tests architecture and activation efficiency at a fixed coefficient
budget. No arithmetic-mechanism discovery is part of this stage.

The pinned local model is `google/gemma-4-E2B-it`, revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`. Configuration, safetensors metadata,
native parameter counts, and a CUDA integration check must agree before fitting.
The complete layer-26 MLP is replaced; its forward must be forbidden during
replacement inference. Teacher activations supervise offline fitting only.

## Designs and fairness

At depth one, fit bottleneck, affine-shortcut, and structured full-width models
with EML and SiLU, seed 1103, and a ceiling of 6,000,000 coefficients. Trainable
parameters and retained statistics both count. EML/SiLU share nonlinear stage
depth and normalization within each architecture; nonlinear inner widths or
structured ranks consume the same budget. These widths/ranks need not be equal
because EML learns two arguments. The full accounting is published.

The bottleneck uses the existing 512-state architecture. The shortcut uses a
full 1536-by-1536 affine map plus a 512-state nonlinear correction. The structured
network retains 1536-dimensional states and uses independent diagonal-plus-low-
rank argument and readout projections, followed by a full-width affine decoder.
Independent EML argument factors are packed into shared GEMM calls without tying
their weights. The structured input normalizer remains explicitly stored at
deployment; its channelwise scaling cannot be folded through LayerNorm.

Every architecture starts from an affine prediction, with zero nonlinear
readouts. This differs from the old study's tiny random readouts and is used for
both activations here. Bottleneck initialization uses the previous ridge/PCA
construction. The shortcut starts from the full ridge map, with an input PCA
encoder and a decoder initialized from affine-residual principal components.
The structured decoder starts from the same full ridge map, with identity
residual states. All preprocessing/initialization cost is metered. Initialization
differences across architectures are disclosed; paired EML/SiLU initial affine
predictions must agree.

All fits use the previous corrected training activations and select checkpoints
on previous selection activations. They use FP32, AdamW, the same minibatch
sequence per seed, 12,000 updates, and the same raw-plus-contribution objective.
Each domain supplies 256 samples per update. The full hyperparameters are in
`protocol.json`. No architecture-specific learning-rate search is allowed.

## Budget and stopping rules

The cap is **eight H100 device-hours**, measured as the sum of elapsed time of
our GPU subprocesses, including startup and validation. This is a shared-device
elapsed-time budget, not a measurement of active hardware compute. The scheduler
reserves time before launching jobs, records failures/timeouts, and never stops
another user's process. A fit has a 900-second maximum. A timeout does not count
as a completed matched-effort comparison.

Allocate two hours to the initial screen, at most one hour to refinement, four
hours to confirmation, and one hour to benchmark/audit work. Unused early time
can fund later mandatory stages. One hour stays reserved until confirmation
jobs finish. No budget expansion is implicit. If time prevents completion,
report the missing measurements and make no acceptance claim from them.

Screen all six candidates on development activations and small downstream
development sets. A new architecture is promising when at least one activation
improves **both** raw and contribution error by at least 10% relative to its
matching bottleneck, improves development multiplication accuracy by at least
3 percentage points, loses at most 2 points on other development accuracy
endpoints relative to that bottleneck, and increases document cross-entropy
by no more than 0.02 nats relative to the original. Both activation fits must
be finite. These are selection thresholds, not statistically confirmed gains.

Both activations advance together. Rank promising architectures by their mean
fractional objective improvement over the two matching bottleneck controls;
ties favor the shortcut. Confirm at most one new architecture plus the
bottleneck controls. Report all other screen results.

If best checkpoints occur in the last 10% of updates, prioritize a 24,000-update
continuation for both activations of the leading architecture and its controls.
Continuation restores the exact optimizer and minibatch state. Select the longer
effort only if both leading-architecture activations improve the objective by
at least 2% without worsening either error term. Otherwise retain 12,000 updates
and report the longer runs as diagnostics. If no architecture passes downstream
screening but one improves both activation errors by 10%, it may receive this
optimization diagnostic before a final stop decision.

Only after an architecture passes the downstream screen, and if refinement
time remains, compare depth two for that architecture and its bottleneck
controls. Both activations use the same depth and update budget. Do not select
activation-specific depths or training efforts. Final selection uses only
development evidence and is frozen before any confirmation inference.

## Fresh confirmation

All previously inspected evaluations are development data. Before fitting,
freeze fresh ordinary and shifted operand groups, FineWeb-Edu documents, and
ARC-Easy questions. Exclude prior canonical operand pairs and counterfactual
donors, previous document hashes/hostnames, and previous QA IDs/normalized text
hashes. ARC's previously unused official train partition supplies confirmation
questions; this is freshness relative to our experiments, not a claim about
absence from model pretraining.

The primary arithmetic format is the previously tested Python-expression
question, with 1,024 fresh groups per operation. The same strict integer-only
parser and 24-token generation budget remain fixed. Reversed prompts on 256
groups per operation and 128 larger-operand groups per operation are diagnostic.
Malformed outputs remain errors; parseability is reported separately. General
quality uses 1,024 fresh ARC questions and 256 documents with BOS plus 256 text
tokens, using the same native BF16 scoring as the previous study.

Confirm the selected architecture and bottleneck with both activations and
seeds 1103/2207/3301. Screen-seed weights remain eligible because selection never
uses fresh confirmation. All seeds and unconditional results remain visible.

The expansion gate uses conservative paired-regression Clopper-Pearson bounds
for addition, multiplication, division, and ARC, and document/hostname-clustered
cross-entropy bounds. Alpha is 0.05 divided by five endpoints and three seeds
within each activation. Require at most one percentage point accuracy loss,
at most 0.02 nats CE increase, finite execution, and at least fourfold block
compression in every EML seed. Descriptive paired 95% intervals and diagnostic
subsets do not replace this gate. No tuning follows confirmation.

The deployment targets remain at least 20% fewer total parameters, at least
10% lower declared-workload latency, and at most one point of accuracy loss.
They are targets. A single replacement cannot establish the total-parameter
target. Multiple blocks require a passing frozen quality gate and a separately
frozen next-stage protocol; no automatic expansion follows a failed gate.

## Cost and interpretation

Benchmark the actually installed, fully GPU-resident replacement against its
paired original at batches 1/8, prefill lengths 128/512, and 32 fixed cached
decode steps. Use 10 warm-ups and 30 alternating pairs, native eager SDPA,
synchronization, and identical algebraic deployment optimizations. Report wall
and CUDA timing, prefill, decode, throughput, peak allocated/reserved memory,
cache storage, and all model/module parameters and buffers. CPU comparison
modules and transfers are outside the timed deployed path and are disclosed.
Report shared-GPU telemetry and within-run block-bootstrap uncertainty.

Recompute the appropriate bounds in `THEORY.md`. Higher covariance rank is only
a diagnostic. It does not show preserved information, causal mediation, or
answer quality. If both activations benefit similarly, call the result an
architectural improvement. An EML-specific claim requires paired confirmation
evidence against matched SiLU and comparable storage/training/inference cost.
