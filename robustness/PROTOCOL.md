# Prospective robustness study

This extends commit `462246ac050794bee73cf0265800153037b38d46`. Earlier results are exploratory evidence, not confirmation data for this study. The goal is a defensible, useful component-replacement finding with independently generated problems, model-family replication, explicit failure boundaries, and a separate whole-MLP experiment. Public impact cannot be guaranteed by a benchmark pass.

## Stage A: fresh scalar-component confirmation

Models are pinned in `models.json`: Qwen3-1.7B, Qwen3-4B, and SmolLM2-1.7B-Instruct. All three operations—addition, multiplication, exact integer division—are mandatory for every model. Each operation has 2,048 training, 512 validation, and 1,024 primary test operand pairs. All clean and corrupted operand problems are disjoint between splits and from earlier studies. The three models share the new dataset; cross-model results are not independent data replications. Individual operand values can recur in the in-range tests; range shifts hold out larger values.

Training ranges are addition 10..499, multiplication 2..199, and division with quotient 2..499 and divisor 2..79. Addition reserves units sums 17 and 18 for a carry test. Larger-range tests use addition 500..1499, multiplication 200..399, and division quotient 500..1499 with divisor 80..129. Each shift has 512 pairs. Addition has 512 carry pairs. An additional 1,024 ordinary-generation cases per operation are sampled without conditioning on the answer's first digit. They are separate from the contrast-conditioned primary pairs and use no corruption at evaluation.

Training formats remain prose, symbolic, and code. Four held-out formats test equation completion, named operands, reversed presentation order, and irrelevant numeric context. Additional 256-pair tests change both operands. Interpolation strengths 0, .25, .5, .75, 1 train the heads; .125, .375, .625, .875 are the primary test strengths. Strengths -.25 and 1.25 are extrapolation diagnostics. Natural interventions, active-coordinate edits, and random ambient directions are reported separately. Nullspace interventions explicitly test limits of a low-rank representation; they are not expected to establish unrestricted equivalence.

Layer localization uses training activation patching followed by validation selection exactly as in the previous pipeline. Encoders use training activations/gradients only. The fixed head families are EML-square and two-layer SiLU, rank 32 and width 32, with 2,177 head parameters each. Five seeds are fixed: 101, 211, 307, 401, 503. Each fit uses at most 20,000 steps, validation every 100 steps, and the existing early stopping and optimizer rules. The primary family uses active features and derivative-loss weight .1. Controls use active features without derivative loss and PLS features without derivative loss. All fit value MSE plus four times response MSE. Validation selects one seed per family/method; every seed's performance is also reported. No test-based choice among methods, operations, models, or seeds is permitted.

Controls include a constant coefficient, a validation-selected linear predictor, 16 original SwiGLU neurons at approximately matched storage, and a validation-selected sparse original-neuron predictor. The comparator receives the same fitting data and validation budget. Random-feature and PCA encoders are diagnostic controls if added before their evaluations, not substitutes for the primary comparison.

Primary fidelity criteria retain response NRMSE <= .20 and correlation >= .90. The response denominator is RMS original downstream margin change. Report simultaneous 95% interval bounds across the nine operation/model cells using Bonferroni-adjusted operand-group bootstrap intervals. Also report ordinary per-cell intervals and paired EML–SiLU differences without claiming operator superiority from a lone favorable comparison.

Accuracy requires more than a degenerate bootstrap interval. For each independent operand group, count the net number of correct answers lost across the three formats. Bound expected loss conservatively by the sum of the three exceedance probabilities (loss at least 1, 2, or 3 answers), divided by three. Use one-sided exact binomial upper limits, with the error budget divided across three thresholds and nine primary cells. This remains nonzero when no losses are observed. The primary noninferiority target is an upper bound <= one percentage point. Report the unconstrained ordinary cohort separately. Low original-model accuracy limits the interpretation even if fidelity is excellent.

The primary patch acts at the last prompt token. A mandatory subsequent test reapplies the replacement at every generated token, including ordinary unconditioned prompts. This measures accumulating errors and tests use beyond a single prefill intervention. Greedy generation allows 16 new tokens. The chat template disables thinking when supported and uses the fixed assistant prefill `The answer is `. Scoring requires the full continuation to be numeric apart from whitespace and optional final punctuation. Metrics refer to the first answer token, which need not represent one digit in every tokenizer.

All failures and incomplete fits remain in the report. Test files, protocol, model revisions, and source hashes are frozen before model collection. Subsequent necessary software repairs must be logged with their timing and impact; test-driven model retuning requires a new holdout.

## Stage B: whole-MLP replacement and utility

The scalar result does not eliminate the original MLP. A separate vector-valued student will replace one complete MLP block, using training activations from all prompt positions and a general-language corpus. Freeze its architectures, data split, and acceptance criteria in a stage-specific protocol before fitting; do not infer success from the scalar experiment. Compare EML against equally budgeted SiLU and linear/low-rank controls.

Measure complete arithmetic accuracy, held-out language cross-entropy, all-token behavior, stored coefficients including projections/normalization, and end-to-end prefill/decode latency. Latency must compare the original and truly replaced models with identical precision, batches, and token budgets. Reusing the original MLP to compute an orthogonal residual is not a whole-block replacement. Whole-block failures are reportable outcomes, not grounds to relabel the scalar benchmark as acceleration.

## Release requirements

A credible result requires frozen test artifacts, all nine scalar cells and stress conditions, seed/control comparisons, rare-regression uncertainty, the whole-block experiment, GPU validation, an auditable report, and reproducible source. Interpretability claims require additional semantic/intervention evidence; a small equation alone does not identify an arithmetic algorithm. Keep the core library separate from research artifacts and commit reviewable changes as work progresses.

## Pre-collection corrections

Before generating any model outputs, the dataset assertion showed that addition 500..799 admits no equal-length contrasts with different first digits: every sum starts with 1. The held-out addition range was widened to 500..1499. This correction precedes the data/source freeze and all new-study test exposure. Dataset seed: 20260923.

The same generator check found too few division contrasts after historical exclusions: quotients 100..199 all begin with 1. The in-range quotient pool is now 2..499 and the shifted quotient pool 500..1499, retaining disjoint ranges and the original sample counts. No division dataset had been written and no model collection had begun.
