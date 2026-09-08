# What this study can add

The broad idea of replacing neural components with equations is established. A new contribution must come from an evidenced method, result, or failure analysis. No priority or state-of-the-art claim is supported by this repository's current benchmark alone.

| Prior work | Relevant result | Implication for this study |
|---|---|---|
| [SymTorch, Tan, Soubki & Cranmer, 2026](https://arxiv.org/abs/2602.21307) | Uses PySR to distill neural components, including transformer MLP surrogates. Reports an 8.3% throughput improvement with performance degradation. | Component-to-equation replacement and LLM acceleration are not new ideas. Compare fidelity, constraints, and actual timing rather than claiming the concept. |
| [SymRefine, Wei et al., 2026](https://doi.org/10.1016/j.neucom.2026.132719) | Uses symbolic module replacement and subsequent fine-tuning on MLP/CNN benchmarks. | Symbolic compression also has prior non-LLM evidence. Our surrounding model stays frozen, so accuracy tradeoffs are not directly comparable. |
| [Arithmetic Without Algorithms, Nikankin et al., 2025](https://arxiv.org/abs/2410.21272) | Attributes arithmetic performance to combinations of heuristic neurons. | Original-neuron sparse controls are necessary competitors to a learned equation. |
| [Language Models Use Trigonometry to Do Addition, Kantamneni & Tegmark, 2025](https://arxiv.org/abs/2502.00873) | Fits geometric number representations and tests proposed computations using causal interventions. | An accurate latent equation is not automatically a semantic explanation of addition. |
| [Fine-Grained Manipulation of Arithmetic Neurons, Du et al., 2025](https://aclanthology.org/2025.blackboxnlp-1.27/) | Identifies and causally manipulates arithmetic heuristic neurons. | Intervention itself is established; the replacement must preserve the original response on unseen interventions. |
| [Are Arithmetic Heuristic Neurons Form-Invariant?, Naganna et al., 2026 preprint](https://arxiv.org/abs/2607.16693) | Studies shared arithmetic neurons across symbols, text, and code in three Llama models. | Prompt-format transfer alone is not novel, and training on all three formats is weaker evidence than holding a format out. |

Goodfire's [geometric calculator study](https://www.goodfire.com/research/a-geometric-calculator), published May 14, 2026, identifies a shared addition mechanism in Llama 3.1 8B operating on circular number representations across arithmetic, days, and months. Their [SAE geometry study](https://www.goodfire.com/research/can-saes-capture-neural-geometry), published May 21, examines how groups of SAE directions represent curved manifolds. A useful connection is to model the computation between independently characterized representations and test its behavior under controlled edits. Our active coordinates currently have no established day/month, digit, or modular interpretation. This study is not a reproduction of Goodfire's semantic finding.

The hypothesis tested here is narrower: a small differentiable EML mapping, trained with intervention responses and optional derivatives, may preserve a localized scalar contribution on a specified family of arithmetic prompts. The prospective study compares that mapping with equally budgeted neural heads, linear predictors, and original neurons; tests independent operands and another model family; and separates natural edits from failures outside the retained feature subspace. Whether these experiments support a publishable contribution depends on their completed results. Whole-block compression is evaluated separately and cannot be inferred from scalar fidelity.

Literature checked September 8, 2026. Comparisons describe the cited authors' results, not independently reproduced performance. The method and benchmark budgets differ, so their headline numbers should not be ranked directly against ours.

## EML versus SiLU: what the comparison establishes

The [nonpolynomial activation theorem of Leshno et al.](https://pinkus.net.technion.ac.il/files/2021/02/neural.pdf)
concerns approximation with unrestricted width and appropriate biases. It does
not rank finite networks at a fixed parameter or optimization budget. SiLU meets
its activation conditions. Applying the theorem to the mathematical head family,
setting the EML right affine map to zero leaves an exponential activation. The
implemented clipped exponential is also continuous and nonpolynomial; clipping
alone does not remove this abstract approximation property. The theorem supplies
no error guarantee at our fixed parameter budget or in floating-point execution.

The [original EML paper](https://arxiv.org/html/2603.21852v2) constructs calculator
operations through repeated applications of the operator, with complex
intermediates needed for parts of the construction. Our real-valued head uses
one EML stage with logarithm argument `1 + right(x)**2`. It does not implement
that full construction. The identity `eml(x, 1) = exp(x)` also becomes a clipped
exponential in `safe_eml` outside its supported argument range.

At input rank 32 and width 32, the scalar EML and SiLU heads each store 2,177
parameters, including their linear skip. EML has parallel left/right affine
arguments and one nonlinear stage; the comparator has two sequential SiLU
stages. This is a comparison of complete architectures at matched head storage.
It does not isolate activation choice, match depth, or guarantee equal runtime.
The dense feature projection adds storage to both predictors.

Exponential derivatives can grow quickly, but numerical instability has not
been established as the cause of the observed results. Training clips gradient
norms to 1.0. The [GPU diagnostic](head-numerics.json) records zero nonfinite-loss
or nonfinite-gradient steps across the 270 primary fits. On the stored training
and validation features, none of the 135 final EML checkpoints reaches an
exponential or logarithm argument clamp. This post-fit check cannot recover
unrecorded training trajectories, gradient-clipping frequency, or conditioning.
It does not cover the held-out inputs.

A further architecture study would compare depth and width sweeps under explicit
parameter and measured runtime budgets. A SwiGLU control would share EML's two
affine branches and readout; a wider single-stage SiLU would provide another
comparison. Such a study needs separate validation and a fresh holdout after
the present results. Neither mathematical paper implies a general advantage or
disadvantage for EML on this task, and the current evidence should not be framed
as a mathematical limitation of EML.

## A fixed feature projection imposes a separate limit

The current heads receive an affine projection `z = P h + b`, including feature
normalization. Both their nonlinear branch and linear skip use only `z`.
Universality for functions of `z` therefore does not imply universality for
functions of the full hidden state `h`.

If `P delta = 0`, every deterministic head satisfies
`g(P (h + delta) + b) = g(P h + b)`, regardless of its activation, width, or
depth. If the original scalar coefficients at these two inputs are `y0` and
`y1`, their shared prediction `v` obeys the elementary identity

```text
[(v - y0)^2 + (v - y1)^2] / 2
  = (v - (y0 + y1)/2)^2 + (y1 - y0)^2 / 4
  >= (y1 - y0)^2 / 4.
```

In this study's scalar-intervention diagnostic, the remaining MLP contribution
and model inputs stay fixed. An unchanged predicted coefficient thus gives zero
downstream margin response. Against a nonzero original response, this yields
response NRMSE exactly one in exact arithmetic; correlation with a constant
response is undefined. The implemented nullspace projection is approximate in
FP32, so measured responses can be small rather than exactly zero.

The [Gemma addition geometry audit](gemma/geometry-addition.json) illustrates
this distinction. Across 32 operand groups and three formats, the original
coefficient response RMS in the active-feature nullspace is 0.01417, versus
about 0.000002 for either primary head. Their downstream response NRMSE values
are both about 1.0003. The original margin response RMS is only 0.001368, so the
relative error alone overstates the absolute effect. All 12 methods and five
strengths are retained in each raw geometry cohort.

This is a limitation of the fixed representation, shared by EML and SiLU. More
depth may improve approximation within the retained features; recovering
sensitivity to omitted directions requires changing the information supplied
to the predictor. These diagnostic results do not select a new representation
or change the frozen primary criteria.
