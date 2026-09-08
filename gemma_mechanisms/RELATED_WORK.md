# Relation to interpretability research

This context was added after the experimental design was frozen. It does not
change selection, acceptance criteria, or the held-out comparisons.

Goodfire's SAE intervention method reconstructs an activation from features
and retains its reconstruction residual when changing a feature. That helps
isolate an intervention, but retaining an error-correction path does not by
itself establish computational compression. Our complete-MLP replacement has
no teacher-dependent correction path at inference. It must therefore preserve
behavior using only its own learned mapping.
[Understanding and Steering Llama 3](https://www.goodfire.com/research/understanding-and-steering-llama-3)

Causal abstraction tests a proposed correspondence between interpretable
variables and neural states through interchange interventions. The relevant
question is whether corresponding interventions produce corresponding
behavior, rather than whether a probe can decode the variable. Our carry
readout edits and downstream blocking tests target that distinction, but do
not validate a carry algorithm.
[Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997)

Goodfire's `causalab` provides workflows for causal models, interchange
interventions, and learned subspace alignment, including modular arithmetic
examples. It is a relevant implementation reference for a future attempt to
learn a richer alignment. This study uses its own pinned Gemma intervention
code; it neither runs that library nor claims to reproduce its experiments.
[Causalab](https://github.com/goodfire-ai/causalab)

Goodfire's parameter-decomposition work trains subcomponents to preserve
behavior under combinations of ablations, including adversarially chosen
ones. That is a stronger objective than observational activation matching.
Our equation failures illustrate why matching observed transitions cannot
substitute for intervention tests. We did not implement VPD, and the current
comparison establishes no advantage over it.
[Interpreting Language Model Parameters](https://www.goodfire.com/research/interpreting-lm-parameters)

Arithmetic may also be implemented through collections of heuristics rather
than a single schoolbook algorithm. Prior evidence for such behavior in other
models motivates testing carry interpretations as hypotheses, not labels
guaranteed by the task. It does not establish which mechanism this Gemma
checkpoint uses.
[Arithmetic Without Algorithms](https://arxiv.org/abs/2410.21272)

The useful contribution here is a reproducible, computation-removing
replacement path and a set of falsifiable causal tests with explicit failure
bounds. The dependence of later Gemma layers on shared KV is expected from
the architecture. Its confirmation is an input-sufficiency check, not a novel
arithmetic algorithm or an EML-specific impossibility theorem.
