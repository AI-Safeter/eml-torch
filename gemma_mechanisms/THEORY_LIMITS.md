# What the failure bounds do and do not establish

These arguments concern two specific restrictions in this experiment. Neither
is a theorem that EML cannot outperform SiLU.

## A narrow affine output decoder

The trained student outputs `b + D z(x)`, where `D` has at most `r` columns.
Regardless of the depth or nonlinear activation used to obtain `z(x)`, every
real-arithmetic output lies in one affine subspace of dimension at most `r`.

Center the teacher targets and form their covariance using the training loss's
equal weighting of arithmetic and language domains. Let its eigenvalues be
`λ₁ ≥ … ≥ λ_d ≥ 0`. For any chosen decoder subspace, squared error includes
the squared target component orthogonal to that subspace. The smallest possible
average orthogonal error over all rank-`r` subspaces is the sum of the last
`d-r` eigenvalues. Projecting onto the first `r` eigenvectors attains it.

Consequently, variance-normalized raw output MSE is bounded below by

```
(λ_(r+1) + … + λ_d) / (λ_1 + … + λ_d).
```

The audit compares this covariance-tail fraction with the normalized FP32
training error, allowing numerical tolerance for the normalization arithmetic.
This bound still applies when the decoder learns a different subspace: the
leading eigenspace is already the best possible rank-`r` subspace. It does not
require that the learned encoder recover that optimal projection.

The bound addresses raw MLP output and this training distribution. It does not
bound answer accuracy or the nonlinear post-normalized residual contribution.
It also does not describe arbitrary EML networks, wider/full-rank decoders,
skip paths outside the bottleneck, or every consequence of BF16 rounding.
Error above the floor can come from the encoder, nonlinear capacity, optimization,
or the competing contribution loss; the floor alone cannot separate them.

## An intermediate that omits a causal input

Suppose two interventions produce exactly the same proposed input `x`, but
different downstream vectors `y₁` and `y₂`. Every deterministic equation of
`x` alone must predict the same vector `q` in both cases. For equal weighting,

```
(||q-y₁||² + ||q-y₂||²) / 2
  = ||q-(y₁+y₂)/2||² + ||y₁-y₂||²/4.
```

The best shared prediction is their midpoint; its unavoidable average error is
`||y₁-y₂||²/4`. The response analysis uses the corresponding per-quantity
training-scale normalization. This applies equally to EML, SiLU, linear and
symbolic equations with that input signature.

Here the interventions hold the donor's last-token layer-26 residual fixed
while varying the reused KV state. Equality of that residual does not imply
equality of later states or logits. Gemma's later layers also receive shared
KV; excluding it can make a residual-only equation underdetermined.

This tests sufficiency of one proposed input, not the impossibility of recovering
an arithmetic mechanism. An equation with additional validated inputs could
avoid the collision. Restoring native KV is a causal state transplant; it is
not a compressed cache, a symbolic carry algorithm, or an EML replacement.

The omitted-KV limitation applies to the cross-layer equation. The native MLP
itself is a deterministic function of its complete input vector. Its compact
replacement therefore has a well-defined target; missing KV does not explain
that replacement's reconstruction error. The two failure analyses concern
different input signatures and must not be conflated.
