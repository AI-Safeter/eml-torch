# Constraints removed and retained

Use normalized input `x` and normalized native MLP target `y`. All covariance
statements assume finite second moments. Empirical bounds use the same equal
domain weights and normalization as the fitted raw-MSE objective. They do not
bound normalized contribution error or answer accuracy.

## Bottleneck

`f(x) = b + D g(E x + e)`, with `D` having 512 columns. Every output lies in
one affine subspace. Consequently `rank(Cov(f(x))) ≤ rank(D) ≤ 512`. The minimum
raw empirical MSE of any such predictor is bounded below by the tail of the
centered target covariance spectrum, divided by output dimension. This follows
by orthogonally projecting targets onto the best rank-512 affine subspace.
More nonlinear stages inside `g` do not remove this output constraint.

The nonlinear mapping also cannot distinguish input perturbations in `ker(E)`.
These are distinct input and output restrictions; a decoder-only bound does
not quantify all information discarded by the encoder.

## Full-width affine shortcut

`f(x) = S x + s + D g(E x + e)`. This can have full output covariance rank, even
with no nonlinear correction, because `S` is unrestricted and square. There is
no fixed affine-output-subspace restriction on the complete prediction.

However, outside `range(D)` the prediction is still affine in the input, and
nonlinear dependence on `ker(E)` is still absent. Let `X` include a constant
column and let `R = Y - X X⁺ Y` be the residual of the *unregularized optimal*
empirical affine fit, in square-root domain-weighted coordinates. For any
orthogonal projector `P` onto a candidate decoder range, the best achievable
outside-range error is `||R (I-P)||²`. Minimizing over rank-r projectors gives
the tail sum of eigenvalues of `RᵀR`, divided by the normalization used for MSE.
This is a lower bound for the shortcut architecture even when its nonlinear
function is unrestricted. A ridge-fit residual would generally overstate this
bound and must not be substituted for the optimal affine residual.

For a fitted model, separately measure target error outside its learned decoder
range after subtracting its learned shortcut, and prediction error within that
range. Also compute the best-affine outside-range error for that same decoder.
The gap distinguishes a suboptimal learned affine component from the remaining
constraint; it does not uniquely separate encoder loss, nonlinear capacity,
optimization, and competing objectives.

Numerical certification checks the FP64 covariance solve's conditioning,
normal-equation residual, and residual covariance against directly accumulated
residuals. If the solve cannot be certified, report the numerical limitation
instead of promoting a regularized or rank-truncated estimate to a rigorous
lower bound.

## Structured full-width residual states

Each argument/readout map is `diag(d) + U V`, with independent parameters for
the two EML arguments. A nonzero diagonal allows a full-rank map even if `U V`
has small rank. Nonlinear residual stages preserve the ambient state width of
1536, followed by a square affine decoder. This removes both the mandatory
512-dimensional hidden state and the mandatory rank-512 output subspace.

There is no positive covariance-tail bound implied solely by the low-rank
factor sizes: diagonal, coordinatewise nonlinear transformations can already
produce full-rank covariance. The general bound used here is zero. Finite
parameter count, limited structured cross-coordinate mixing, LayerNorm in the
update branch, finite depth, and optimization remain constraints. Identity
skips preserve an input route but do not guarantee injectivity: learned updates
can cancel input directions, and learned decoders can become singular.

## Covariance rank is not information preservation

For a multidimensional input, `f(x) = (x₁, x₁², …, x₁ᵈ)` can have full output
covariance rank for a continuous nondegenerate distribution of `x₁` while
discarding every other input coordinate. Therefore higher covariance rank does
not establish preserved information. Likewise, an unrestricted full-rank
shortcut can be canceled along directions by its correction.

Report covariance spectra and Jacobian/state diagnostics as diagnostics. The
held-out answer and general-language tests determine whether the removed MLP
computation was usefully replaced. None of these fits establishes a recovered
arithmetic algorithm or a general efficiency theorem for EML.
