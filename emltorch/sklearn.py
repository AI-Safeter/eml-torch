"""
scikit-learn compatible wrapper for emltorch.

Exposes `EMLRegressor`, a drop-in estimator that calls `emltorch.fit` under
the hood. scikit-learn is an optional dependency: if it isn't installed,
`BaseEstimator` / `RegressorMixin` gracefully fall back to `object`, which
means the class still works standalone (with `fit` / `predict`) - just
without participation in sklearn pipelines / grid search.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - optional dep
    class BaseEstimator:  # type: ignore
        def get_params(self, deep=True):
            return {k: getattr(self, k) for k in self._param_names()}

        def set_params(self, **params):
            for k, v in params.items():
                setattr(self, k, v)
            return self

        def _param_names(self):
            return []

    class RegressorMixin:  # type: ignore
        def score(self, X, y):
            pred = self.predict(X)
            y_arr = np.asarray(y, dtype=float).reshape(-1)
            pred_arr = np.asarray(pred, dtype=float).reshape(-1)
            ss_res = float(((y_arr - pred_arr) ** 2).sum())
            ss_tot = float(((y_arr - y_arr.mean()) ** 2).sum()) or 1e-12
            return 1.0 - ss_res / ss_tot

    _HAS_SKLEARN = False


from .api import fit as _fit


class EMLRegressor(BaseEstimator, RegressorMixin):
    """scikit-learn compatible EML symbolic regressor.

    Calls :func:`emltorch.fit` under the hood. After fitting:

    - ``self.expression_`` - discovered formula string
    - ``self.r2_``         - R2 on training data
    - ``self.a_``, ``self.b_`` - affine rescaling coefficients
    - ``self.tree_``       - the stored best tree (for predict())
    - ``self.best_idx_``   - which slot in the population was best
    """

    def __init__(
        self,
        depth: int = 3,
        strategy: str = "auto",
        population: int | None = None,
        generations: int | None = None,
        device: str = "cuda",
        r2_target: float = 0.99,
    ):
        self.depth = depth
        self.strategy = strategy
        self.population = population
        self.generations = generations
        self.device = device
        self.r2_target = r2_target

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Fit a symbolic formula.

        Args:
            X: array-like of shape (N, V) or (N,). Each row is a sample.
            y: array-like of shape (N,).
        """
        X_t = _to_tensor(X)
        y_t = _to_tensor(y).reshape(-1)

        # sklearn convention: X is (N, V). emltorch.fit expects (V, N) or (N,).
        if X_t.dim() == 1:
            x_arg = X_t
        elif X_t.shape[1] == 1:
            x_arg = X_t.reshape(-1)
        else:
            x_arg = X_t.T.contiguous()

        result = _fit(
            x_arg,
            y_t,
            depth=self.depth,
            strategy=self.strategy,
            population=self.population,
            generations=self.generations,
            device=self.device,
            r2_target=self.r2_target,
        )

        # --- Re-run fit to also retrieve the tree for predict(). ---
        # The public api.fit drops the tree on the floor, so for predict()
        # we rebuild by calling evolve() directly and caching its tree.
        from .evolution import EvolutionConfig, evolve

        if X_t.dim() == 1:
            V = 1
        elif X_t.shape[1] == 1:
            V = 1
        else:
            V = X_t.shape[1]

        N = y_t.shape[0]

        population = self.population
        if population is None:
            population = {1: 256, 2: 256, 3: 1024, 4: 2048}.get(self.depth, 4096)
        strategy = self.strategy
        if strategy == "auto":
            strategy = "random" if self.depth <= 2 else "evolution"
        generations = self.generations
        if generations is None:
            generations = 1 if strategy == "random" else 20

        x_for_evolve = x_arg if x_arg.dim() == 2 else x_arg.unsqueeze(0)
        cfg = EvolutionConfig(
            depth=self.depth,
            num_vars=V,
            population=population,
            generations=generations,
            elite_fraction=0.1,
            mutations_per_child=0 if strategy == "random" else 1,
            device=self.device,
            r2_target=self.r2_target,
        )
        evo_result = evolve(x_for_evolve, y_t, cfg)

        self.expression_ = evo_result.best_expression
        self.r2_ = float(evo_result.best_r2)
        self.mse_ = float(evo_result.best_mse)
        self.a_ = float(evo_result.best_a)
        self.b_ = float(evo_result.best_b)
        self.tree_ = evo_result.best_tree
        self.best_idx_ = int(evo_result.best_idx)
        self.n_features_in_ = V
        self.n_samples_seen_ = N
        # Preserve the public-API expression too, in case the user prefers it.
        self.fit_result_ = result
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X):
        """Evaluate the stored best tree (+ affine a, b) on new inputs."""
        if not hasattr(self, "tree_"):
            raise RuntimeError("EMLRegressor must be fit before calling predict().")

        X_t = _to_tensor(X)
        if X_t.dim() == 1:
            x_arg = X_t.unsqueeze(0)
        elif X_t.shape[1] == 1:
            x_arg = X_t.reshape(1, -1)
        else:
            x_arg = X_t.T.contiguous()

        V, N = x_arg.shape
        device = self.tree_.leaf_logits.device
        dtype = self.tree_.dtype

        # Broadcast the single-sample input across the batch dim to match
        # the tree's fixed population size.
        B = self.tree_.num_trees
        x_pop = x_arg.to(device=device, dtype=dtype).unsqueeze(0).expand(B, V, N).contiguous()

        with torch.no_grad():
            preds_all = self.tree_(x_pop)          # (B, N)
        pred = preds_all[self.best_idx_]           # (N,)

        if pred.is_complex():
            pred = pred.real
        pred = pred.to(torch.float32)
        out = self.a_ + self.b_ * pred
        return out.cpu().numpy()

    # Introspection used by the fallback BaseEstimator
    def _param_names(self):
        return ["depth", "strategy", "population", "generations",
                "device", "r2_target"]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _to_tensor(a) -> torch.Tensor:
    if torch.is_tensor(a):
        return a.to(torch.float32)
    return torch.as_tensor(np.asarray(a), dtype=torch.float32)


__all__ = ["EMLRegressor"]
