"""
emltorch — GPU-batched symbolic regression via the EML operator.

The EML operator `eml(x, y) = exp(x) - ln(y)`, combined with the constant 1,
can represent every elementary function as a uniform binary tree.

Reference: Andrzej Odrzywolek, "All elementary functions from a single binary
operator", arXiv:2603.21852 (2026).

Quick start:

    import torch
    import emltorch as eml

    x = torch.linspace(0.5, 5.0, 512)
    y = torch.log(x)

    result = eml.fit(x, y, depth=3, population=1024)
    print(result.expression)    # '+0.0000 + (+1.0000) * [eml(1, eml(eml(1, x), 1))]'
    print(result.r2)            # 1.0
"""

__version__ = "0.2.0"

from .operator import safe_eml
from .tree import BatchedEMLTree
from .trainer import EMLTrainer, EMLConfig, EMLResult
from .symbolic import extract_expressions, annotate
from .hybrid import HybridEMLTrainer, HybridConfig, HybridResult
from .evolution import EvolutionConfig, EvolutionResult, evolve

from .api import fit, FitResult
from . import interp  # noqa: F401
from .gradient import diff_formula, gradient_at, sensitivity_vector, torch_gradient_fn

__all__ = [
    "__version__",
    # Public, stable API
    "fit",
    "FitResult",
    "interp",
    # Symbolic differentiation
    "diff_formula",
    "gradient_at",
    "sensitivity_vector",
    "torch_gradient_fn",
    # Lower-level (may change in 0.x)
    "safe_eml",
    "BatchedEMLTree",
    "EMLTrainer",
    "EMLConfig",
    "EMLResult",
    "HybridEMLTrainer",
    "HybridConfig",
    "HybridResult",
    "EvolutionConfig",
    "EvolutionResult",
    "evolve",
    "extract_expressions",
    "annotate",
]
