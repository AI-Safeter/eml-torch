"""GPU-batched symbolic regression with the EML operator."""

__version__ = "0.7.1"

from .api import (
    BoostedResult,
    FitResult,
    MultiSeedResult,
    ParetoResult,
    fit,
    fit_multi_seed,
    fit_pareto,
    fit_residual_boost,
)
from .api import (
    _expression_complexity as _expression_complexity,
)
from .evolution import EvolutionConfig, EvolutionResult, evolve
from .head import EMLHead
from .operator import safe_eml
from .polish import polish
from .smt import (
    EML_AXIOMS_SMT2,
    eml_formula_to_z3,
    eml_tree_to_smt2,
    eml_tree_to_smt2_intervals,
)
from .symbolic import annotate, extract_expressions
from .tree import BatchedEMLTree

__all__ = [
    "__version__",
    # Core fit API
    "fit",
    "fit_multi_seed",
    "fit_residual_boost",
    "fit_pareto",
    "FitResult",
    "MultiSeedResult",
    "BoostedResult",
    "ParetoResult",
    # Building blocks
    "safe_eml",
    "EMLHead",
    "BatchedEMLTree",
    "evolve",
    "EvolutionConfig",
    "EvolutionResult",
    "polish",
    "extract_expressions",
    "annotate",
    # SMT / formal verification bridge
    "eml_formula_to_z3",
    "eml_tree_to_smt2",
    "eml_tree_to_smt2_intervals",
    "EML_AXIOMS_SMT2",
]
