"""emltorch — GPU-batched symbolic regression via the EML operator.

The EML operator `eml(x, y) = exp(x) - ln(y)`, combined with the constant 1,
represents every elementary function as a uniform binary tree.

Reference: Andrzej Odrzywolek, "All elementary functions from a single binary
operator", arXiv:2603.21852 (2026).

Quick start:

    import torch
    import emltorch as eml

    x = torch.linspace(0.5, 5.0, 512)
    y = torch.log(x)

    result = eml.fit(x, y, depth=3, population=1024)
    print(result.expression)
    print(result.r2)
"""

__version__ = "0.7.1"

from .operator import safe_eml
from .tree import BatchedEMLTree
from .symbolic import extract_expressions, annotate
from .evolution import EvolutionConfig, EvolutionResult, evolve
from .polish import polish
from .api import (
    fit,
    fit_multi_seed,
    fit_residual_boost,
    fit_pareto,
    FitResult,
    MultiSeedResult,
    BoostedResult,
    ParetoResult,
    _expression_complexity,
)
from .smt import (
    SafetyCertificate,
    eml_formula_to_z3,
    certify_linear_threshold_safe,
    find_min_norm_witness,
    optimize_min_linf_witness,
    emit_smtlib2,
    emit_raw_weight_concentration_cert,
    eml_tree_to_smt2,
    eml_tree_to_smt2_intervals,
    EML_AXIOMS_SMT2,
    EML_LEMMAS,
    with_lemmas,
    softmax_jacobian_g1,
    softmax_jacobian_g1_max,
    attention_block_lipschitz_clean,
    attention_block_lipschitz_interval,
    emit_attention_lipschitz_smt2_block,
)

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
    "BatchedEMLTree",
    "evolve",
    "EvolutionConfig",
    "EvolutionResult",
    "polish",
    "extract_expressions",
    "annotate",
    # SMT / formal verification bridge
    "SafetyCertificate",
    "eml_formula_to_z3",
    "certify_linear_threshold_safe",
    "find_min_norm_witness",
    "optimize_min_linf_witness",
    "emit_smtlib2",
    "emit_raw_weight_concentration_cert",
    "eml_tree_to_smt2",
    "eml_tree_to_smt2_intervals",
    "EML_AXIOMS_SMT2",
    "EML_LEMMAS",
    "with_lemmas",
    # Attention-block Lipschitz primitives
    "softmax_jacobian_g1",
    "softmax_jacobian_g1_max",
    "attention_block_lipschitz_clean",
    "attention_block_lipschitz_interval",
    "emit_attention_lipschitz_smt2_block",
]
