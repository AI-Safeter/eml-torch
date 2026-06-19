"""emltorch.certify — portable, dual-solver certificates of attention-head
concentration for any HuggingFace causal LM.

This subpackage promotes the previously script-trapped SMT cert machinery
(verify_z3/verify_cvc5 duplicated across 31 scripts; the _cert_v3 builder) into
a reusable, tested library API.

Scope honesty: a concentration certificate proves a property of *observed
attention scores under an L_inf box*, NOT model-level robustness. SMT verifies
the cert TEXT; only a full-forward PGD check establishes model soundness (H15).
"""

from __future__ import annotations

from .solvers import (
    SolverBackend,
    Z3Backend,
    CVC5Backend,
    SolverResult,
    DualResult,
    dual_verify,
)
from .concentration import attention_concentration_cert, num_to_smt
from .extract import (
    HeadScores,
    extract_head_logprob_scores,
    extract_all_heads_logprob_scores,
)
from .atlas import (
    CertifiedRadius,
    certified_radius,
    AttentionCertAtlas,
    AtlasResult,
)

__all__ = [
    "SolverBackend",
    "Z3Backend",
    "CVC5Backend",
    "SolverResult",
    "DualResult",
    "dual_verify",
    "attention_concentration_cert",
    "num_to_smt",
    "HeadScores",
    "extract_head_logprob_scores",
    "extract_all_heads_logprob_scores",
    "CertifiedRadius",
    "certified_radius",
    "AttentionCertAtlas",
    "AtlasResult",
]
