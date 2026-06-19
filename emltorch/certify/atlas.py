"""Certified-radius search and head certification.

The honest deliverable: for a head's attention row, find the largest L_inf box
radius ``rho`` at which softmax concentration ``softmax_target > tau`` is
provable by BOTH solvers. A genuinely concentrated head yields ``rho* > 0``; a
non-concentrated head (target prob < tau) yields ``rho* = 0`` -- the search is
sound and non-vacuous by construction (it uses the shift-invariant
``softmax_interval`` form).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .concentration import attention_concentration_cert
from .extract import extract_all_heads_logprob_scores
from .solvers import dual_verify

_DEFAULT_RHOS = (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)


@dataclass
class CertifiedRadius:
    radius: float  # max rho with dual-UNSAT softmax-concentration; 0.0 if none
    tau: float
    target_idx: int
    verdicts: dict = field(default_factory=dict)  # rho -> "unsat"/"sat"/"disagree"


def certified_radius(
    scores: Sequence[float],
    target_idx: int,
    tau: float,
    rhos: Sequence[float] = _DEFAULT_RHOS,
    timeout_ms: int = 8000,
) -> CertifiedRadius:
    """Largest rho (from ``rhos``, tried high->low) at which both solvers prove
    ``softmax_target > tau`` over the L_inf(rho) box. Returns 0.0 if none."""
    verdicts: dict = {}
    best = 0.0
    for rho in sorted(rhos, reverse=True):
        cert = attention_concentration_cert(
            scores, target_idx, tau=tau, rho_box=rho, form="softmax_interval"
        )
        dual = dual_verify(cert, timeout_ms=timeout_ms)
        verdicts[rho] = dual.verdict
        if dual.verdict == "unsat" and dual.agree and best == 0.0:
            best = rho
    return CertifiedRadius(
        radius=best, tau=tau, target_idx=target_idx, verdicts=verdicts
    )


@dataclass
class AtlasResult:
    """Per-head certified-concentration records for one prompt on one model."""

    model_name: str
    prompt: str
    tau: float
    query_pos: int
    records: list = field(default_factory=list)  # list of per-head dicts

    def to_json(self) -> dict:
        """JSON-serializable view (all values are str/int/float/list)."""
        return {
            "model_name": self.model_name,
            "prompt": self.prompt,
            "tau": self.tau,
            "query_pos": self.query_pos,
            "records": [
                {
                    "layer": int(r["layer"]),
                    "head": int(r["head"]),
                    "target_idx": int(r["target_idx"]),
                    "target_prob": float(r["target_prob"]),
                    "certified_radius": float(r["certified_radius"]),
                    "attends_to_token": str(r["attends_to_token"]),
                }
                for r in self.records
            ],
        }


class AttentionCertAtlas:
    """Sweep all (layer, head) of a HF causal LM on one prompt and certify each
    head's softmax-concentration radius via the honest ``softmax_interval`` form.

    Efficiency: the model is run forward ONCE (output_attentions=True) and the
    attentions are reused across all heads (no per-head re-forward). Each head
    uses its OWN argmax key as the cert target. ``target_prob`` is recorded so
    a vacuous (non-concentrated) head cannot hide -- it gets radius 0.0.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        prompt: str = "",
        tau: float = 0.5,
        rhos: Optional[Sequence[float]] = None,
        query_pos: int = -1,
        timeout_ms: int = 8000,
    ):
        self.model_name = model_name
        self.device = device
        self.prompt = prompt
        self.tau = tau
        self.rhos = tuple(rhos) if rhos is not None else _DEFAULT_RHOS
        self.query_pos = query_pos
        self.timeout_ms = timeout_ms

    def _load(self):
        import transformers
        import torch

        tok = transformers.AutoTokenizer.from_pretrained(self.model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name, attn_implementation="eager", dtype=torch.float32
        ).eval()
        if self.device != "cpu":
            model = model.to(self.device)
        return model, tok

    def run(self, model=None, tokenizer=None) -> AtlasResult:
        """Certify every head. Optionally pass an already-loaded model+tokenizer."""
        if model is None or tokenizer is None:
            model, tokenizer = self._load()

        head_rows = extract_all_heads_logprob_scores(
            model, tokenizer, self.prompt, device=self.device, query_pos=self.query_pos
        )
        records = []
        for hs in head_rows:
            cr = certified_radius(
                hs.scores,
                hs.argmax_idx,
                tau=self.tau,
                rhos=self.rhos,
                timeout_ms=self.timeout_ms,
            )
            records.append(
                {
                    "layer": hs.layer,
                    "head": hs.head,
                    "target_idx": hs.argmax_idx,
                    "target_prob": hs.target_prob,
                    "certified_radius": cr.radius,
                    "attends_to_token": hs.tokens[hs.argmax_idx],
                }
            )
        return AtlasResult(
            model_name=self.model_name,
            prompt=self.prompt,
            tau=self.tau,
            query_pos=self.query_pos,
            records=records,
        )
