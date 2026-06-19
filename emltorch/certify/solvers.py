"""Unified SMT solver backends for portable certificate verification.

Replaces the 31 copy-pasted ``verify_z3`` / ``verify_cvc5`` pairs scattered
across sae-eml scripts with a single tested abstraction. Callers pass a plain
SMT-LIB2 string; the cvc5 file-only InputParser quirk is hidden inside
``CVC5Backend``.

Verdicts are normalized to lowercase ``"unsat"`` / ``"sat"`` / ``"unknown"``.
A missing solver yields ``"error:NotInstalled"`` rather than a silent skip, and
a z3/cvc5 disagreement is surfaced as a first-class ``DualResult.agree=False``.
"""

from __future__ import annotations

import os
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SolverResult:
    """Outcome of a single solver run."""

    verdict: str  # "unsat" | "sat" | "unknown" | "error:<Type>"
    elapsed_s: float
    solver: str


@dataclass
class DualResult:
    """Outcome of running both solvers on the same cert text."""

    z3: SolverResult
    cvc5: SolverResult
    agree: bool
    verdict: str  # agreed verdict, or "disagree"


class SolverBackend(ABC):
    """A portable SMT-LIB2 verifier."""

    name: str = "abstract"

    @abstractmethod
    def verify(self, smt2_text: str, timeout_ms: int = 30000) -> SolverResult: ...


def _normalize(raw: str) -> str:
    raw = raw.strip().lower()
    if raw in ("unsat", "sat", "unknown"):
        return raw
    return "unknown"


class Z3Backend(SolverBackend):
    """z3 via in-memory ``parse_smt2_string`` (no temp file needed)."""

    name = "z3"

    def verify(self, smt2_text: str, timeout_ms: int = 30000) -> SolverResult:
        t0 = time.time()
        try:
            import z3
        except ImportError:
            return SolverResult("error:NotInstalled", 0.0, self.name)
        try:
            solver = z3.Solver()
            solver.set("timeout", int(timeout_ms))
            solver.add(z3.parse_smt2_string(smt2_text))
            verdict = _normalize(str(solver.check()))
        except Exception as exc:  # malformed cert, solver crash, etc.
            return SolverResult(
                f"error:{type(exc).__name__}", time.time() - t0, self.name
            )
        return SolverResult(verdict, time.time() - t0, self.name)


class CVC5Backend(SolverBackend):
    """cvc5 via the file-only ``InputParser`` API.

    cvc5 has no ``parse_smt2_string`` equivalent, so we write the cert to a
    temp file and drive the command loop. This quirk is hidden from callers.
    """

    name = "cvc5"

    # The portable EML certs carry 9 quantified Exp/Ln axioms with :pattern
    # triggers. cvc5 does NOT instantiate them under its default config (returns
    # "unknown"), unlike z3's E-matching. full-saturate-quant turns on the
    # instantiation that lets cvc5 discharge the same certs z3 does -- without
    # it, "dual-verify" silently degrades to z3-only. This is baked in so the 31
    # ex-duplicated call sites can't each forget it.
    QUANT_OPTIONS = {"full-saturate-quant": "true"}

    def verify(self, smt2_text: str, timeout_ms: int = 30000) -> SolverResult:
        t0 = time.time()
        try:
            import cvc5
        except ImportError:
            return SolverResult("error:NotInstalled", 0.0, self.name)

        path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".smt2", delete=False
            ) as fh:
                fh.write(smt2_text)
                path = fh.name

            solver = cvc5.Solver()
            try:
                solver.setOption("tlimit-per", str(int(timeout_ms)))
                for key, val in self.QUANT_OPTIONS.items():
                    solver.setOption(key, val)
            except Exception:
                pass

            parser = cvc5.InputParser(solver)
            parser.setFileInput(cvc5.InputLanguage.SMT_LIB_2_6, path)
            sm = parser.getSymbolManager()

            verdict = "unknown"
            while True:
                cmd = parser.nextCommand()
                if cmd.isNull():
                    break
                out = cmd.invoke(solver, sm).strip()
                if out in ("sat", "unsat", "unknown"):
                    verdict = out
                    break
            verdict = _normalize(verdict)
        except Exception as exc:
            return SolverResult(
                f"error:{type(exc).__name__}", time.time() - t0, self.name
            )
        finally:
            if path and os.path.exists(path):
                os.unlink(path)
        return SolverResult(verdict, time.time() - t0, self.name)


def dual_verify(
    smt2_text: str,
    timeout_ms: int = 30000,
    backends: list[SolverBackend] | None = None,
) -> DualResult:
    """Run z3 and cvc5 on the same cert; report both verdicts and agreement.

    Disagreement is never swallowed — ``agree=False`` and ``verdict="disagree"``
    make a solver mismatch a loud, inspectable outcome.
    """
    if backends is None:
        backends = [Z3Backend(), CVC5Backend()]
    by_name = {b.name: b.verify(smt2_text, timeout_ms) for b in backends}
    z3_res = by_name.get("z3", SolverResult("error:NotRun", 0.0, "z3"))
    cvc5_res = by_name.get("cvc5", SolverResult("error:NotRun", 0.0, "cvc5"))
    agree = z3_res.verdict == cvc5_res.verdict
    return DualResult(
        z3=z3_res,
        cvc5=cvc5_res,
        agree=agree,
        verdict=z3_res.verdict if agree else "disagree",
    )
