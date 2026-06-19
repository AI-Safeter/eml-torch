import Lake
open Lake DSL

package «eml-axioms» where
  -- Lean 4 package for EML SMT axiom soundness witness.
  -- See EmlAxioms/Basic.lean for theorems.

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib «EmlAxioms» where
  -- Single module; mirrors emltorch.smt.EML_AXIOMS_SMT2 axiom block.
