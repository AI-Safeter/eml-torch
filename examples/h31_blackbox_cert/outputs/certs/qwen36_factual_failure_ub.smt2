; H31 qwen36_factual_failure_ub
; Formula: +0.5954 + (-0.1353) * [eml(x2, eml((x2 - x4), 1))]
; Var ranges: {'x2': (0.05, 0.25), 'x4': (3.522344970703125, 4.355861663818359)}
; Claim (SAFE):  formula > 0.1
; Interval-propagation analytic bound: formula ∈ [-1.609117e-01, 1.041475e-02]
; UNSAT below proves SAFE for all variable assignments in the ranges.
; Logic: QF_LRA  (no transcendentals — interval arithmetic + linear arith only)
(set-logic QF_LRA)
(declare-const x2 Real)
(assert (>= x2 0.050000000000000003))
(assert (<= x2 0.250000000000000000))
(declare-const x4 Real)
(assert (>= x4 3.522344970703124911))
(assert (<= x4 4.355861663818359375))
; ─── Per-node Exp/Ln intervals + structural equalities ───
; Exp((- x2 x4))  L ∈ [-4.305862e+00, -3.272345e+00]
(declare-const exp_L_1 Real)
(assert (>= exp_L_1 0.013489256314250945))
(assert (<= exp_L_1 0.037917408569972666))
; Ln(1.000000000000000000)  R ∈ [1.000000e+00, 1.000000e+00]
(declare-const ln_R_2 Real)
(assert (>= ln_R_2 (- 0.000000001999999972)))
(assert (<= ln_R_2 0.000000002000000082))
(declare-const eml_3 Real)
(assert (= eml_3 (- exp_L_1 ln_R_2)))
; Exp(x2)  L ∈ [5.000000e-02, 2.500000e-01]
(declare-const exp_L_4 Real)
(assert (>= exp_L_4 1.051271094324752964))
(assert (<= exp_L_4 1.284025418971767030))
; Ln(eml_3)  R ∈ [1.348926e-02, 3.791741e-02]
(declare-const ln_R_5 Real)
(assert (>= ln_R_5 (- 4.305861814084508410)))
(assert (<= ln_R_5 (- 3.272344915956904909)))
(declare-const eml_6 Real)
(assert (= eml_6 (- exp_L_4 ln_R_5)))
; Negation of SAFE
(assert (not (> (+ 0.595400000000000040 (* (- 0.135300000000000004) eml_6)) 0.100000000000000006)))
(check-sat)
