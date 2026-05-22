; H31 qwen36_factual_working_lb
; Formula: +0.5954 + (-0.1353) * [eml(x2, eml((x2 - x4), 1))]
; Var ranges: {'x1': (5.0, 6.0), 'x2': (-0.1, 0.1), 'x3': (-0.1, 0.1), 'x4': (1.9390155673027039, 2.3998414278030396), 'x5': (9.596486735697948, 10.833047442591619)}
; Claim (SAFE):  formula > 0.1
; Interval-propagation analytic bound: formula ∈ [1.076418e-01, 2.241567e-01]
; UNSAT below proves SAFE for all variable assignments in the ranges.
; Logic: QF_LRA  (no transcendentals — interval arithmetic + linear arith only)
(set-logic QF_LRA)
(declare-const x2 Real)
(assert (>= x2 (- 0.100000000000000006)))
(assert (<= x2 0.100000000000000006))
(declare-const x4 Real)
(assert (>= x4 1.939015567302703857))
(assert (<= x4 2.399841427803039551))
; ─── Per-node Exp/Ln intervals + structural equalities ───
; Exp((- x2 x4))  L ∈ [-2.499841e+00, -1.839016e+00]
(declare-const exp_L_1 Real)
(assert (>= exp_L_1 0.082098014972444067))
(assert (<= exp_L_1 0.158973849313911481))
; Ln(1.000000000000000000)  R ∈ [1.000000e+00, 1.000000e+00]
(declare-const ln_R_2 Real)
(assert (>= ln_R_2 (- 0.000000001999999972)))
(assert (<= ln_R_2 0.000000002000000082))
(declare-const eml_3 Real)
(assert (= eml_3 (- exp_L_1 ln_R_2)))
; Exp(x2)  L ∈ [-1.000000e-01, 1.000000e-01]
(declare-const exp_L_4 Real)
(assert (>= exp_L_4 0.904837416131122230))
(assert (<= exp_L_4 1.105170920180818639))
; Ln(eml_3)  R ∈ [8.209801e-02, 1.589739e-01]
(declare-const ln_R_5 Real)
(assert (>= ln_R_5 (- 2.499841454164165810)))
(assert (<= ln_R_5 (- 1.839015552722018221)))
(declare-const eml_6 Real)
(assert (= eml_6 (- exp_L_4 ln_R_5)))
; Negation of SAFE
(assert (not (> (+ 0.595400000000000040 (* (- 0.135300000000000004) eml_6)) 0.100000000000000006)))
(check-sat)
