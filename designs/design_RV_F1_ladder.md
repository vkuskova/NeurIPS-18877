# Design RV-F1: M2 on the realized-volatility panel — decisive baseline:
# interaction-augmented pooled linear

Status: DRAFT — locks on Valya's sign-off. No M2 training on RV data occurs
before lock. Screen RV-S1 (both gates run 2026-07-25; Gate 1 PASS, Gate 2
PARTIAL PASS) is frozen as this design's motivation and is not re-run.

## 1. Hypothesis under test — stated honestly

This is NOT a test of the submitted paper's cancellation-recovery mechanism.
RV-S1 established that the RV panel's heterogeneity is amplification-type
(spillover coefficients rise ~3× from calm to crisis; nothing reverses sign),
so pooling loses little here and hard stratification already fails
(strat-vs-pooled negative, 7/8 assets). The live hypothesis is narrower:

  Does M2's soft-gated architecture extract regime-varying structure better
  than the linear first-order representation of the same structure?

That first-order representation — coefficients varying linearly in the gate
variable — is the interaction-augmented pooled linear model (§3), and it is
the decisive baseline. Beating plain pooled linear while losing to the
interaction model means "add one interaction term," and the design says so in
advance. The rebuttal reports the outcome at whichever level it lands (§6).

Secondary purpose, valuable regardless of the predictive outcome: audit the
contraction-certificate constants on M2 trained on a real panel with genuine
headroom and strong heterogeneity — a direct empirical answer to kDiP Q5
("a real panel where the constants are non-trivial"), whichever way the
constants come out.

## 2. Fixed inputs (inherited from RV-S1, not renegotiable)

- Data: `/content/drive/MyDrive/GNAVAR/data/rv_dataset.csv`, SHA-256
  `9f6065e3...a0098c` (full hash in RV-S1 meta; re-verified at run time).
  8 indices, T = 2,615 days, wide format.
- Transforms: y = log(vol); target Δy_t; p = 4 own-vol lags; spillover
  covariate X = leave-one-out cross-asset mean of y, 4 lags; usable days
  2,611; test = final 523 days (identical boundary all assets;
  4,184 test obs); training = 2,088 days (16,704 obs).
- Regimes: R = 3, pooled training tertiles of y_{t-1}; cut points as computed
  and stored in RV-S1 meta. Test-era regime counts 2,919 / 989 / 276 —
  the calm-skew and the low-N regime-2 cell are recorded up front; OVERALL is
  the primary scope, per-regime results are secondary.
- Inference: moving-block bootstrap over days, block 25, B = 9,999,
  studentized, RNG seed 20260810 (new). Secondary: per-asset Newey–West
  (25 lags) DM, all 8 reported. Same machinery as RV-S1.

## 3. Models

Deterministic baselines (no seeds), trained on the training portion:
- B0 RW: Δy = 0.
- B1 Pooled linear: Δy on [4 own lags, 4 spillover lags] + asset intercepts
  (identical to RV-S1 Gate 1; refit, must reproduce the screen's test MSE
  0.074620 to 6 dp as a cross-check).
- B2 Interaction-augmented pooled linear (DECISIVE): B1's regressors plus
  ψ_it × each of the 8 base regressors, where ψ_it = y_{i,t-1} — the same
  scalar state summary M2's gate uses. 16 slopes + asset intercepts. This is
  the linear first-order approximation of gate-modulated coefficients, using
  the same conditioning variable as M2, which is what makes it the fair
  decisive comparator.
- B3 Stratified linear: per-regime B1 (context only; already loses to B1 in
  the screen).

M2 (PANAR-X, ported):
- Architecture and training code ported from the NeurIPS-18877 reproducibility
  bundle (`code/vdem/vdem_v1h_v3_cell.py`, `code/common/NAVAR.py`), headline
  configuration mode: unconstrained soft gating (soft_only, λ_M = 0), R = 3,
  tempered softmax over ψ = y_{t-1}, regime centers at the training tertile
  midpoints (ψ quantiles 1/6, 1/2, 5/6), per-regime scalar bias (as in the
  bundle's M2 code — correction logged 2026-07-25: the draft said per-unit
  intercepts, which the bundle's M2 does not have; the port is faithful to the
  code), covariate channel = the 4 spillover
  lags. Input dimensions adapted to this panel; nothing else redesigned.
- Hyperparameter selection, pre-registered to avoid both blind porting and
  forking paths: a fixed 2×2 grid — learning rate {1e-3, 3e-4} × hidden width
  {16, 32} — evaluated on a validation split (final 20% of TRAINING days;
  test data untouched), 5 seeds per cell, selected by mean pooled validation
  MSE. The selected cell is then retrained on the full training set, 5 seeds.
  Selection rule is deterministic; validation MSEs for all 4 cells are
  reported in meta. No other tuning of any kind.
- Prediction objects: M2 [ENS] = 5-seed mean (primary; fair against
  deterministic baselines per the 18877-R1 §3 reasoning — no baseline-side
  variance to ensemble away). M2 [PR] = per-seed; consistency gate: all 5
  per-seed statistics reported, <4/5 sign agreement triggers a verbatim caveat.

Certificate audit (reporting item, not a gate): on the 5 final trained seeds,
gradient-probe estimates of L_eq,max, M, L_φ,eq·M, B_trans and the A3 dwell
condition, using the bundle's audit code unchanged. Reported alongside the
V-Dem values (L_eq,max ≈ 0.29, A3 vacuous) whichever way they land.

## 4. Test families

d_i = e²_baseline,i − e²_M2,i unless stated; + favors the second-listed model.

- F1 (DECISIVE): M2 [ENS] vs B2 interaction-augmented linear — overall +
  3 regimes.
- F2: M2 [ENS] vs B1 pooled linear — overall + 3 regimes.
- F3 (context): B2 vs B1 — does the interaction itself pay out of sample? If
  B2 ≤ B1, the amplification found by Gate 2 is not exploitable OOS by the
  linear form, which reframes F1 (a null there would then be unsurprising and
  the informative comparison is F2).
- F4 (context): M2 [ENS] vs RW.
Holm across the 3 per-regime tests within F1 and within F2; each overall test
stands alone at α = 0.05. F3/F4 unadjusted, labeled context. Regime-2 cells
(276 obs) carry an explicit low-power flag in the output table.

## 5. Compute plan

Colab T4 (Pro recommended). Estimated: 20 validation runs + 5 final runs at
~2–4 min each in unpenalized soft mode ≈ 1.5–2.5 h, plus seconds for
baselines, bootstrap, and audit probes. Single notebook, Drive-mounted;
intermediate checkpoints written to the output directory so a session reset
does not force silent re-runs (any resume is logged in meta).

## 6. Pre-specified claim levels for the rebuttal

- Level A — F1 overall significant in M2's favor: soft gating exceeds the
  linear first-order representation of the same regime structure on a panel
  with real headroom. Strongest available architecture defense; goes into the
  rebuttal as new evidence with the full protocol attached.
- Level B — F2 overall significant in M2's favor but F1 not: regime-varying
  structure helps and M2 captures it, but a single pre-specified interaction
  captures it as well. Reported as such; rebuttal uses it as taxonomy evidence
  plus the certificate audit, not as an architecture win.
- Level C — neither: consistent with the NDFM finding that the nonlinear
  architecture does not pay over strong linear baselines under fair
  comparison. Reported as such; the rebuttal's empirical contribution is then
  the two-panel taxonomy, the corrected V-Dem null, and the certificate audit.
Per-regime results modulate the narrative but cannot upgrade the level on
their own (overall is primary).

## 7. Stopping rule

One execution after lock: validation selection, final training, tests, audit —
all in that single execution. No re-tuning, no added seeds, no alternative
inference, no re-runs of any unwelcome number. Checkpoint-resume after a
session reset is permitted and logged; it re-uses stored weights, never
retrains a completed run.

## 8. Outputs

To /content/drive/MyDrive/NeurIPS2026_3/rv_f1/:
- `results_RVF1_dm_tests.csv` — one row per test (family, scope, object tag,
  N, MSEs, mean d, bootstrap-t, p_boot, per-asset HAC DMs, per-seed sign
  agreement, Holm p, low-power flag).
- `results_RVF1_audit_constants.csv` — per-seed certificate constants + V-Dem
  reference row.
- `results_RVF1_meta.json` — data hash, config grid + validation MSEs +
  selected cell, seeds, RNG seeds, B, block length, runtime, any resume
  events.
- `rvf1_checkpoints/` — per-run model weights and per-observation test
  predictions (the canonical artifacts every reported number traces to).

## 9. Blocking items (Valya)

1. Sign-off on B2's definition (ψ × all 8 regressors; ψ = y_{t-1}).
2. Sign-off on the pre-registered 2×2 hyperparameter grid and
   validation-selection rule.
3. Sign-off on claim levels (§6) — they pre-commit what the rebuttal may say.
4. Confirm rebuttal-window feasibility of the compute plan on your Colab tier.
