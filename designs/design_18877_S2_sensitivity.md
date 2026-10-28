# Design 18877-S2: Sensitivity sweep — regime definition, state summary,
# cut points, and sign-reversal stability

Status: DRAFT — locks on Valya's sign-off. Part B (GPU) does not start before
lock. Answers metareview items: "sensitivity to the number and definition of
regimes, the choice of state summary and cut points, and whether the
sign-reversal pattern is stable across samples" (also Bxp5 Q3, kDiP Q4,
UNwV Q3–Q4).

## 1. Purpose and the no-upgrade rule

The forecasting claim is withdrawn (18877-R1, claim level C); this sweep does
not attempt to resurrect it. Its two legitimate purposes:

1. Robustness of the paper's surviving asset — the sign-reversal diagnostic —
   across regime definitions, state summaries, cut points, samples, and
   covariate subsets.
2. Robustness of the corrected null — M2 ≈ stratified linear ≈ RW under
   clustered inference — to the same regime-definition choices, i.e., showing
   the null is not itself an artifact of R = 3 tertiles of y_{t-1}.

NO-UPGRADE RULE, pre-committed: if any sweep variant shows a nominally
significant M2 advantage, it is reported with family-wise correction across
all variants tested, explicitly labeled exploratory, and does NOT enter the
rebuttal as a rescued forecasting claim. A sweep run after seeing a null
cannot manufacture a confirmatory result; the design says so and the rebuttal
will say so.

## 2. Fixed inputs

- Data: the bundle's canonical CSV (on Drive:
  NeurIPS2026_3/data/merged_completevars_balanced_zero_missing.csv), same
  pipeline constants as the submission: Y = v2xel_frefair, X = {v2csantimv,
  v2jureform, v2psorgs, v2mecorrpt}, L = 5, ΔY target, last-10-years-per-
  country test split, X standardized on training only. Diagnostic code ported
  from the bundle's `regime_stratified_diagnostic.py`; M2 code as in RV-F1's
  port (soft_only, λ_M = 0 headline mode).
- Inference: country-block bootstrap-t, B = 9,999 (18877-R1 machinery),
  RNG seed 20260815. Cluster = country throughout.
- Headline diagnostic pattern being tracked, from the submission: v2psorgs at
  lag t−4 — pooled ≈ −0.000 vs per-regime −0.019 / +0.018 / +0.016.

## 3. Part A — linear sensitivity (CPU, minutes)

A1. Diagnostic grid. Full crossing of:
- State summary ψ: ψ1 = y_{t-1} (submission); ψ2 = mean(y_{t-1..t-3})
  (smoothed level); ψ3 = first principal component of standardized
  (y_{t-1}, four X_{t-1}) computed on training (the "learned low-dimensional
  representation" UNwV asks about, in its simplest honest form).
- Number of regimes R ∈ {2, 3, 4, 5}, equal-quantile cuts on training ψ.
- Cut perturbation (R = 3 only, per ψ): interior cut points shifted ±5
  percentile points (four extra variants per ψ).
Per variant, report: per-regime coefficient table for all X×lag slopes;
sign-reversal inventory between the outermost regimes (opposite signs, both
|coef| > 2 cluster-robust SEs); whether the v2psorgs t−4 pattern
(pooled-collapse + outer-regime activity) is present; stratified-linear and
pooled-linear test MSE under that regime routing.

A2. Stability across samples, canonical spec (ψ1, R = 3, tertiles):
- (a) Era splits: refit the diagnostic on the training era split in halves,
  and on three rolling windows matching the kdd_v1 origin windows
  (1996–2005, 2006–2015, 2011–2020 test decades with their preceding
  training eras) so the linear evidence lines up with existing M2 artifacts.
- (b) Country jackknife: 200 random half-country subsamples (seeded);
  report, per key slope, the fraction of subsamples preserving its canonical
  sign, and the fraction in which the full reversal pattern survives.
- (c) Leave-one-covariate-out: refit with each X dropped in turn; report
  whether the remaining reversals persist (kDiP's covariate-set concern).

A3. Stratification gain under clustered inference (new pre-specified test —
the submission's +3.02% pooled→stratified improvement was never given a
p-value, and 18877-R1 tested stratified-vs-RW but not stratified-vs-pooled):
country-block bootstrap-t for stratified vs pooled linear, overall + per
regime, at the canonical spec; point estimates only (no tests) for the other
A1 variants. Expectation stated in advance: given C3's p = 0.83 for
stratified-vs-RW, this is likely null too, and reporting it null is the
honest completion of the reviewers' question.

## 4. Part B — M2 sensitivity (GPU, ~70–100 min)

Retrain M2 (5 seeds each, headline soft_only mode, per-seed + [ENS] objects)
in exactly four variants — chosen to answer the reviewers' axes at minimum
compute, no more:
- B-R2: R = 2 (ψ1, median cut; centers at within-half medians)
- B-R4: R = 4 (ψ1, quartile cuts)
- B-PSI2: ψ2 smoothed summary (R = 3)
- B-CUT: interior cuts shifted +5pp (R = 3, ψ1)
Per variant: M2 [ENS] vs the variant-matched stratified linear, and vs RW,
overall test set, country-block bootstrap-t; per-seed sign gates as in R1.
Per-regime numbers reported descriptively (regime counts change across
variants; per-regime tests would be underpowered and multiplicity-heavy —
excluded by design). Holm correction across the four variants within each
comparison family. Per-observation predictions saved per run (closing the
frontier artifact's known gap). The canonical R = 3 result is taken from
18877-R1's outputs, not re-run.

## 5. What the rebuttal may say, by outcome

- Diagnostic stable across A1/A2 (expected for at least the ψ1 family): the
  paper's core finding is robust to R, cut points, samples, and covariate
  subsets; reported with the variant tables attached.
- Diagnostic unstable somewhere: reported exactly where and how; if the
  v2psorgs pattern is fragile (e.g., era-specific), the diagnostic claim is
  qualified accordingly in the rebuttal — before a reviewer finds it.
- Part B null across variants (expected): the corrected equivalence
  M2 ≈ stratified ≈ RW is not an artifact of the submission's specific regime
  definition — which converts the reviewers' sensitivity question from a
  threat into support for the corrected framing.
- Any Part B significance: exploratory label + Holm, per §1. No exceptions.

## 6. Stopping rule and outputs

One execution per part after lock (Part A and Part B may run in separate
sessions; B uses checkpoint-resume as in RV-F1). No variant additions after
lock, no re-runs of unwelcome numbers, deviations logged.

To /content/drive/MyDrive/NeurIPS2026_3/rebuttal_S2/:
- `results_S2_A1_diagnostic_grid.csv` (one row per variant × slope)
- `results_S2_A2_stability.csv` (era / jackknife / LOCO summaries)
- `results_S2_A3_stratification_tests.csv`
- `results_S2_B_m2_variants.csv`
- `results_S2_meta.json` + `s2_checkpoints/`

## 7. Blocking items (Valya)

1. The ψ set (§3 A1) — in particular ψ3 as the "learned representation"
   response to UNwV.
2. The four Part B variants (§4) — the compute/coverage tradeoff is set at
   four; adding variants adds ~18 min each and multiplicity.
3. The no-upgrade rule (§1) and the A3 expectation statement (§3) — both
   pre-commit rebuttal language.
4. Part B GPU window (~70–100 min on T4, separate from the RV-F1 session).
