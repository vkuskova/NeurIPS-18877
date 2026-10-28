# Design 18877-R1: M2 vs. stratified linear and random walk under panel-clustered inference

Status: DRAFT — locks on Valya's sign-off. No code is written or run before lock.
Role: This experiment decides the rebuttal. Its outcome determines which of three
pre-specified claim levels (A/B/C, §7) the response to NeurIPS #18877 can honestly make.

---

## 1. Question

Does the regime-aware neural architecture (M2) produce lower squared forecast error
than (a) the regime-stratified linear model and (b) the random walk, on the same test
observations, under inference that respects country-panel dependence — overall and
per regime?

Honesty note for the record: the direction of the hypotheses (tail-regime gains) is
informed by already-printed seed-mean point estimates (Table 1 vs. Table 2 of the
submission). This protocol is therefore not a pre-registration of unseen outcomes; it
is a fixed analysis plan locked before the paired tests and clustered inference are
computed, which is what prevents forking-paths inference. The rebuttal will describe
it exactly this way.

## 2. Fixed inputs (canonical artifacts — names to be confirmed before lock)

All quantities are computed from stored per-observation predictions of the single
canonical run. Nothing is retrained.

Canonical source: `panar-x-reproducibility.tar.gz` from the anonymous repo
`github.com/author0725/NeurIPS-18877`, file
`artifacts/vdem_v1h_postanalysis.pkl`,
SHA-256 `018a8d24d7af18f8f3f85cdc4bed8ed54166c0e55dc9170f91df29478d762e0a`
(matches the bundle's own manifest; verified 2026-07-24). One file supplies all
inputs, mutually row-aligned:

- I1. Test-set frame: `targets` (ΔY), `regime_assignment`, `country_id`, `year`.
- I2. M2: `predictions['M2_per_seed']` (5 seeds, unconstrained soft cell);
  ensemble = `predictions['M2_mean']` (verified equal to the per-seed mean).
- I3. Stratified linear: `predictions['linear_stratified_M1.5']`.
- I4. Pooled linear (context): `predictions['linear_pooled_M1.5']`.
- I5. Extended-seed M2 predictions (20-seed audit set): NOT in the bundle; the
  seed-asymmetry audit postdates it. Included only if Valya points to Drive
  artifacts at lock; otherwise dropped without blocking.
- RW: `predictions['rw']` (verified identically zero, i.e., forecast ΔY = 0).
- C4 baseline: `predictions['NN_pooled_M1.5']` — a single stored prediction
  vector; per the internal audit this is the single-seed object behind the unfair
  headline, and C4 is reported with that caveat attached.
- Also present, not in scope unless Valya adds it: `predictions['NN_stratified_M1.5']`
  (stratified NN, apparently single-seed). It would isolate soft gating vs. hard NN
  stratification, but carries the same single-seed asymmetry problem against
  M2 [ENS]; default excluded.

Geometry verified pre-lock (read-only; no losses or differentials computed):
1,510 rows, 151 countries × exactly 10 test years, regime N = 252/621/637,
no NaNs. Any mismatch at run time halts pending diagnosis.

## 3. Prediction objects and fairness rationale

- M2 [ENS]: mean of the 5 canonical per-seed predictions. Primary object. This is
  legitimate against deterministic baselines, unlike the broken headline comparison:
  the pooled-NN comparison was unfair because the baseline *could* be ensembled and
  was not (seed asymmetry). Stratified linear and RW are deterministic — there is no
  baseline-side variance to ensemble away, so [ENS]-vs-deterministic compares
  procedures as deployed. The design doc and rebuttal state this reasoning explicitly.
- M2 [PR]: each seed separately. Consistency gate, not a headline: all 5 per-seed DM
  statistics are reported in full for every test. If an [ENS] result is significant
  but fewer than 4/5 seeds agree in sign, the result is reported with that caveat
  verbatim — no seed selection, no reweighting.
- Every reported number carries its [ENS]/[PR] tag. Mixed-object comparisons are
  prohibited.

## 4. Step 0 — recomputation before any testing

From I1–I4, recompute every MSE printed in submission Tables 1 and 2 (per regime and
overall, all comparators) and recompute the Table 2 Overall DM row from the stored
loss differentials. Reconciliation table: printed value vs. recomputed value, one row
per number. Discrepancies are findings, not things to fix silently; the duplicated
Overall statistic (+2.41, p=0.016, identical to Regime 2) is resolved here first.
If Step 0 changes any per-regime point estimate that motivated §1, the design still
runs unchanged — the tests are the point, not the point estimates.

## 5. Test families

Loss differential per observation i: d_i = e²_baseline,i − e²_M2,i (positive favors
M2, matching the submission's sign convention). Paired at the observation level.

- C1 (primary): M2 [ENS] vs. stratified linear — overall + regimes 0/1/2. 4 tests.
- C2 (primary): M2 [ENS] vs. RW — overall + regimes 0/1/2. 4 tests.
- C3 (context): stratified linear vs. RW — overall + regimes. Establishes whether
  anything on the ladder beats RW robustly (Bxp5's implicit question).
- C4 (context): M2 [ENS] vs. pooled NN M1.5 under the same clustered inference —
  reported for continuity with the submission's headline, with the internal audit
  finding (ensemble-fair comparison is null) stated alongside. Not a claim source.

Regime membership: the paper's fixed training-tertile routing (I1). Cluster = country
always, regardless of regime; a country's observations may split across regimes over
time, and each per-regime test clusters on the countries present in that regime.

## 6. Inference (pre-specified, in order of authority)

- Primary: country-block bootstrap of the mean loss differential, studentized
  (bootstrap-t). Resample 151 countries with replacement; each drawn country
  contributes its full within-country differential series intact (preserves serial
  dependence within country). B = 9,999. RNG seed fixed at 20260724. Two-sided
  p-values. Per-regime tests resample from the countries present in that regime.
- Secondary (reported alongside, never substituted if less favorable): cluster-robust
  t-test on country-level mean differentials, t reference with G−1 df.
- Multiplicity: within each family (C1, C2), Holm–Bonferroni across the m = 3
  per-regime tests, matching the submission's convention; the overall test in each
  family stands alone at α = 0.05. C3/C4 are context and reported unadjusted with
  that label.
- α = 0.05 two-sided throughout. No one-sided tests.

## 7. Pre-specified claim levels for the rebuttal

- Level A — C1 shows a Holm-significant tail-regime win (regime 0 and/or 2) AND C2
  is Holm-significant in the same regime(s): rebuttal claims the nonlinear
  architecture delivers localized gains beyond both strong baselines, concentrated
  where the sign-reversal diagnostic predicts; abstract headline is rewritten around
  this, overall-tie with RW stated plainly.
- Level B — significant vs. RW in the tails but not vs. stratified linear: rebuttal
  concedes the architecture matches but does not exceed stratification; the paper's
  forecasting contribution is reframed as regime structure (however implemented)
  being what matters, with the architecture's remaining value being the certificate
  and end-to-end soft gating. Weak position; acceptance odds drop accordingly.
- Level C — no Holm-significant result in C1 tails: the forecasting claim is
  withdrawn in the rebuttal, and Valya decides between defending as
  diagnostic-plus-theory or withdrawing to the KDD resubmission (registered design
  already exists). This design does not make that call.
Nominal-but-not-Holm results are reported as such and do not upgrade the level.

## 8. Stopping rule

One execution of this protocol after lock. No re-running with alternative inference
choices, bootstrap schemes, seeds, ensemble sizes, or regime routings in search of a
better level. The extended-seed robustness check (I5), if the artifacts exist, runs
in the same execution or not at all. Any deviation (e.g., a Step 0 halt) is logged
with reason before resuming.

## 9. Outputs

- `results_18877_R1_step0_reconciliation.csv` — printed vs. recomputed, all Table 1/2
  numbers plus the Overall DM row resolution.
- `results_18877_R1_dm_tests.csv` — one row per test: family, scope, object tag,
  N_obs, N_countries, mean d, bootstrap-t stat, p_boot, p_cluster_t, Holm-adjusted p,
  per-seed sign agreement (k/5).
- `results_18877_R1_meta.json` — artifact filenames used (I1–I5), RNG seed, B,
  timestamp, git-less provenance note.
All written to the Drive project folder; these three files are the sole source for
any number entering the rebuttal.

## 10. Lock status

Resolved from the reproducibility bundle (2026-07-24): artifact sources (§2, single
pickle, SHA-verified), M2 canonical object = 5-seed ensemble `M2_mean` from the
unconstrained soft cell (matches Table 2), cluster = country (test geometry is
151 countries × 10 years; country is the only defensible cluster), RW = stored
zero-forecast vector.

Remaining before lock — Valya:
1. Sign off on the claim-level definitions in §7 (they pre-commit the rebuttal's
   framing).
2. State whether I5 (20-seed audit predictions) and/or `NN_stratified_M1.5` enter
   as labeled robustness/context; default is both excluded.
3. Project root on Drive for the output files (e.g.
   /content/drive/MyDrive/NeurIPS2026_3/rebuttal_R1/).
