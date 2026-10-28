# PANAR-X Paper-to-Code Traceability Table

This document maps every numerical claim, table, figure, and audit constant
in the paper to the specific code cell, function, or line range that
produces it. Each entry includes:

- **Paper location**: section §, equation, table, or figure label, with
  line numbers in `paper/main.tex`
- **Claim**: the exact value or content reported in the paper
- **Artifact**: which saved pickle (in `artifacts/`) contains it
- **Source script + cell**: the script in `code/` and the function or
  section block that generates it
- **Verify line**: the line of `verify_paper_numbers.py` that asserts the
  value, when applicable

For full numerical detail, run `python verify_paper_numbers.py` from the
package root: 141 checks pass against the archived artifacts.

---

## Paper structure (post-trim, 20 pages)

The trimmed version has body §1–§7 (9 main pages) followed by Appendices
A–E (10 pages including References):

| Section | Paper (main.tex) line | Coverage |
|---|---|---|
| §1 Introduction | 78 | abstract, contributions |
| §2 Related work | 153 | NAVAR, switched systems, regime-switching panels |
| §3 Architecture | 215 | M1, M1.5, M2, soft gating |
| §4 Stability theorem | 300 | Theorem 1, derived constants |
| §5 Training and audit | 408 | training procedure, gradient probe |
| §6 V-Dem demonstration | 447 | diagnostic, baselines, sweep, stability |
| §7 Synthetic stress test | 669 | Pareto frontier headline, mechanism |
| §8 Limitations and conclusion | 775 | combined |
| App A Full proofs | 829 | derivations only, no numbers |
| App B Synthetic data and Pareto frontier | 1019 | DGP, Table B.1 (unconstrained), Table B.2 (full 45-run) |
| App C V-Dem data | 1131 | variable selection, sample, regime boundaries |
| App D Audit protocol | 1201 | gradient probe, certified bound |
| App E Hyperparameter sensitivity | 1321 | hyperparameters and runtime |

---

## Section 3: Architecture

The body has no numerical claims tied to artifacts; the architecture is
descriptive. Implementation:

| Architecture description | Source |
|---|---|
| M1.5: additive Y/X channels, no regime structure | `code/synthetic/synthetic_v1f_sweep.py` class `M15`, lines 133–148; `code/vdem/vdem_v1h_v3_cell.py` class `M15_PANAR`, lines 268–284 |
| M2: regime-aware soft mixture, per-edge nets per regime | `code/synthetic/synthetic_v1f_sweep.py` class `M2`, lines 160–199; `code/vdem/vdem_v1h_v3_cell.py` class `M2_RegimeAware`, lines 369–421 |

---

## Section 4: Stability theorem

§4 describes the constants and theorem statement. The numbers cited
appear at the §4 anchor paragraph (paper lines ~395–405) when previewing
the V-Dem audit:

| Paper claim | Value | Verify line |
|---|---|---|
| §4 V-Dem $\hat L_{\text{eq,max}} \approx 0.29$ (5-seed mean) | `0.29` | line 366 |
| §4 V-Dem $\hat B_{\text{trans}} \approx 0.30$ (5-seed mean) | `0.30` | line 368 |
| §4 V-Dem $\hat L_{\text{eq,max}} < 1$ (trivial-contraction precondition) | `True` | line 370 |
| §4 V-Dem $\hat B_{\text{trans}} \leq 1$ ((A3) vacuously satisfied) | `True` | line 372 |

**Source**: `code/vdem/vdem_v1h_v3_cell.py full_audit()` (lines 634–812)
called per (mode, $\lambda_M$, seed) inside the sweep loop. Aggregation
to 5-seed mean for the unconstrained-soft cell is in
`verify_paper_numbers.py` lines 362–365 (loads from
`artifacts/vdem_v1h_frontier.pkl`).

---

## Section 6: V-Dem demonstration

All §6 numbers come from `artifacts/vdem_v1h_frontier.pkl` and
`artifacts/vdem_v1h_postanalysis.pkl`, produced respectively by
`code/vdem/vdem_v1h_v3_cell.py` and `code/vdem/vdem_v1h_postanalysis.py`.

### §6 setup (paper lines 460–482)

V-Dem v15 panel, $Y$ = `v2xel_frefair`, $X$ = `{v2csantimv, v2jureform,
v2psorgs, v2mecorrpt}`, lag $L=5$, last 10 years per country reserved
for test.

| Paper claim | Value | Verify line |
|---|---|---|
| V-Dem unique countries | `151` | line 645 |
| V-Dem year range | `1970`–`2025` | lines 647, 649 |
| Train + test sample size after lag drop | `7,701` (= 6,191 + 1,510) | line 653 |

**Source**: `code/vdem/vdem_v1h_v3_cell.py build_panel_tensors()` (lines
108–146). Same logic mirrored in `vdem_v1h_postanalysis.py` (lines 80–144)
and `regime_stratified_diagnostic.py` (lines 66–113).

### §6.1 Diagnostic - Table 5.3 (`tab:vdem-strat`) (paper line 498)

**Paper location**: §6.1 body table, paper lines 498–516.

| Paper claim | Value | Verify line |
|---|---|---|
| Pooled linear M1.5 MSE (regime 0) | `0.00520` | line 230 (loop) |
| Pooled linear M1.5 MSE (regime 1) | `0.00487` | line 230 (loop) |
| Pooled linear M1.5 MSE (regime 2) | `0.00143` | line 230 (loop) |
| Pooled linear M1.5 MSE (overall) | `0.00347` | line 210 |
| Stratified linear M1.5 MSE (regime 0) | `0.00518` | line 232 (loop) |
| Stratified linear M1.5 MSE (regime 1) | `0.00465` | line 232 (loop) |
| Stratified linear M1.5 MSE (regime 2) | `0.00140` | line 232 (loop) |
| Stratified linear M1.5 MSE (overall) | `0.00337` | line 212 |
| Improvement strat over pool (overall) | `+3.02%` | line 219 |

**Source - primary**: `code/vdem/regime_stratified_diagnostic.py` (320
lines, top-level script). Per-regime fits at lines 171–215; overall pooled
vs stratified comparison at lines 217–280.

**Source - also computed in main sweep**: `code/vdem/vdem_v1h_v3_cell.py`
lines 197–254. Saved into `vdem_v1h_frontier.pkl['baselines']`.

### Figure 1 (`fig:diagnostic-recovery`) (paper line 518)

**Paper location**: §6.1, paper lines 518–542.

Two-row figure. Row 1: linear coefficients heatmap at lag $t-4$ across
pooled + 3 regimes (panel A); bar-chart focus on `v2psorgs` $X_{t-4}$
(panel B). Row 2: cumulative paired squared-error advantage of M2 over
pooled NN M1.5 across the test set (panels C/D/E).

| Paper claim | Value | Verify line |
|---|---|---|
| Panel B: pooled coef on `v2psorgs` $X_{t-4}$ | `≈ -0.000` | line 610 |
| Panel B: regime 0 coef | `-0.019` | line 602 (loop) |
| Panel B: regime 1 coef | `+0.018` | line 602 (loop) |
| Panel B: regime 2 coef | `+0.016` | line 602 (loop) |
| Panel C: regime 0 cumulative final | `+0.102` | line 444 (loop) |
| Panel D: regime 1 cumulative final | `+0.015` | line 444 (loop) |
| Panel E: regime 2 cumulative final | `+0.059` | line 444 (loop) |
| Panel C: regime 0 MSE reduction | `-7.3%` | line 424 (loop) |
| Panel D: regime 1 MSE reduction | `-0.5%` | line 424 (loop) |
| Panel E: regime 2 MSE reduction | `-6.3%` | line 424 (loop) |
| Abstract / §6.1 prose: "4 of 4 covariates show sign reversal" | `4` | line 634 |

**Source**: `code/vdem/build_figure.py`, full file 244 lines:
- Lines 19–34: load V-Dem postanalysis pickle and CSV
- Lines 39–72: rebuild panel from CSV
- Lines 74–77: parse coefficients into (5, 4) array (5 lags × 4 covariates)
- Lines 98–123: panel A - heatmap
- Lines 124–144: panel B - bar chart focused on `v2psorgs` at lag $t-4$
- Lines 145–238: panels C/D/E - cumulative-difference per regime,
  using `M2_mean` and `NN_pooled_M1.5` from the postanalysis pickle

### §6.2 Baselines (paper lines 554–571)

| Paper claim | Value | Verify line |
|---|---|---|
| RW test MSE | `0.00340` | line 208 |
| Pooled linear M1.5 vs RW | `−2.25%` (worse) | not asserted (arithmetic) |
| Pooled linear M1.5 MSE | `0.00347` | line 210 |
| Stratified linear M1.5 vs RW | `+0.84%` | not asserted (arithmetic) |
| Stratified linear M1.5 MSE | `0.00337` | line 212 |
| Pooled NN M1.5 MSE (5-seed mean) | `0.00352` | line 214 |

**Source**: `code/vdem/vdem_v1h_v3_cell.py` lines 197–352 (baselines block).
Saved into `vdem_v1h_frontier.pkl['baselines']` dict.

### §6.3 Regime-aware sweep - Table 5.4 (`tab:vdem-regime-dm`) (paper line 582)

| Regime | Paper claim | Verify line |
|---|---|---|
| R0 (low) | $N=252$, MSE_M2=0.00515, MSE_pool=0.00555, DM=$+2.20$, $p=0.028$ | lines 339, 343, 349, 351 (loop) |
| R1 (mid) | $N=621$, MSE_M2=0.00478, MSE_pool=0.00481, DM=$+0.30$, $p=0.765$ | lines 339, 343, 349, 351 (loop) |
| R2 (high) | $N=637$, MSE_M2=0.00136, MSE_pool=0.00146, DM=$+2.41$, $p=0.016$ | lines 339, 343, 349, 351 (loop) |
| Overall | $N=1{,}510$, MSE_M2=0.00340, MSE_pool=0.00352, DM=$+2.41$, $p=0.0159$ | lines 360, 362, 366, 368 |
| Pooled NN M1.5 − M2 gap range across 9 cells | `+0.00007 to +0.00009` | not directly asserted |
| Headline paired DM | `+2.41` | line 323 |
| Headline two-sided $p$ | `0.0159` | line 325 |
| Hard-routing 5-seed mean MSE | `0.00344` | line 262 |

**Source - sweep**: `code/vdem/vdem_v1h_v3_cell.py` lines 731–820 (sweep
loop). Each cell uses `train_m2_unified()` (lines 456–536) and
`full_audit()` (lines 634–812). Hard-routing variant: `mode='hard_routing'`
branch in `train_m2_unified` (lines 460–470).

**Source - DM test**: `code/vdem/vdem_v1h_postanalysis.py
diebold_mariano()` (function defined at lines 476–502; headline DM call
loop at lines 504–524). Per-regime DM at lines 551–563. Uses HAC variance
estimation with $h=1$.

**Source - verify**: re-computed end-to-end in `verify_paper_numbers.py`
(lines 295–326) using a self-contained `paired_dm()` function (lines
295–303); both approaches agree to 4 decimal places on this test set.

### §6.3 Holm–Bonferroni correction prose (paper lines 587–590)

Holm–Bonferroni thresholds for $m=3$ comparisons:
- Strictest: $\alpha/3 = 0.0167$ - regime 2 ($p=0.016$) survives
- Second: $\alpha/2 = 0.025$ - regime 0 ($p=0.028$) does not survive
- Last: $\alpha = 0.05$ - regime 1 ($p=0.765$) does not reject

Pure arithmetic from the per-regime $p$-values verified above. Not
asserted independently.

### §6.4 Stability verification (paper lines 632–668)

| Paper claim | Value | Verify line |
|---|---|---|
| All 9 audited cells satisfy $L_{\text{eq,max}} < 1$ in 5/5 seeds | `True` | line 234 |
| $L_{\text{eq,max}}$ range across cells (means) | `[0.130, 0.292]` | lines 241, 244 (also lines 403, 405) |
| Max $\rho$ across audited cells | $\leq 0.36$ | line 255 (also line 390) |
| Number of audited (cell, seed) ρ values | `45` | line 392 |
| V-Dem $M$ at unconstrained ($\lambda_M = 0$) | `0.87` | line 269 (also line 374) |
| V-Dem $L_{\text{eq,max}}$ at unconstrained | `0.29` | line 271 |
| V-Dem $M$ at $\lambda_M = 0.10$ (most enforced) | `0.05` | line 278 |
| V-Dem $L_{\text{eq,max}}$ at $\lambda_M = 0.10$ | `0.13` | line 280 |
| V-Dem gap at $\lambda_M = 0$ | `+0.00009` | line 313 |
| V-Dem gap at $\lambda_M = 0.10$ | `+0.00007` | line 315 |
| Overall V-Dem MSE reduction (raw, M2 vs pool) | `≈ 3.3%` | line 413 |

**Source**: `code/vdem/vdem_v1h_v3_cell.py full_audit()` (lines 634–812).
Per-regime $L_Y$, $L_X$, $M$, $L_\phi$, $L_{\phi,\text{eq}}$, $L_{\text{eq,max}}$,
$B_{\text{trans}}$, $\alpha_{\text{emp}}$, $\rho$ computed per seed and
saved per run.

---

## Section 7: Synthetic stress test (paper line 669)

The trimmed paper moves the synthetic Pareto frontier mostly to the
appendix; the body retains a 2-page summary of the headline contrast.
All numbers come from `artifacts/synthetic_v1f_frontier.pkl`.

### §7 prose values (paper lines 686–711)

| Paper claim | Value | Verify line |
|---|---|---|
| Synthetic α (full panel) | `≈ 0.20` | line 496 |
| Synthetic α (training set) | `0.217` | line 494 |
| M1.5 5-seed mean MSE | `0.1497` | line 79 |
| M2 5-seed mean MSE | `0.1410` | line 87 |
| Gap | `+0.0087` | line 100 |
| Gap 95% t-CI | `[+0.0065, +0.0109]` | lines 110, 112 |
| $\hat L_{\max}$ at unconstrained | `0.77` | line 122 |
| $\hat L_{\phi,\text{eq}}$ at unconstrained | `0.27` | line 124 |
| $\hat M$ at unconstrained | `5.94` | line 126 |
| $\hat L_{\text{eq,max}}$ at unconstrained | `2.38` | line 128 |
| $L_{\phi,\text{eq}} \cdot M$ (gating-correction term) | `≈ 1.6` | line 503 |
| OLS slope of gap on $L_{\text{eq,max}}$ | `+0.0075 ± 0.0011` | lines 172, 174 |
| OLS $R^2$ | `0.54` | line 176 |
| OLS $p$-value | `< 10^{-7}` | line 178 |

**Source**: `code/synthetic/synthetic_v1f_sweep.py main()` (lines 449–530):
- M1.5 baseline 5-seed mean - lines 463–471
- M2 sweep 9 cells × 5 seeds - lines 484–515
- `audit_M2()` (lines 325–436) computes per-seed audit constants

### Table 5.5 (`tab:synth-summary`) - Stress test contrast (paper line 716)

**Paper location**: §7 body table, paper lines 716–741.

| Cell | Paper value | Verify line |
|---|---|---|
| Synthetic, $\lambda_M=0$: $M$ | `5.94` | line 187 (also 126) |
| Synthetic, $\lambda_M=0$: $L_{\text{eq,max}}$ | `2.38` | line 189 (also 128) |
| Synthetic, $\lambda_M=0.10$: $M$ | `0.47` | line 191 |
| Synthetic, $\lambda_M=0.10$: $L_{\text{eq,max}}$ | `0.91` | line 193 |
| Synthetic, $\lambda_M=0.10$: gap | `−0.0112` | line 195 |
| V-Dem columns | (same as §6.4 above) | lines 269–280, 313, 315 |

The V-Dem gap at $\lambda_M = 0$ is `+0.00009` and at $\lambda_M = 0.10$
is `+0.00007` (asserted at lines 313, 315).

---

## Appendix A - Proofs (paper line 829)

No empirical numbers; mathematical derivations only.

## Appendix B - Synthetic data and full Pareto frontier (paper line 1019)

### App B.1 DGP parameters (paper line 1022)

DGP details - regime occupancy, transition rate.

| Paper claim | Value | Verify line |
|---|---|---|
| Regime 0 occupancy (training) | `28.2%` | line 481 |
| Regime 1 occupancy (training) | `37.5%` | line 483 |
| Regime 2 occupancy (training) | `34.2%` | line 485 |
| $\alpha_{\text{train}}$ | `0.217` | line 494 |
| $\alpha$ (full panel) | `0.20` | line 496 |

**Source**: regenerated end-to-end in `verify_paper_numbers.py` (lines
449–497) using `np.random.default_rng(42)` and the documented DGP. The
DGP itself is in `code/synthetic/synthetic_v1f_sweep.py generate_panel()`
(lines 74–122).

### App B.3 Table B.1 (`tab:synth-unconstrained`) (paper line 1060)

Body Table 5.1's content moved here in the trim. 5-seed mean test MSE at
the soft-penalty unconstrained ($\lambda_M=0$) cell.

| Paper claim | Value | Verify line |
|---|---|---|
| RW test MSE | `0.1893` | not asserted (single-value derivation) |
| AR(1) test MSE | `0.1895` | not asserted |
| M1 test MSE | `0.1895` | line 77 |
| M1.5 test MSE | `0.1497` | line 79 |
| M2 5-seed mean test MSE | `0.1410` | line 87 |
| Nonlinear X gain (M1.5 vs M1, abs) | `+0.0398` | line 96 |
| Nonlinear X gain (% of M1) | `+20.98%` | line 98 |
| Regime gain (M2 vs M1.5, abs) | `+0.0087` | line 100 |
| Regime gain (% of M1.5) | `+5.82%` | line 102 |
| Regime gap 95% t-CI | `[+0.0065, +0.0109]` | lines 110, 112 |

**Source**: `code/synthetic/synthetic_v1f_sweep.py main()`:
- `m15_test_mse` and `m1_test_mse` - baseline section, lines 463–482
- M2 5-seed values for `('soft_only', 0.0)` cell - sweep loop, lines 485–515
- 95% t-CI: computed inside `verify_paper_numbers.py` (lines 105–113)

### App B.4 Table B.2 (`tab:synth-frontier-full`) - 9-cell Pareto frontier (paper line 1089)

Body Table 5.2's content moved here in the trim.

| Cell | Paper values (Leqmax, M, Lphi·M, gap [t-CI], sat) |
|---|---|
| `soft_only`, $\lambda_M=0.00$ | `2.38, 5.94, 1.62, +0.0087 [+0.0065,+0.0109], 0/5` |
| `soft_only`, $\lambda_M=0.05$ | `1.02, 0.70, 0.19, +0.0046 [+0.0015,+0.0078], 2/5` |
| `soft_only`, $\lambda_M=0.10$ | `0.96, 0.42, 0.11, +0.0028 [+0.0003,+0.0054], 3/5` |
| `proj_epoch`, $\lambda_M=0.00$ | `2.56, 6.79, 1.93, +0.0071 [+0.0067,+0.0076], 0/5` |
| `proj_epoch`, $\lambda_M=0.05$ | `1.02, 0.71, 0.19, -0.0049 [-0.0076,-0.0022], 2/5` |
| `proj_epoch`, $\lambda_M=0.10$ | `0.92, 0.44, 0.12, -0.0073 [-0.0097,-0.0048], 5/5` |
| `proj_batch`, $\lambda_M=0.00$ | `2.59, 6.95, 1.98, +0.0066 [+0.0058,+0.0075], 0/5` |
| `proj_batch`, $\lambda_M=0.05$ | `0.99, 0.74, 0.20, -0.0089 [-0.0122,-0.0055], 2/5` |
| `proj_batch`, $\lambda_M=0.10$ | `0.91, 0.47, 0.12, -0.0112 [-0.0142,-0.0083], 5/5` |

**Verify lines**: 153, 155, 157, 159, 161 inside the loop over the 9 cells - **45 frontier-table assertions** (5 columns × 9 cells).

**Source**: `code/synthetic/synthetic_v1f_sweep.py main()` - sweep loop
over `modes × lam_Ms × SEEDS`, lines 484–515. The `train_M2()` call at
line 500 selects the enforcement mode; the soft-Lipschitz penalty,
projection-every-epoch, and projection-every-batch logic are inside
`train_M2()` lines 265–324. Spectral-norm projection in
`project_per_edge_nets()` lines 235–244.

## Appendix C - V-Dem data and preprocessing (paper line 1131)

Descriptive. Regime boundaries from training-set tertile cutpoints on
$Y_{t-1}$. Specific cutpoint values are computed at runtime by
`code/vdem/vdem_v1h_v3_cell.py` lines 90–96, identical in
`vdem_v1h_postanalysis.py` (lines 144–164) and
`regime_stratified_diagnostic.py` (lines 122–136).

## Appendix D - Audit protocol (paper line 1201)

### D.1 Gradient-probe (paper line 1215)

Describes the gradient-probe protocol used by
`audit_per_regime_Lipschitz()` in `code/vdem/vdem_v1h_v3_cell.py`
(lines 537–562). The probe takes a held-out batch, evaluates per-sample
Jacobians, returns the sample maximum.

### D.2 Certified spectral-norm bound (paper line 1268)

Per-regime probe vs certified upper bound across 3 seeds.

| Paper claim | Value | Verify line |
|---|---|---|
| Regime 0: probe (mean ± std) | `0.129 ± 0.011` | lines 517, 519 |
| Regime 0: cert | `0.136 ± 0.014` | lines 521, 524 |
| Regime 1: probe | `0.152 ± 0.022` | lines 517, 519 |
| Regime 1: cert | `0.157 ± 0.023` | lines 521, 524 |
| Regime 2: probe | `0.024 ± 0.015` | lines 517, 519 |
| Regime 2: cert | `0.030 ± 0.012` | lines 521, 524 |
| Max-over-regimes: probe mean | `0.155` | line 533 |
| Max-over-regimes: cert mean | `0.164` | line 535 |
| Cert ≥ probe in all 9 (seed, regime) cells | `True` | line 342 |
| Cert/probe ratio mean | `1.07` | line 348 |
| Empirical gap on V-Dem (text claim) | `≈ 7%` | derivable from above |

**Source**: `code/vdem/certified_bound_audit.py`, full file 167 lines.
Two core estimators:
- **Probe**: `probe()` (lines 98–115). Per-sample Jacobian on a held-out
  batch, max over samples.
- **Certified**: `cert()` (lines 116–125). Computes
  $\sigma(W^{(r,e)}_1) \cdot \sigma(W^{(r,e)}_2)$ per edge and sums over
  Y-channel edges per regime; analytical upper bound from spectral norms
  of the layer weights.
- **Comparison**: `main()` (lines 126–166) trains 3 seeds with vanilla
  MSE for 100 epochs, runs both estimators, saves
  `artifacts/certified_audit_results.pkl`.

## Appendix E - Hyperparameters (paper line 1321)

Descriptive. Values match the constants and `train_*` function signatures:

| Paper claim | Source |
|---|---|
| V-Dem hidden width 32, lag 5 | `vdem_v1h_v3_cell.py` lines 78–94 (`HIDDEN`, `L_LAG`) |
| V-Dem Adam lr 3e-4, weight decay 1e-4 | `train_simple()` defaults at line 285, `train_m2_unified()` defaults at line 456 |
| V-Dem batch 128 | `train_simple()` and `train_m2_unified()` `batch=128` defaults |
| V-Dem pooled epochs 300, stratified epochs 900 | `BASELINE_EPOCHS_POOLED`, `BASELINE_EPOCHS_STRATIFIED` constants lines 84–88 |
| Soft target $\tilde L_Y = 0.55$ | `train_m2_unified()` parameter `target_LY_soft=0.55` (line 459) |
| Projection target 0.85 | `train_m2_unified()` parameter `projection_target=0.85` (line 460) |
| $\lambda_\phi = 0.005$ | `train_m2_unified()` parameter `lam_phi=0.005` (line 459) |
| Synthetic equivalents | `synthetic_v1f_sweep.py main()` lines 500–503 (same parameter values) |

---

## Section 1 (Introduction) and Section 8 (Limitations and conclusion)

§1 cites values that all appear in §6/§7/§A; §8 likewise. Specific abstract
claim "4 of 4 covariates show sign reversal at one or more lag" is
asserted at line 634.

---

## Coverage summary

`verify_paper_numbers.py` runs **141 checks** in **<30 seconds** and
asserts:

- App B.3 baselines and gain decomposition (9 checks)
- §7 / §6.4 audit constants for V-Dem and synthetic (10 checks)
- App B.4 9-cell frontier - every cell × every column (45 checks)
- §7 OLS slope claim (4 checks)
- §7 / Table 5.5 cross-experiment table - synthetic & V-Dem columns
  including gaps (11 checks)
- §6.2 V-Dem baselines (5 checks)
- §6.2 / Tables 1, 2 panel sizing and regime occupancy (7 checks)
- §6.1 Table 5.3 per-regime linear stratification - pooled and
  stratified MSE in each regime (8 checks)
- §6.4 stability conditions (5 checks)
- §6.3 hard-routing MSE (1 check)
- §6.3 Table 5.4 per-regime DM - DM, p-value, MSE_M2, MSE_pool in each
  regime + overall (16 checks)
- Figure 1 cumulative-advantage final values per regime (3 checks)
- Figure 1B per-regime party-organization coefficients (4 checks)
- App B.1 synthetic DGP regime structure and α (5 checks)
- App D.2 certified bound - per-regime detail and aggregate (14 checks)
- Abstract / §6.1 "4 of 4 covariates show sign reversal" (1 check)
- §6.4 / §8 raw MSE reduction and per-regime % MSE reductions (4 checks)

**All 141 pass against the archived artifacts.**

Numbers cited in the paper but not currently asserted (computable on
demand from the four artifact pickles or with arithmetic):

- Table 5.3 per-regime improvement percentages (`+0.36%`, `+4.49%`,
  `+2.01%` - pure arithmetic from the asserted pooled and stratified
  per-regime MSEs)
- Holm–Bonferroni thresholds (pure arithmetic from per-regime $p$-values
  which are asserted)
- The "−2.25% (worse)" and "+0.84%" comparisons of pooled / stratified
  linear vs RW (pure arithmetic from the asserted MSEs)
- "≈ 7%" empirical gap in App D.2 (arithmetic from probe/cert means
  which are asserted)

To extend the verification script to cover any of these, load the
relevant pickle and add a `check()` call. Each takes a few lines.


---

# Camera-ready additions (NeurIPS 2026 final version)

Line numbers above refer to the submitted `paper/submitted/main.tex`. The
camera-ready `paper/main.tex` reorganizes the paper (several blocks moved to
appendices for the 10-page budget); the mapping below is by section, table and
figure label of the camera-ready. Every row is checked by
`verify_camera_ready_numbers.py`.

| Camera-ready claim | Location | Artifact | Script |
|---|---|---|---|
| Table 2 rows 1-3 (M2-ens vs RW, vs stratified linear; stratified vs RW): mean d, cluster t, cluster p, bootstrap p | Table 2 (`tab:vdem-regime-dm`), Sec. 6.2 | `artifacts/rebuttal_R1/results_18877_R1_dm_tests.csv` families C2, C1, C3, scope overall | `scripts/NeurIPS_18877_camera_ready.ipynb` cell 1 |
| Table 2 row 4 (stratified vs pooled linear, bootstrap p = 0.273) and the +3.02% gain | Table 2; Sec. 6.1; Sec. 6.4 | `artifacts/rebuttal_S2/results_S2_A3_stratification_tests.csv` | notebook cell 3 |
| Table 2 caption: regime-2 closest case p = 0.051 / 0.202 / Holm 0.606; no per-regime test significant | Table 2 caption | `results_18877_R1_dm_tests.csv` family C1 scope regime2; `p_holm` column | cell 1 |
| Submitted DM at three decimals (+2.411 / +2.196 / +0.299 / +2.407) | Sec. 6.2, Fig. 1 C-E | `artifacts/rebuttal_R1/results_18877_R1_table2_precision.csv` | cell 2 |
| Table 1 per-run and ensemble values over 20 seeds; unpaired t = +6.73 / +8.67; 39 of 40; seed-fair row 5 of Table 2; stratified-NN p = 0.51 / 0.54 | Table 1, Table 2 row 5, Sec. 6.2, Appendix | `artifacts/vdem_v1h_postanalysis_20seed.pkl` (primary block, seeds 0-19) and `artifacts/kdd_s81_freshseed.pkl` (fresh block, seeds 100-119) | `verify_camera_ready_numbers.py` |
| 24-variant grid: mechanism 24/24, low-stratum coefficient negative 24/24, significant 13/24, outer reversal 0/24, canonical 4/4 covariates straddle zero | Sec. 6.4, Appendix (sensitivity) | `artifacts/rebuttal_S2/results_S2_A1_diagnostic_grid.csv` | cell 3 |
| Jackknife 94.5% / 1.5%; era windows (-0.024 to -0.015; +0.001 in the most recent half); leave-one-covariate-out | Sec. 6.4, Appendix | `artifacts/rebuttal_S2/results_S2_A2_stability.csv` | cell 3 |
| M2 retrained under 4 variants: vs RW p >= 0.80, vs stratified linear Holm p >= 0.16 | Sec. 6.4 | `artifacts/rebuttal_S2/results_S2_B_m2_variants.csv`; weights in `s2_checkpoints/` | cell 4 |
| Eight-configuration stratified-linear table (gain vs pooled; standardized party-org coefficients) | Appendix (sensitivity), `tab:vdem-sensitivity` | `artifacts/rebuttal_S2/s2_grid.csv` | `verify_camera_ready_numbers.py` |
| RV screen: 28.2% headroom (0.0746 vs 0.1039), bootstrap p at the 1e-4 floor; Wald W = 59.9, p = 5.4e-7; 0 of 8 slopes reverse; 2,615 / 2,611 days; 523 test days; N = 4,184 | Sec. 8, Appendix (RV) | `artifacts/rv_screen_S1/results_RVS1_gate1_headroom.csv`, `results_RVS1_gate2_diagnostic.csv`, `results_RVS1_meta.json` | cell 5 |
| Table 4 (per-seed audit), Sec. 8 ranges (L_Y 0.55-0.58, gating term 0.43-0.68, L_eq 1.07-1.25), seed-4 constants (0.984, 1.010, 0.271/0.619, rho 0.991) and corrected constants (0.996, 0.305, 0.9995) | Table 4 (`tab:rv-audit`), Sec. 8 | `artifacts/rv_f1/results_RVF1_audit_constants.csv` (Corr. = L_max + L_phi,eq * M) | cell 6 |
| Sec. 8 predictive results: t = -3.82 (p = 0.0014), calm regime t = -4.85 (Holm 0.0012, 5/5), t = -3.57 (p = 0.0024), interaction vs pooled t = -0.06 (p = 0.95), M2 vs RW t = +8.57 (p = 1e-4); test MSEs; claim level C; hyperparameter grid and selected cell | Sec. 8, Appendix (RV), checklist Q6/Q8 | `artifacts/rv_f1/results_RVF1_dm_tests.csv`, `results_RVF1_meta.json` | cell 6 |
| Corrected Lemma 2 bound on V-Dem: at most 0.362 across 45 runs, shift at most 0.006, all certified, rho <= 0.37 | Sec. 4, Sec. 6.3 | `artifacts/vdem_v1h_frontier.pkl` (`L_max + max_r L_phi,eq^(r) * M` per run) | `verify_camera_ready_numbers.py` |
| Ten-cell V-Dem parity 0.00343-0.00345, seed sd 0.00003-0.00005, including hard routing | Sec. 9 | `artifacts/vdem_v1h_frontier.pkl` | `verify_camera_ready_numbers.py` |
| Figure 1 C-E labels: MSE 7.3% / 0.5% / 6.3% below pooled; N = 252 / 621 / 637 | Fig. 1 | `artifacts/vdem_v1h_postanalysis.pkl` | `code/vdem/build_figure.py` |
