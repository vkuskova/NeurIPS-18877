# Design RV-S1: Second-panel screening — realized volatility

Status: DRAFT — locks on Valya's sign-off plus the data specification in §2.
Role: Gatekeeper only. Decides whether the RV panel qualifies for a full locked
ladder experiment (a separate design, written only if this screen passes). M2 does
not run anywhere in this protocol. Whatever the screen shows is reported; the panel
was chosen before any RV number was computed, and this document is the record of
that choice.

## 1. Pre-commitment

The realized-volatility panel is THE second panel. If the screen fails either gate,
the conclusion is "no qualifying second panel identified among available data" —
we do not move to GDELT or WDI and screen again in search of a pass. (A future
screen of a different panel would require its own design, written for its own
reasons, and would carry this failed screen in its record.)

## 2. Data specification (filled 2026-07-25 from verified schema; read-only check)

- D1. Source: `/content/drive/MyDrive/GNAVAR/data/rv_dataset.csv`, SHA-256
  `9f6065e3ac046a1a18512668fa71457dba5dc51bc1cb6e802a1441cd96a0098c`. This copy
  is canonical; the duplicate under `Democracy as EP/data/` is not used.
  Provenance: the G-NAVAR project's realized-volatility experiment data
  (Oxford-Man-style RIC-coded equity indices).
- D2. Units: 8 equity indices, columns `.SPX, .GDAXI, .FCHI, .FTSE, .OMXSPI,
  .N225, .KS11, .HSI` (wide format; melted to long for the panel).
- D3. Time: no calendar column; the integer row index is the time axis
  (2,615 daily observations, ~10.4 trading years). Recorded as a provenance
  limitation in meta.
- D4. RV measure: the stored values directly, on the daily-volatility (sd)
  scale (medians 0.6–0.9%, min 0.0011). Strictly positive and NaN-free
  (verified), so no floor-clipping is needed before the log.
- D5. No covariates beyond RV exist in the file. The covariate channel is
  defined as the lagged cross-asset mean: X_it = mean of y_jt over the 7 other
  assets (the natural spillover summary, mirroring the paper's Y-lags +
  covariate-channel structure).

Fixed transformations, not negotiable after lock:
- y_it = log(RV_it).
- Target: one-step-ahead Δy_it (mirrors the paper's ΔY convention). RW forecasts
  Δy = 0.
- Lags: p = 4 for both own-volatility and the cross-asset-mean covariate
  (8 slopes per model; mirrors the paper's lag depth).
- The panel is balanced with T = 2,615 per asset; no drop rule needed.

## 3. Split, regimes, inference axis

- Split: per asset, final 20% of time points = test (523 test days × 8 assets =
  4,184 test observations; 2,092 training days). No shuffling; strictly
  chronological. Identical day boundary for all assets.
- Regimes: R = 3, tertiles of y_{t-1} computed on the TRAINING portion pooled
  across assets (mirrors the paper's training-tertile routing exactly). Note
  recorded: assets with persistently higher vol levels will concentrate in
  upper regimes, exactly as persistent democracies did in V-Dem.
- Inference: AMENDED from the draft. With G = 8 units, cluster-by-asset
  inference (bootstrap over 8 clusters; cluster-t with 7 df) is not defensible;
  the draft assumed ~30 units. The panel is time-dominant (T = 2,615), so the
  inference axis flips: moving-block bootstrap over DAYS. Each resampled block
  of consecutive days carries all 8 assets' values intact, preserving serial
  dependence within blocks and the strong cross-asset comovement of volatility.
  Block length 25 trading days (≈ one month; pre-specified, not tuned),
  B = 9,999, studentized (bootstrap-t), RNG seed 20260801. Secondary: per-asset
  Diebold–Mariano tests with HAC (Newey–West, 25 lags) variance, all 8 reported.
- Scope-of-claim consequence, recorded honestly: with 8 units, any conclusion
  is about global equity-index volatility, not a broad cross-section of assets.

## 4. Gate 1 — headroom (condition 1)

Models, deterministic, no seeds, trained on training portion only:
- RW (Δy = 0)
- Pooled linear: Δy_it on y_{i,t-1..t-4} + X_{i,t-1..t-4} (cross-asset mean,
  §2 D5) + asset intercepts
- Stratified linear: same specification fit separately per regime

Test: pooled linear vs RW, all 4,184 test observations, time-block bootstrap-t
(§3), two-sided. GATE 1 PASSES iff pooled linear beats RW with p_boot < 0.05
and positive mean differential. (Stratified linear vs RW and stratified vs
pooled are computed and reported as context; they are not gates. Per-asset HAC
DM tests reported alongside.)

## 5. Gate 2 — regime-dependent structure (condition 2)

On the TRAINING portion, mirror the paper's sign-reversal diagnostic:
- Fit pooled linear and per-regime linear models (same specification as §4;
  8 slopes each: 4 own-vol lags + 4 spillover lags).
- Heterogeneity test: joint Wald test that all 8 slopes are equal across the
  three regimes. Statistic W = d'Σ̂⁻¹d with d the stacked 16-dim difference
  vector (β⁽⁰⁾−β⁽¹⁾, β⁽¹⁾−β⁽²⁾) and Σ̂ its covariance estimated from the same
  moving-block bootstrap over training days (B = 9,999); reference χ²(16).
- Sign-reversal inventory: every slope whose regime-0 and regime-2 coefficients
  have opposite signs, with both magnitudes exceeding twice their block-
  bootstrap SEs.

GATE 2 PASSES iff the Wald test rejects at p < 0.05 AND the inventory is
non-empty. (Heterogeneity without sign reversal — coefficients scaling but not
flipping, e.g., persistence strengthening in crisis regimes without changing
sign — is reported as a partial pass with the note that the paper's
cancellation mechanism specifically requires reversal; whether partial pass
justifies the full experiment is Valya's call, made after seeing which
coefficients move.)

## 6. Outcomes

- PASS (both gates): a full ladder design (RV-F1) is written, mirroring
  18877-R1 — M2 [ENS] with per-seed gates vs stratified linear and RW,
  asset-clustered, pre-specified claim levels — locked before any M2 training
  on RV data begins. Screen results are frozen as the motivation section of
  that design.
- FAIL either gate: reported as-is. The screen itself is content for the KDD
  diagnostic framing (the headroom test is a reusable panel-qualification tool)
  regardless of direction.
- No re-specification of lags, covariates, splits, regime counts, or RV column
  after lock, in either direction.

## 7. Outputs

To /content/drive/MyDrive/NeurIPS2026_3/rv_screen_S1/:
- `results_RVS1_gate1_headroom.csv` — every pairwise test, same columns as the
  18877-R1 test file.
- `results_RVS1_gate2_diagnostic.csv` — pooled and per-regime coefficients,
  cluster SEs, Wald statistic, sign-reversal inventory.
- `results_RVS1_meta.json` — data file hash, asset list, drop list, split
  boundaries, RNG seed, gate verdicts.

## 8. Lock status

Resolved 2026-07-25: data specification (§2, single canonical file,
SHA-verified, schema read-only), inference amendment (§3, time-block bootstrap
replacing asset clustering at G = 8), gate statistics concretized (§4–5).

Remaining before lock — Valya:
1. Sign-off on the inference amendment (§3) and the narrowed scope-of-claim it
   implies (8 global equity indices).
2. Sign-off on the spillover covariate definition (§2 D5) and the 8-slope
   specification.
3. Sign-off on gates as stated, including the partial-pass handling in §5.
Output directory: /content/drive/MyDrive/NeurIPS2026_3/rv_screen_S1/ (per §7).
