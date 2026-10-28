# PANAR-X Reproducibility Package

This archive contains everything needed to verify and reproduce the empirical
results in:

> **Stability-Constrained Regime-Aware Forecasting for Heterogeneous Panel
> Time Series**. Dmitry Zaytsev, Valentina V. Kuskova, and Michael Coppedge.
> Advances in Neural Information Processing Systems 39 (NeurIPS 2026).
> University of Notre Dame (Lucy Family Institute for Data & Society,
> Kellogg Institute for International Studies, Department of Political
> Science).

Repository: https://github.com/vkuskova/NeurIPS-18877. Archived releases of
this repository are deposited on Zenodo; cite the version DOI given in the
paper.

The paper introduces PANAR-X, a regime-aware additive forecasting architecture
with verifiable contraction conditions, and demonstrates it on V-Dem
democratic-development panel data and on a synthetic Pareto-frontier sweep.

## Camera-ready update (NeurIPS 2026 final version)

This package was refreshed for the camera-ready. Everything the submitted
package contained is still here, unchanged; the additions are:

| Added | What it holds |
|---|---|
| `designs/` | The four pre-registered design documents, locked before any confirmatory run: `design_18877_R1_M2_vs_strong_baselines.md` (panel-appropriate inference), `design_18877_S2_sensitivity.md` (regime-construction sensitivity), `design_RV_S1_screening.md` (second-panel screen), `design_RV_F1_ladder.md` (realized-volatility confirmatory experiment). |
| `artifacts/rebuttal_R1/` | Country-clustered and country-block-bootstrap tests behind Table 2 (`results_18877_R1_dm_tests.csv`), the three-decimal recomputation of the submitted DM statistics (`results_18877_R1_table2_precision.csv`), the step-0 reconciliation of every printed submission number (`results_18877_R1_step0_reconciliation.csv`), and run metadata. |
| `artifacts/rebuttal_S2/` | The 24-variant diagnostic grid (`results_S2_A1_diagnostic_grid.csv`), era/jackknife/leave-one-covariate-out stability (`results_S2_A2_stability.csv`), the stratified-vs-pooled panel test (`results_S2_A3_stratification_tests.csv`), the four M2 retraining variants (`results_S2_B_m2_variants.csv`), the independent reimplementation behind the eight-configuration stratified-linear table (`s2_grid.csv`), metadata, and the 20 M2 checkpoints (`s2_checkpoints/`). |
| `artifacts/rv_screen_S1/` | Gate 1 (headroom) and Gate 2 (heterogeneity) results of the realized-volatility screen, with metadata. |
| `artifacts/rv_f1/` | Realized-volatility confirmatory run: predictive tests (`results_RVF1_dm_tests.csv`), the per-seed certificate audit behind Table 4 (`results_RVF1_audit_constants.csv`), metadata including the pre-registered hyperparameter grid and validation MSEs, and all 25 checkpoints (`rvf1_checkpoints/`). |
| `artifacts/vdem_v1h_postanalysis_20seed.pkl`, `artifacts/kdd_s81_freshseed.pkl` | The two 20-seed V-Dem blocks: the primary block (seeds 0-19) behind Table 1's per-run and ensemble columns, Table 2 row 5, and the primary-set statistics of Sec. 6.2; and the pre-committed fresh block (seeds 100-119) behind the replication statistics. Each holds per-seed and ensemble predictions for M1, pooled M1.5, stratified M1.5, and M2 on the test partition, plus the stored fair-comparison tables. |
| `verify_camera_ready_numbers.py` | Checks covering every number added or changed in the camera-ready (Table 2, Sec. 6.4 sensitivity claims, Sec. 8 and Table 4, the corrected Lemma 2 constants, the ten-cell parity in Sec. 9, and the relabeled Figure 1), each computed from the artifacts above. Runs in seconds, no GPU. |
| `code/vdem/build_figure.py` | Updated: panels C-E of Figure 1 now label the per-regime relative MSE reduction (computed in-script) and the submitted per-run DM statistics at three decimals. |

Audit path for the camera-ready:

```bash
python verify_paper_numbers.py          # 141 checks, submitted-version numbers
python verify_camera_ready_numbers.py   # camera-ready additions
```

Both are expected to end with all checks passed.

Scripts that produced the rebuttal artifacts: `scripts/NeurIPS_18877_camera_ready.ipynb`
holds the six locked-design cells (R1, Table 2 precision, S2 Part A, S2 Part B,
RV-S1, RV-F1) in run order, with their outputs preserved; its header cell
documents the Drive layout each cell expects.

Realized-volatility input data: the panel `rv_dataset.csv` (daily realized
volatility of eight equity indices) is not redistributed with this package.
Readers who wish to re-run the RV-S1 and RV-F1 cells should construct it from
the source described in the original paper, Son et al. (2023), cited in the
camera-ready. The SHA-256 of the file used (recorded in
`artifacts/rv_screen_S1/results_RVS1_meta.json` and
`artifacts/rv_f1/results_RVF1_meta.json`) is checked by both cells before any
computation, and every number the paper reports from that panel is already
reproducible from the archived artifacts via `verify_camera_ready_numbers.py`.

The paper itself is not included in this package; it is available from the
NeurIPS 2026 proceedings.

## Two paths through this package

### Path 1 - Audit (5 minutes, no training)

If you only want to verify that the numbers reported in the paper match the
saved artifacts:

```
pip install -r requirements.txt
python verify_paper_numbers.py
```

This loads the four pickles in `artifacts/` and recomputes every numerical
claim made in the paper (141 total checks: tables, audit constants, paired DM
tests, certified bounds, plus prose-level claims in the §4 anchor, §6
narrative, §7 synthetic, §8 limitations, the abstract's 4-of-4
sign-reversal claim, Figure 1 cumulative values, and V-Dem panel sizing).
Each check prints PASS/FAIL with the section/table where the number
appears in the paper. Total runtime: under 30 seconds.

### Path 2 - Replicate (full retraining)

If you want to regenerate the artifacts from scratch:

**Option A - Local machine (CLI):**
```
pip install -r requirements.txt
bash scripts/run_all.sh
```

**Option B - Google Colab (recommended for GPU):**

Upload `panar-x-reproducibility.tar.gz` and `scripts/NeurIPS2026_18877_pipeline.ipynb` to
Colab. Open the notebook, set runtime to T4 GPU, and run cells
top-to-bottom. The notebook handles Drive mounting, dependency install,
all five training stages, verification, drift analysis, and saves
outputs to `MyDrive/panar-x-replication/run_<timestamp>/`.

This runs (in order): the synthetic Pareto-frontier sweep, the V-Dem 50-run
sweep, postanalysis, the certified-bound audit, and the figure builder.
Expected runtime: ~8 hours CPU or ~70 minutes Colab T4 GPU; see Section
"Compute requirements" below.

## Package contents

```
panar-x-reproducibility/
├── README.md                       (this file)
├── LICENSE                         (MIT for code; V-Dem under CC BY 4.0)
├── requirements.txt                (pinned Python dependencies)
├── verify_paper_numbers.py         (Path-1 entry point: submitted-version numbers)
├── verify_camera_ready_numbers.py  (Path-1 entry point: camera-ready additions)
├── PAPER_TO_CODE_TRACEABILITY.md   (every paper claim -> artifact -> script)
│
├── designs/                        (Pre-registered design documents, locked before any run)
│   ├── design_18877_R1_M2_vs_strong_baselines.md
│   ├── design_18877_S2_sensitivity.md
│   ├── design_RV_S1_screening.md
│   └── design_RV_F1_ladder.md
│
├── code/                           (All training, audit, and figure code)
│   ├── synthetic/
│   │   ├── synthetic_v1_baselines.py    (M1, M1.5, M2 single-seed reference)
│   │   ├── synthetic_v1f_sweep.py       (45-run synthetic Pareto sweep)
│   │   └── README.md
│   ├── vdem/
│   │   ├── vdem_v1h_v3_cell.py            (50-run V-Dem sweep)
│   │   ├── vdem_v1h_postanalysis.py       (paired DM, baseline comparison)
│   │   ├── regime_stratified_diagnostic.py (diagnostic, Table 1)
│   │   ├── certified_bound_audit.py       (certified spectral-norm audit)
│   │   ├── build_figure.py                (Figure 1; camera-ready labels)
│   │   └── README.md
│   └── common/
│       ├── NAVAR.py                       (reference: original NAVAR code)
│       ├── dataloader.py
│       └── README.md
│
├── data/                           (V-Dem v15 panel; CC BY 4.0)
│   ├── merged_completevars_balanced_zero_missing.csv
│   └── README.md                   (data provenance and preprocessing)
│
├── artifacts/
│   ├── synthetic_v1_results.pkl    (M1/M1.5/M2 baseline reference, single seed)
│   ├── synthetic_v1f_frontier.pkl  (45 synthetic runs: 9 cells x 5 seeds)
│   ├── vdem_v1h_frontier.pkl       (50 V-Dem runs: 10 cells x 5 seeds + baselines)
│   ├── vdem_v1h_postanalysis.pkl   (V-Dem predictions for headline cell + baselines)
│   ├── certified_audit_results.pkl (Spectral-norm vs gradient-probe Lipschitz)
│   ├── kdd_s81_freshseed.pkl       (20-seed V-Dem runs, primary and fresh seed blocks; Table 1, Table 2 row 5)
│   ├── rebuttal_R1/                (panel-appropriate inference; Table 2)
│   ├── rebuttal_S2/                (regime-construction sensitivity; Sec. 6.4 and appendix)
│   ├── rv_screen_S1/               (realized-volatility screen)
│   ├── rv_f1/                      (realized-volatility confirmatory run; Table 4)
│   └── README.md                   (provenance of each artifact)
│
└── scripts/
    ├── run_all.sh                        (Path-2 CLI entry point, submitted experiments)
    ├── NeurIPS2026_18877_pipeline.ipynb  (Path-2 Colab entry point, submitted experiments)
    └── NeurIPS_18877_camera_ready.ipynb  (six locked-design cells, camera-ready experiments)
```

## Compute requirements

| Stage | Code | Time (CPU) | Time (Colab GPU) | Output |
|-------|------|------------|------------------|--------|
| Synthetic v1 baselines | `code/synthetic/synthetic_v1_baselines.py` | ~5 min | ~5 min | `artifacts/synthetic_v1_results.pkl` |
| Synthetic sweep (45 runs) | `code/synthetic/synthetic_v1f_sweep.py` | ~4.5 hours | ~30 min | `artifacts/synthetic_v1f_frontier.pkl` |
| V-Dem sweep (50 runs) | `code/vdem/vdem_v1h_v3_cell.py` | ~3.5 hours | ~30 min | `artifacts/vdem_v1h_frontier.pkl` |
| V-Dem postanalysis | `code/vdem/vdem_v1h_postanalysis.py` | ~2 hours | ~22 min | `artifacts/vdem_v1h_postanalysis.pkl` |
| Certified-bound audit | `code/vdem/certified_bound_audit.py` | ~10 min | ~2 min | `artifacts/certified_audit_results.pkl` |
| Figure builder | `code/vdem/build_figure.py` | ~15 sec | ~15 sec | `figures/figure_diagnostic.pdf` |
| **Total replication** | | **~8 hours CPU** | **~85 min GPU** | All artifacts and figure |

The V-Dem sweep and postanalysis dominate because both include per-batch
Jacobian penalties that are expensive on CPU. We strongly recommend running
the full pipeline on GPU (Colab Free is sufficient, though Colab Pro is
needed because total runtime now exceeds the free idle timeout of 90 min).

For audit-only verification (the recommended path for reviewers), no GPU
is needed - `verify_paper_numbers.py` runs in under 30 seconds on any
machine.

## Determinism notes

CPU runs are bit-exact reproducible across machines: random seeds are fixed
in code via `torch.manual_seed` and `np.random.default_rng`, and no
non-deterministic CUDA kernels are used. GPU runs may differ at the level
of `~1e-6` in test MSE due to non-determinism in CUDA matmul kernels; this
does not affect the paper's numerical claims, all of which are reported to
4 decimal places or to whole percentages. The verification script
(`verify_paper_numbers.py`) uses tolerances appropriate to this precision
(typically `tol_abs=5e-5` to `5e-3`).

## Dependencies

Python 3.10+. Core libraries: `torch>=2.0`, `numpy`, `pandas`, `scipy`,
`scikit-learn`, `matplotlib`. Pin file in `requirements.txt`.

## Citing this work

[citation block - to be filled by authors after acceptance]

## Licensing

- **Code**: MIT License (see LICENSE)
- **V-Dem data**: Creative Commons Attribution 4.0 International (CC BY 4.0).
  Original V-Dem v15 data: Coppedge et al. (2025), "V-Dem [Country–Year/Country–Date]
  Dataset v15", https://doi.org/10.23696/vdemds25
