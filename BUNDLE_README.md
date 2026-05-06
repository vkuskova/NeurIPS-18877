# PANAR-X Reproducibility Package

This archive contains everything needed to verify and reproduce the empirical
results in:

> **Stability-Constrained Regime-Aware Forecasting for Heterogeneous Panel
> Time Series** — Anonymous (NeurIPS 2026 submission).

The paper introduces PANAR-X, a regime-aware additive forecasting architecture
with verifiable contraction conditions, and demonstrates it on V-Dem
democratic-development panel data and on a synthetic Pareto-frontier sweep.

## Two paths through this package

### Path 1 — Audit (5 minutes, no training)

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

### Path 2 — Replicate (full retraining)

If you want to regenerate the artifacts from scratch:

**Option A — Local machine (CLI):**
```
pip install -r requirements.txt
bash scripts/run_all.sh
```

**Option B — Google Colab (recommended for GPU):**

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
├── LICENSE                         (MIT for code; V-Dem under CC BY 4.0; see
│                                    paper/ for paper licensing)
├── requirements.txt                (pinned Python dependencies)
├── verify_paper_numbers.py         (Path-1 entry point)
│
├── paper/                          (Paper sources and compiled PDF)
│   ├── main.pdf                    (compiled paper, 20 pages: 9 body + 1 refs + 10 appendix)
│   ├── main.tex                    (single-file source — body + appendices)
│   ├── checklist.tex               (NeurIPS paper checklist — back matter)
│   ├── references.bib
│   ├── neurips_2026.sty
│   └── figure_diagnostic.pdf       (Figure 1)
│
├── code/                           (All training, audit, and figure code)
│   ├── synthetic/
│   │   ├── synthetic_v1_baselines.py    (M1, M1.5, M2 single-seed reference)
│   │   ├── synthetic_v1f_sweep.py       (45-run synthetic Pareto sweep)
│   │   └── README.md
│   ├── vdem/
│   │   ├── vdem_v1h_v3_cell.py            (50-run V-Dem sweep)
│   │   ├── vdem_v1h_postanalysis.py       (paired DM, baseline comparison)
│   │   ├── regime_stratified_diagnostic.py (Table 5.3 diagnostic)
│   │   ├── certified_bound_audit.py       (Appendix D.2 spectral-norm audit)
│   │   ├── build_figure.py                (Figure 1)
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
├── artifacts/                      (Saved pickles from the original runs)
│   ├── synthetic_v1_results.pkl    (M1/M1.5/M2 baseline reference, single seed)
│   ├── synthetic_v1f_frontier.pkl  (45 synthetic runs: 9 cells × 5 seeds)
│   ├── vdem_v1h_frontier.pkl       (50 V-Dem runs: 10 cells × 5 seeds + baselines)
│   ├── vdem_v1h_postanalysis.pkl   (V-Dem predictions for headline cell + baselines)
│   ├── certified_audit_results.pkl (Spectral-norm vs gradient-probe Lipschitz)
│   └── README.md                   (provenance of each pickle)
│
└── scripts/
    └── run_all.sh                  (Path-2 CLI entry point)
    └── NeurIPS2026_18877_pipeline.ipynb               (Path-2 Colab entry point)
```

## Compute requirements

| Stage | Code | Time (CPU) | Time (Colab GPU) | Output |
|-------|------|------------|------------------|--------|
| Synthetic v1 baselines | `code/synthetic/synthetic_v1_baselines.py` | ~5 min | ~5 min | `artifacts/synthetic_v1_results.pkl` |
| Synthetic sweep (45 runs) | `code/synthetic/synthetic_v1f_sweep.py` | ~4.5 hours | ~30 min | `artifacts/synthetic_v1f_frontier.pkl` |
| V-Dem sweep (50 runs) | `code/vdem/vdem_v1h_v3_cell.py` | ~3.5 hours | ~30 min | `artifacts/vdem_v1h_frontier.pkl` |
| V-Dem postanalysis | `code/vdem/vdem_v1h_postanalysis.py` | ~2 hours | ~22 min | `artifacts/vdem_v1h_postanalysis.pkl` |
| Certified-bound audit | `code/vdem/certified_bound_audit.py` | ~10 min | ~2 min | `artifacts/certified_audit_results.pkl` |
| Figure builder | `code/vdem/build_figure.py` | ~15 sec | ~15 sec | `paper/figure_diagnostic.pdf` |
| **Total replication** | | **~8 hours CPU** | **~85 min GPU** | All artifacts and figure |

The V-Dem sweep and postanalysis dominate because both include per-batch
Jacobian penalties that are expensive on CPU. We strongly recommend running
the full pipeline on GPU (Colab Free is sufficient, though Colab Pro is
needed because total runtime now exceeds the free idle timeout of 90 min).

For audit-only verification (the recommended path for reviewers), no GPU
is needed — `verify_paper_numbers.py` runs in under 30 seconds on any
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

[citation block — to be filled by authors after acceptance]

## Licensing

- **Code**: MIT License (see LICENSE)
- **Paper**: see paper/ for licensing
- **V-Dem data**: Creative Commons Attribution 4.0 International (CC BY 4.0).
  Original V-Dem v15 data: Coppedge et al. (2025), "V-Dem [Country–Year/Country–Date]
  Dataset v15", https://doi.org/10.23696/vdemds25
