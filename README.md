# NeurIPS 2026 Submission 18877 — Reproducibility Repository

This repository contains the reproducibility package for:

> **Stability-Constrained Regime-Aware Forecasting for Heterogeneous Panel
> Time Series**
> Anonymous (NeurIPS 2026 submission #18877)

The paper introduces PANAR-X, a regime-aware additive forecasting architecture
for panel time series with a verifiable contraction certificate. It is
demonstrated on V-Dem democratic-development panel data and on a synthetic
Pareto-frontier sweep.

---

## What's in this repository

```
NeurIPS-18877/
├── README.md                              ← this file
├── main.pdf                               ← the paper (anonymized)
├── verify_paper_numbers.py                ← audit script (Path A entry point)
├── requirements.txt                       ← Python dependencies for verify
├── NeurIPS2026_18877_pipeline.ipynb       ← Colab notebook (Path B, Option 2)
├── PAPER_TO_CODE_TRACEABILITY.md          ← maps every paper claim → verify-script line → source code
├── BUNDLE_README.md                       ← detailed README from inside the bundle
└── panar-x-reproducibility.tar.gz         ← full reproducibility bundle (~1.1 MB)
```

The canonical artifact is the **bundle** (`panar-x-reproducibility.tar.gz`).
Everything else in this repository is either a copy of a file from inside the
bundle, surfaced for direct GitHub viewing, or — in the case of this README —
a landing page for the repository.

---

## Two reproduction paths

The package supports two paths depending on what you want to do.

### Path A — Audit only (recommended for reviewers)

The fastest path. **No training, no GPU, ~30 seconds total.** Loads the four
binary artifact pickles inside the bundle and asserts that 141 numerical
claims in the paper match the saved values exactly.

#### Option 1: From the command line

```bash
# 1. Clone or download this repository.
# 2. Extract the bundle:
tar xzf panar-x-reproducibility.tar.gz
cd panar-x-reproducibility

# 3. Install dependencies and run:
pip install -r requirements.txt
python verify_paper_numbers.py
```

Expected output: `141/141 PASS`.

#### Option 2: In Google Colab

If you don't have a local Python environment handy, you can run the audit in
Colab:

1. Open https://colab.research.google.com
2. Upload `panar-x-reproducibility.tar.gz` (the bundle from this repo) into
   the Colab session via *Files → Upload*
3. In a code cell, run:
   ```
   !tar xzf panar-x-reproducibility.tar.gz
   %cd panar-x-reproducibility
   !pip install -r requirements.txt --quiet
   !python verify_paper_numbers.py
   ```

Expected output: `141/141 PASS`.

### What the 141 checks cover

Every numerical claim in the paper, organized by section:

- Tables 1, 2, 3 (V-Dem regime-stratified diagnostic, paired DM tests, full Pareto frontier)
- Table 4 (synthetic baselines)
- Table 5 (synthetic 9-cell Pareto sweep)
- Audit constants in §4 (anchor paragraph) and §6.4 (V-Dem stability)
- Paired Diebold–Mariano statistics including per-regime tests
- Certified spectral-norm bounds (Appendix D.2)
- Figure 1 cumulative-advantage values
- V-Dem panel sizing (countries, year range, train/test split)
- Prose-level claims throughout: §1 abstract sign-reversal, §4 anchor constants,
  §6 narrative, §7 stress-test prose, §8 limitations

For a comprehensive cross-reference of which paper claim is verified by which
line of which file, see [`PAPER_TO_CODE_TRACEABILITY.md`](PAPER_TO_CODE_TRACEABILITY.md).

---

### Path B — Full replication (regenerate everything from scratch)

For reviewers who want to verify that the pipeline reproduces, not just that
the numbers in the paper match the saved artifacts. **This takes ~3.5 hours
on a GPU; we recommend Path A unless you specifically want to verify the
training pipeline.**

#### Option 1: Local machine with GPU

```bash
tar xzf panar-x-reproducibility.tar.gz
cd panar-x-reproducibility
pip install -r requirements.txt
bash scripts/run_all.sh
```

This runs all six pipeline stages in order: synthetic v1 baselines → synthetic
Pareto sweep → V-Dem 50-run sweep → V-Dem postanalysis → certified-bound audit
→ figure builder. Outputs go to `artifacts/` and `paper/`.

#### Option 2: Google Colab

The notebook [`NeurIPS2026_18877_pipeline.ipynb`](NeurIPS2026_18877_pipeline.ipynb)
is a Colab-ready version of the full pipeline with live-streaming progress in
each cell, plus stages for verification and a numerical-drift report.

To use it:

1. Upload `panar-x-reproducibility.tar.gz` to Colab via *Files → Upload*.
2. Open `NeurIPS2026_18877_pipeline.ipynb` in Colab. (Or open it directly from
   this repository: GitHub renders `.ipynb` files; click the file name and
   then the *Open in Colab* button if your browser has the extension, or
   download and upload the file manually.)
3. *Runtime → Change runtime type → T4 GPU* (or any GPU).
4. Run cells top-to-bottom.

Expected runtime: **~3.5 hours on a free T4 GPU**, dominated by the V-Dem
sweep (~2.9 hours alone). Total runtime exceeds the free-tier Colab idle
timeout of 90 minutes, so Colab Pro is recommended for the full replication.
The audit-only path (Path A) does not have this constraint.

#### Numerical drift on independent hardware

Re-running the pipeline on a GPU different from the one used to produce the
original archived artifacts will produce numerically similar but not bit-exact
results, due to non-deterministic CUDA matmul kernels. Empirically observed
drift on T4 GPU vs. the archived artifacts:

- Test MSE values: $\sim 10^{-5}$
- Audit constants based on max-over-samples Jacobian probes: up to $\sim 10^{-3}$
- All substantive paper claims (paired-DM significance, $L_{\text{eq,max}} < 1$
  in $5/5$ seeds across all $9$ audited cells, the sign-reversal diagnostic,
  the synthetic stress-test ordering): **reproduce independent of hardware**.

The notebook's stage 10 quantifies this drift between regenerated and original
artifacts; the README inside the bundle ([`BUNDLE_README.md`](BUNDLE_README.md))
covers it in more detail.

---

## What's in the bundle

The `panar-x-reproducibility.tar.gz` archive contains the full reproducibility
package: paper sources (LaTeX + compiled PDF), all training and analysis code,
the preprocessed V-Dem panel data, the five binary artifact pickles produced
by the original training runs, run scripts (Bash + Colab notebook), the
verification script, and a paper-to-code traceability map.

For the complete table of contents and detailed instructions for each path,
see [`BUNDLE_README.md`](BUNDLE_README.md) (a copy of the README from inside
the bundle, surfaced here for direct viewing without extraction).

---

## Software requirements

For the audit-only path: Python 3.10+ with `numpy`, `scipy`, `pandas`,
`scikit-learn`, `torch`. For full replication: same plus `matplotlib` for
figure generation. All pinned in `requirements.txt`.

---

## License

- **Code**: MIT License (see `LICENSE` inside the bundle)
- **V-Dem v15 data**: included under [Creative Commons Attribution 4.0
  International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/);
  Coppedge et al. (2025), *V-Dem [Country–Year/Country–Date] Dataset v15*,
  https://doi.org/10.23696/vdemds25
- **Paper**: NeurIPS submission, all rights reserved by the authors

---

## Contact

This repository is anonymized for double-blind review. Please direct any
questions through the OpenReview discussion forum for submission #18877.
