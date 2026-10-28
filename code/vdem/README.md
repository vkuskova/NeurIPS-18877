# V-Dem experiments

## What's here

Five scripts, each producing one or more artifacts used by the paper:

| Script | Output | Section / table |
|--------|--------|-----------------|
| `vdem_v1h_v3_cell.py` | `artifacts/vdem_v1h_frontier.pkl` | §7.4 baselines, §7.6 stability conditions, §8 V-Dem cross-experiment |
| `vdem_v1h_postanalysis.py` | `artifacts/vdem_v1h_postanalysis.pkl` | §7.5 headline DM, Table 5.4 per-regime DM |
| `regime_stratified_diagnostic.py` | (writes to `vdem_v1h_frontier.pkl`'s `baselines` field) | §7.3 / Table 5.3 (+3.02% diagnostic) |
| `certified_bound_audit.py` | `artifacts/certified_audit_results.pkl` | App D.2 certified bound vs probe |
| `build_figure.py` | `figures/figure_diagnostic.pdf` | Figure 1 |

## How to run (full pipeline)

From the package root:

```
# 1. Train the 50-run sweep + write baselines
python code/vdem/vdem_v1h_v3_cell.py

# 2. Re-load trained models, save predictions for paired DM
python code/vdem/vdem_v1h_postanalysis.py

# 3. Compute the certified-bound audit (Appendix D.2)
python code/vdem/certified_bound_audit.py

# 4. Build the paper figure
python code/vdem/build_figure.py
```

`scripts/run_all.sh` wires these in order along with the synthetic sweep.

## Expected runtimes

| Stage | CPU | Colab GPU |
|-------|-----|-----------|
| `vdem_v1h_v3_cell.py` | ~3.5 hours | ~30 min |
| `vdem_v1h_postanalysis.py` | ~2 min | ~30 sec |
| `certified_bound_audit.py` | ~3 min | ~30 sec |
| `build_figure.py` | ~15 sec | ~10 sec |

The V-Dem sweep is dominated by the per-batch Jacobian penalty
(`jacobian_softmax_norm_Y`) which evaluates per-sample autograd on each
batch. GPU is strongly recommended; the `vdem_v1h_v3_cell.py` script auto-
detects GPU via `torch.cuda.is_available()`.

## Hyperparameters

- Hidden width: 32; Lag window L=5
- Adam lr=3e-4; weight decay=1e-4; batch=128
- Pooled baseline epochs: 300
- Stratified baseline epochs: 900 (3× compensating for ~1/3 sample size per regime)
- M2 epochs: 300 per cell
- Spectral-norm targets: soft 0.55, projection 0.85
- Disagreement penalty lambda_M ∈ {0.0, 0.05, 0.10}
- Gating-temperature lambda_phi: 0.005
- 5 seeds per cell

Full hyperparameter rationale in Appendix E of the paper.

## Data path

All scripts in this directory read from
`data/merged_completevars_balanced_zero_missing.csv` (relative to the
package root). The Colab versions of these scripts (e.g.,
`vdem_v1h_v3_cell.py`'s `DATA_PATH`) reference the original Drive path
`/content/drive/MyDrive/panar-x-reproducibility/data/...` — change this to the
relative path or to your own absolute path before running locally.

## Determinism

CPU: bit-exact reproducible. All seeds set via `torch.manual_seed` and
`np.random.seed` immediately before each model instantiation.

GPU: reproducible up to ~1e-6 in test MSE due to non-deterministic CUDA
matmul kernels. The verification script (`verify_paper_numbers.py`) uses
appropriate tolerances (typically 5e-5 to 5e-3 absolute) that absorb
GPU-level non-determinism.
