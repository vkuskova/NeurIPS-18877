# Synthetic Pareto-frontier sweep

## What this generates

`artifacts/synthetic_v1f_frontier.pkl` — 45 synthetic-DGP runs spanning
9 audited cells × 5 seeds, plus M1 and M1.5 baseline 5-seed-mean
test MSEs. Used in Section 6 of the paper.

## How to run

From the package root:

```
python code/synthetic/synthetic_v1f_sweep.py
```

The script generates its own data in code using `np.random.default_rng(42)` —
no external data files required. Output is saved to
`artifacts/synthetic_v1f_frontier.pkl`.

## Expected runtime

- CPU: ~4.5 hours
- Colab GPU: ~30 minutes

## DGP

Synthetic panel: 30 units × 50 timesteps × 1 endogenous Y + 1 exogenous X.
Three regimes determined by `Y_{t-1}` thresholds at -0.6 and 0.6, with
distinct AR coefficients for Y and beta coefficients for X per regime.
Per-unit intercepts ~ N(0, 0.5²); innovations ~ N(0, 0.3²); covariate is
AR(0.9) with N(0, 0.5²) innovations.

The full DGP specification is reproduced in Appendix B of the paper.

## Hyperparameters

- Hidden width: 32; Lag window L=5; Adam lr=3e-3; weight decay=1e-4
- Batch size: 128; Training epochs: 300 per run
- Soft-only mode: lambda_Y=0.20 with target L_Y=0.55
- Projection modes: hard-project per-edge weights every epoch or batch
  to spectral-norm 0.85
- Three lambda_M values: {0.0, 0.05, 0.10}
- Five seeds per cell: {0, 1, 2, 3, 4}
- Gating temperature regularizer lambda_phi=0.005
- Frozen regime centers: (-0.6, 0.0, 0.6)

## Determinism

Bit-exact CPU reproducibility: seeds set via `torch.manual_seed` and
`np.random.seed` immediately before each model instantiation. Data
generation uses `np.random.default_rng(DGP_SEED)` and is identical
across all training runs.

## Output structure

```python
{
    'm15_test_mse':  float,                        # M1.5 5-seed mean
    'm1_test_mse':   float,                        # M1 5-seed mean
    'results': {
        ('soft_only', 0.0):                [run_dict, ...],   # 5 seeds
        ('soft_only', 0.05):               [...],
        ('soft_only', 0.1):                [...],
        ('projection_every_epoch', 0.0):   [...],
        ('projection_every_epoch', 0.05):  [...],
        ('projection_every_epoch', 0.1):   [...],
        ('projection_every_batch', 0.0):   [...],
        ('projection_every_batch', 0.05):  [...],
        ('projection_every_batch', 0.1):   [...],
    }
}
```

Each `run_dict` contains: `seed`, `mode`, `lam_M`, `test_mse`, `routing`,
audited theorem constants `L_max`, `LX_max`, `L_X_per_regime`,
`L_Y_per_regime`, `M`, `tau`, `L_phi`, `L_phi_eq_max`, `L_phi_eq_M`,
`L_phi_eq_per_regime`, `L_eq_per_regime`, `L_eq_max`, `B_trans`,
`alpha_star`, `alpha_emp`, `rho`.

The keys (other than the descriptive `routing` field) match the saved
`artifacts/synthetic_v1f_frontier.pkl` exactly, so a fresh run produces
the same structure ready for `verify_paper_numbers.py`.
