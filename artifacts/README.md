# Artifacts

This directory contains the four binary artifacts (pickled Python objects)
produced by the experiments described in the paper. Each pickle is a snapshot
of the trained-model audit constants and (where applicable) saved test-set
predictions, sufficient to reproduce every numerical claim in the paper
without retraining.

## File inventory

### `synthetic_v1f_frontier.pkl`  (~14 KB)

**What it is:** Output of the synthetic Pareto-frontier sweep, 45 runs total
(9 cells: 3 enforcement modes × 3 disagreement-penalty values, × 5 seeds).

**Produced by:** `code/synthetic/synthetic_v1f_sweep.py`

**Used in paper:** Tables 5.1 and 5.2; Section 6 audit constants; Section 6.3
slope claim; Section 8 cross-experiment table (synthetic columns).

**Structure:**
```python
{
    'm15_test_mse':  float,            # 5-seed mean test MSE for pooled M1.5
    'm1_test_mse':   float,            # 5-seed mean test MSE for Y-only M1
    'results': {
        ('soft_only',                0.0):  [run_seed_0, ..., run_seed_4],
        ('soft_only',                0.05): [...],
        ('soft_only',                0.1):  [...],
        ('projection_every_epoch',   0.0):  [...],
        ('projection_every_epoch',   0.05): [...],
        ('projection_every_epoch',   0.1):  [...],
        ('projection_every_batch',   0.0):  [...],
        ('projection_every_batch',   0.05): [...],
        ('projection_every_batch',   0.1):  [...],
    }
}
```

Each `run_seed_i` is a dict with keys `seed`, `test_mse`, `L_max`, `M`,
`L_phi_eq_max`, `L_phi_eq_M`, `L_eq_max`, `B_trans`, plus per-regime
constants. See `code/synthetic/synthetic_v1f_sweep.py` for the
generating code.


### `vdem_v1h_frontier.pkl`  (~17 KB)

**What it is:** Output of the V-Dem 50-run sweep (10 cells: 9 audited modes
× lambda_M values plus a hard-routing baseline, × 5 seeds), with
audited theorem constants per run and pooled baseline MSE.

**Produced by:** `code/vdem/vdem_v1h_v3_cell.py`

**Used in paper:** Section 7.4 (baselines), Section 7.5 (hard-routing
predictive baseline), Section 7.6 (stability conditions, all 9 cells),
Section 8 (V-Dem columns of cross-experiment table).

**Structure:**
```python
{
    'config':    { ... },                     # hyperparameters and run settings
    'baselines': {                            # all baselines with single-run or
        'RW':                          float, # 5-seed-mean test MSE
        'linear_Y_t-1':                float,
        'pooled_linear_M1':            float,
        'pooled_linear_M1.5':          float,
        'stratified_linear_M1.5':      float,
        'M1_NN':                       float,
        'pooled_NN_M1.5':              float,
        'stratified_NN_M1.5':          float,
    },
    'results': {
        ('hard_routing',             0.0):  [run_seed_0, ..., run_seed_4],
        ('soft_only',                0.0):  [...],
        ('soft_only',                0.05): [...],
        # ... 7 more cells
    }
}
```

**Note:** Each run dict contains `test_mse` (a scalar) but does NOT contain
per-observation predictions. Per-observation predictions are saved only in
`vdem_v1h_postanalysis.pkl` for the headline cell, used in the paired DM
tests (Table 5.4). Extending paired DM to all 9 cells requires modifying
`vdem_v1h_v3_cell.py` to save `pred.cpu().numpy()` per run and re-running
the sweep.


### `vdem_v1h_postanalysis.pkl`  (~170 KB)

**What it is:** V-Dem test-set predictions for all baselines and for the M2
architecture (5 seeds, unconstrained-soft cell), aligned to the same 1,510
test observations. Used to compute paired Diebold–Mariano statistics.

**Produced by:** `code/vdem/vdem_v1h_postanalysis.py`

**Used in paper:** Section 7.5 (headline DM); Table 5.4 (per-regime DM);
Figure 1C–E (cumulative paired squared-error advantage).

**Structure:**
```python
{
    'targets':            ndarray (1510,),     # test ΔY values
    'regime_assignment':  ndarray (1510,),     # in {0, 1, 2}
    'y_lag':              ndarray (1510,),     # Y_{t-1}
    'country_id':         ndarray (1510,),     # for traceability
    'year':               ndarray (1510,),     # for traceability
    'predictions': {
        'rw':                       ndarray (1510,),   # random walk
        'linear_Y_tm1_only':        ndarray (1510,),
        'linear_pooled_M1.5':       ndarray (1510,),
        'linear_stratified_M1.5':   ndarray (1510,),
        'NN_pooled_M1':             ndarray (1510,),
        'NN_pooled_M1.5':           ndarray (1510,),
        'NN_stratified_M1.5':       ndarray (1510,),
        'M2_per_seed':              list of 5 ndarrays (1510,) (PANAR-X M2),
        'M2_mean':                  ndarray (1510,),  # M2_per_seed averaged
    }
}
```


### `certified_audit_results.pkl`  (~400 B)

**What it is:** Per-regime certified spectral-norm Lipschitz upper bounds
vs gradient-probe sample-max estimates, computed on the same trained M2
models (3 seeds).

**Produced by:** `code/vdem/certified_bound_audit.py`

**Used in paper:** Appendix D.2 (certified bound vs gradient probe).

**Structure:**
```python
{
    'results': [
        {'seed': 0, 'probe': [L0, L1, L2], 'cert': [L0, L1, L2]},
        {'seed': 1, ...},
        {'seed': 2, ...},
    ],
    'pmax_seeds': [...],   # max-over-regime for each seed (probe)
    'cmax_seeds': [...],   # max-over-regime for each seed (cert)
    'training':  str       # description of training procedure
}
```

## Recomputing artifacts

Each artifact can be regenerated by running its source script as documented
in `code/<subdir>/README.md`. See top-level `scripts/run_all.sh` for the
full pipeline.

## SHA-256 (verify integrity)

```
8fad07108ef707544321b88104e185f655206a0c2278f953df482fd35719579b  certified_audit_results.pkl
68c7c1f9dae367ff342a91aa8ca82a66280dccdd3e9388a118bea20fdfd3eb92  synthetic_v1f_frontier.pkl
39fbb88dc739b905e62bed39f722386a3bb9c6531b2350739b7be273a13e5b03  vdem_v1h_frontier.pkl
018a8d24d7af18f8f3f85cdc4bed8ed54166c0e55dc9170f91df29478d762e0a  vdem_v1h_postanalysis.pkl
```


## Camera-ready additions

Four subdirectories were added for the NeurIPS 2026 camera-ready; each holds
CSV/JSON outputs written by the corresponding locked design in `../designs/`
(plus model checkpoints where the design trained models). File-level
descriptions are in the top-level README (section "Camera-ready update") and
the claim-level mapping in `../PAPER_TO_CODE_TRACEABILITY.md`.

- `rebuttal_R1/` (design_18877_R1): panel-appropriate inference, three-decimal recomputation, step-0 reconciliation.
- `rebuttal_S2/` (design_18877_S2): 24-variant diagnostic grid, stability checks, stratification test, four M2 retraining variants with checkpoints.
- `rv_screen_S1/` (design_RV_S1): realized-volatility screen (two gates).
- `rv_f1/` (design_RV_F1): realized-volatility confirmatory run with certificate audit, predictive tests, metadata and checkpoints.

- `kdd_s81_freshseed.pkl`: 20-seed V-Dem runs on the pre-committed fresh seed set (100-119): per-seed and ensemble predictions for M1, pooled M1.5, stratified M1.5, and M2, plus the panel-appropriate fair comparison table. Read with the stub unpickler in `verify_camera_ready_numbers.py` if torch is not installed.
- `vdem_v1h_postanalysis_20seed.pkl`: the primary 20-seed V-Dem block (seeds 0-19), same structure as the fresh block; SHA-256 dff52af6dcdf8a6603d05296465e52a6ab243718084bf825b64a2159739a7610, checked by `verify_camera_ready_numbers.py`.
