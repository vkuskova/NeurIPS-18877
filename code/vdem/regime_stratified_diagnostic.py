# =============================================================================
# Regime-stratified linear diagnostic
#
# Hypothesis (ICML paper analog): X→Y functions may have opposite signs or
# shapes in different regimes. A regime-blind M1.5 averages across regimes
# and sees the effects partially cancel, making X look uninformative.
# A regime-aware model would recover the signal.
#
# Test: fit linear M1.5 (Y+X lags → ΔY) SEPARATELY on each Y_{t-1} tertile.
# Route test observations to their stratum's model. Compare to pooled M1.5.
#
# If stratified beats pooled: regime-conditional signal exists -> M2 viable
# If stratified ties pooled: no regime-conditional structure to exploit
# If stratified worse than pooled: per-stratum overfitting outweighs gain
#
# Cell is pure linear (np.linalg.lstsq); no NN, no overfitting concerns.
# Runs in ~30 seconds on V-Dem.
# =============================================================================
# Cell is pure linear (np.linalg.lstsq); no NN, no overfitting concerns.
# Runs in ~30 seconds on V-Dem.
# =============================================================================
import os
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Path resolution: prefer local (package-relative) paths; fall back to Colab.
# -----------------------------------------------------------------------------
_LOCAL_DATA = Path('data/merged_completevars_balanced_zero_missing.csv')
_COLAB_DATA = Path('/content/drive/MyDrive/panar-x-reproducibility/data/merged_completevars_balanced_zero_missing.csv')

if _LOCAL_DATA.exists():
    DATA_PATH = str(_LOCAL_DATA)
elif os.path.exists('/content/drive') or _COLAB_DATA.exists():
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception:
        pass
    DATA_PATH = str(_COLAB_DATA)
else:
    raise FileNotFoundError(
        f"Could not find data: tried {_LOCAL_DATA} (relative) and "
        f"/content/drive (Colab). Run from the reproducibility package root."
    )

print(f"DATA_PATH: {DATA_PATH}")

# -----------------------------------------------------------------------------
# Configuration: same as v1h V-Dem setup
# -----------------------------------------------------------------------------
Y_VAR = 'v2xel_frefair'
X_VARS = ['v2csantimv', 'v2jureform', 'v2psorgs', 'v2mecorrpt']
L_LAG = 5
TEST_YEARS_PER_COUNTRY = 10
N_X = len(X_VARS)

# -----------------------------------------------------------------------------
# Load and prepare data (mirrors v1h pipeline)
# -----------------------------------------------------------------------------
print("Loading V-Dem dataset...")
df = pd.read_csv(DATA_PATH)
df = df.sort_values(['country_id', 'year']).reset_index(drop=True)

def build_panel_tensors(df, y_var, x_vars, L, test_years_per_country):
    train, test = [], []
    for cid in df['country_id'].unique():
        sub = df[df['country_id'] == cid].sort_values('year').reset_index(drop=True)
        if len(sub) < L + test_years_per_country + 1:
            continue
        country_test_threshold = sub['year'].iloc[-test_years_per_country]
        y_arr = sub[y_var].values
        x_arr = sub[x_vars].values
        years = sub['year'].values
        for i in range(L, len(sub)):
            y_window = y_arr[i-L:i+1]
            x_window = x_arr[i-L:i]
            if np.isnan(y_window).any() or np.isnan(x_window).any():
                continue
            row = {
                'y_hist': y_window[:-1],   # length L
                'x_hist': x_window,         # (L, N_X)
                'y_lag1': y_window[-2],     # Y_{t-1}, used for regime assignment
                'dY': y_window[-1] - y_window[-2],
            }
            if years[i] < country_test_threshold:
                train.append(row)
            else:
                test.append(row)
    return train, test

train_data, test_data = build_panel_tensors(
    df, Y_VAR, X_VARS, L_LAG, TEST_YEARS_PER_COUNTRY)
print(f"Train: {len(train_data)}, Test: {len(test_data)}")

def to_arrays(data):
    Y_lags = np.stack([r['y_hist'] for r in data])      # (N, L)
    X_lags = np.stack([r['x_hist'] for r in data])       # (N, L, N_X)
    y_lag1 = np.array([r['y_lag1'] for r in data])       # (N,)
    dY = np.array([r['dY'] for r in data])                # (N,)
    X_flat = X_lags.reshape(len(X_lags), -1)             # (N, L*N_X)
    return Y_lags, X_flat, y_lag1, dY

Y_lags_tr, X_lags_tr, ylag1_tr, dY_tr = to_arrays(train_data)
Y_lags_te, X_lags_te, ylag1_te, dY_te = to_arrays(test_data)

# Standardize X using TRAINING set only (per-feature, across all lag positions)
x_mean = X_lags_tr.mean(axis=0)
x_std = X_lags_tr.std(axis=0)
x_std = np.where(x_std < 1e-8, 1.0, x_std)
X_lags_tr = (X_lags_tr - x_mean) / x_std
X_lags_te = (X_lags_te - x_mean) / x_std

# -----------------------------------------------------------------------------
# Regime assignment: tertiles of training Y_{t-1}
# -----------------------------------------------------------------------------
b1, b2 = np.percentile(ylag1_tr, [100/3, 100*2/3])
print(f"\nRegime boundaries (Y_{{t-1}} tertiles): "
      f"low<{b1:.3f} | mid in [{b1:.3f},{b2:.3f}) | high>={b2:.3f}")

def assign_regime(y_lag1, b1, b2):
    r = np.zeros(len(y_lag1), dtype=int)
    r[y_lag1 < b1] = 0
    r[(y_lag1 >= b1) & (y_lag1 < b2)] = 1
    r[y_lag1 >= b2] = 2
    return r

r_tr = assign_regime(ylag1_tr, b1, b2)
r_te = assign_regime(ylag1_te, b1, b2)
print(f"Train regime occupancy: {[f'{(r_tr==k).sum()}' for k in range(3)]}")
print(f"Test regime occupancy:  {[f'{(r_te==k).sum()}' for k in range(3)]}")

# -----------------------------------------------------------------------------
# Pooled M1.5: linear regression of dY on (Y lags, X lags), all data
# -----------------------------------------------------------------------------
def fit_predict_lin(X_train, y_train, X_test):
    """Fit OLS with intercept and return test predictions + coefficients."""
    A_tr = np.column_stack([np.ones(len(X_train)), X_train])
    beta, *_ = np.linalg.lstsq(A_tr, y_train, rcond=None)
    A_te = np.column_stack([np.ones(len(X_test)), X_test])
    pred = A_te @ beta
    return pred, beta

XY_tr = np.hstack([Y_lags_tr, X_lags_tr])
XY_te = np.hstack([Y_lags_te, X_lags_te])

pooled_pred_te, pooled_beta = fit_predict_lin(XY_tr, dY_tr, XY_te)
pooled_mse = float(((pooled_pred_te - dY_te) ** 2).mean())
rw_mse = float((dY_te ** 2).mean())

# Pooled M1 (Y-only) for comparison
m1_pred_te, m1_beta = fit_predict_lin(Y_lags_tr, dY_tr, Y_lags_te)
m1_mse = float(((m1_pred_te - dY_te) ** 2).mean())

print(f"\n{'baseline':<35s}  {'MSE':>10}  {'vs RW':>10}")
print(f"{'-'*35}  {'-'*10}  {'-'*10}")
print(f"{'RW (predict 0)':<35s}  {rw_mse:>10.6f}  {'(ref)':>10}")
print(f"{'Pooled linear M1 (Y lags)':<35s}  {m1_mse:>10.6f}  "
      f"{(rw_mse-m1_mse)/rw_mse*100:>+9.2f}%")
print(f"{'Pooled linear M1.5 (Y+X lags)':<35s}  {pooled_mse:>10.6f}  "
      f"{(rw_mse-pooled_mse)/rw_mse*100:>+9.2f}%")

# -----------------------------------------------------------------------------
# Stratified M1.5: fit one linear model per regime; route test obs accordingly
# -----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("STRATIFIED LINEAR M1.5 (one model per regime)")
print("=" * 78)

stratified_pred_te = np.full(len(dY_te), np.nan)
stratified_betas = {}

for r in range(3):
    tr_mask = (r_tr == r)
    te_mask = (r_te == r)
    n_tr = tr_mask.sum()
    n_te = te_mask.sum()
    
    if n_tr < 50 or n_te < 5:
        # Fall back to pooled prediction for sparse regimes
        print(f"  Regime {r}: too few obs (train={n_tr}, test={n_te}); "
              f"using pooled prediction")
        stratified_pred_te[te_mask] = pooled_pred_te[te_mask]
        continue
    
    XY_tr_r = XY_tr[tr_mask]
    dY_tr_r = dY_tr[tr_mask]
    XY_te_r = XY_te[te_mask]
    dY_te_r = dY_te[te_mask]
    
    pred_r, beta_r = fit_predict_lin(XY_tr_r, dY_tr_r, XY_te_r)
    stratified_pred_te[te_mask] = pred_r
    stratified_betas[r] = beta_r
    
    mse_r = float(((pred_r - dY_te_r) ** 2).mean())
    pooled_mse_r = float(((pooled_pred_te[te_mask] - dY_te_r) ** 2).mean())
    rw_mse_r = float((dY_te_r ** 2).mean())
    
    print(f"\n  Regime {r}: train={n_tr}, test={n_te}")
    print(f"    RW MSE in this regime:        {rw_mse_r:.6f}")
    print(f"    Pooled M1.5 MSE in regime:    {pooled_mse_r:.6f}")
    print(f"    Stratified M1.5 MSE in regime: {mse_r:.6f}")
    print(f"    Stratified vs Pooled: "
          f"{(pooled_mse_r - mse_r)/pooled_mse_r*100:+.2f}%")

# Overall stratified MSE on full test set
strat_mse = float(((stratified_pred_te - dY_te) ** 2).mean())
print(f"\n{'-' * 78}")
print(f"OVERALL: pooled M1.5 = {pooled_mse:.6f}")
print(f"OVERALL: stratified  = {strat_mse:.6f}")
print(f"OVERALL: improvement = {(pooled_mse - strat_mse)/pooled_mse*100:+.2f}%")

# -----------------------------------------------------------------------------
# Coefficient sign check: do X→ΔY relationships flip across regimes?
# -----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SIGN-CHANGE CHECK: do X coefficients flip across regimes?")
print("=" * 78)
print("Each entry is the lag-1 coefficient (most recent X) on ΔY.")
print("If signs flip across regimes, regime-blind models cancel the signal.\n")

# Coefficient layout in the design matrix [intercept, Y_lags, X_lags_flat]:
#   - intercept at column 0
#   - Y lags at columns 1..L (Y_{t-L} at col 1, Y_{t-1} at col L)
#   - X_lags_flat starts at column 1+L
#
# X_lags_flat is X_lags.reshape(N, -1). With X_lags shape (N, L, N_X), this
# produces a LAG-MAJOR flatten: X_flat[n, l*N_X + j] = X_lags[n, l, j].
# So X variable j at lag l is at column: 1 + L + l*N_X + j
# Most recent lag is l=L-1 (position L-1 corresponds to time t-1):
#   X_{t-1, j} is at column 1 + L + (L-1)*N_X + j

# Show coefficients across all 5 lags to make the regime structure visible
print("\nCoefficients per regime, by X variable and lag")
print("(rows: X variable; columns: lags from t-L (oldest) to t-1 (most recent))")
for r in range(3):
    if r not in stratified_betas:
        print(f"\nRegime {r}: skipped (sparse)")
        continue
    beta = stratified_betas[r]
    print(f"\nRegime {r}:")
    print(f"  {'X variable':<22s}  " + "  ".join([f"  lag-{L_LAG-l}" for l in range(L_LAG)]))
    for j, xv in enumerate(X_VARS):
        print(f"  {xv:<22s}  ", end="")
        for l in range(L_LAG):
            pos = 1 + L_LAG + l * N_X + j
            print(f"  {beta[pos]:+.4f}", end="")
        print()

# Sign-disagreement check on the most-recent lag (l = L-1, X_{t-1})
print("\n" + "-" * 78)
print("Sign-disagreement check (most-recent lag, X_{t-1}, per X variable):")
print("-" * 78)
print(f"  {'X variable':<25s}  " + "  ".join([f"regime {r}" for r in range(3)]))
n_disagree_recent = 0
for j, xv in enumerate(X_VARS):
    pos = 1 + L_LAG + (L_LAG - 1) * N_X + j
    coefs = []
    print(f"  Δ{xv:<22s} ", end="")
    for r in range(3):
        if r in stratified_betas:
            c = stratified_betas[r][pos]
            coefs.append(c)
            print(f"  {c:+.4f}", end="")
        else:
            print(f"      N/A", end="")
    if len(set(np.sign(c) for c in coefs)) > 1:
        n_disagree_recent += 1
        print("  <-- SIGN FLIP")
    else:
        print()

# Sign-disagreement check across ANY lag (broader test for sign-flip patterns)
print("\nSign-disagreement check (across any lag, per X variable):")
n_disagree_any = 0
for j, xv in enumerate(X_VARS):
    flips_at_lags = []
    for l in range(L_LAG):
        pos = 1 + L_LAG + l * N_X + j
        signs = {r: int(np.sign(stratified_betas[r][pos])) for r in stratified_betas
                 if abs(stratified_betas[r][pos]) > 1e-4}  # only meaningful coefs
        if len(set(signs.values())) > 1:
            flips_at_lags.append((l, signs))
    if flips_at_lags:
        n_disagree_any += 1
        print(f"  Δ{xv}:")
        for l, signs in flips_at_lags:
            sign_str = ' '.join([f'r{r}={s:+d}' for r,s in sorted(signs.items())])
            print(f"    lag t-{L_LAG-l}: {sign_str}")

# -----------------------------------------------------------------------------
# Verdict
# -----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)

improvement_pct = (pooled_mse - strat_mse) / pooled_mse * 100
print(f"Stratified M1.5 vs Pooled M1.5:    {improvement_pct:+.2f}%")
print(f"Sign flip at most-recent lag:      {n_disagree_recent}/{N_X} variables")
print(f"Sign flip at any lag:              {n_disagree_any}/{N_X} variables")
print()

if improvement_pct >= 3 and (n_disagree_recent >= 1 or n_disagree_any >= 2):
    print("HYPOTHESIS SUPPORTED: regime-conditional signal exists in V-Dem X channel.")
    print("  Stratified linear beats pooled by a clear margin.")
    print("  At least one X variable has opposite sign in different regimes.")
    print("  -> M2 architecture should be able to exploit this; v1h sweep is justified.")
elif improvement_pct >= 1:
    print("HYPOTHESIS WEAKLY SUPPORTED: small regime-conditional signal exists.")
    print("  Stratified linear shows some gain but it's marginal.")
    print("  Coefficient magnitudes differ by regime even if signs don't fully flip.")
    print("  -> M2 may show small improvements; consider whether worth committing compute.")
else:
    print("HYPOTHESIS NOT SUPPORTED: no detectable regime-conditional signal.")
    print("  Pooled and stratified perform similarly.")
    print("  X channel doesn't have regime-conditional structure that M2 could recover.")
    print("  -> The V-Dem null result reflects genuine signal absence, not architectural")
    print("     limitations of the regime-blind M1.5 baseline.")
