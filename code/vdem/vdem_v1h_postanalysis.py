# =============================================================================
# V-Dem v1h post-analysis: alternative metrics for small structured improvements
#
# The headline sweep showed M2 beats pooled NN M1.5 by ~2.4% MSE on ΔY,
# statistically significant across 9 audited cells. This cell computes more
# sensitive measures that may be more revealing for a near-zero-mean target:
#   1. Skill scores vs RW (rescaled MSE for interpretability)
#   2. Directional accuracy (sign prediction; balanced accuracy)
#   3. Spearman rank correlation (model vs actual ordering)
#   4. Diebold-Mariano test (paired comparison; tighter CIs than seed-to-seed)
#   5. Regime-conditional MSE (where does the gain live?)
#   6. Tail performance (|ΔY_actual| > 0.05; substantively important cases)
#
# To compute these we need PER-OBSERVATION PREDICTIONS, which the original
# sweep didn't save. So we retrain the headline cells: pooled M1, pooled M1.5,
# stratified NN M1.5, and M2 (soft_only, lam_M=0) across 5 seeds. ~10 minutes.
# =============================================================================
import os
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp_stats

# -----------------------------------------------------------------------------
# Path resolution: prefer local (package-relative) paths; fall back to Colab.
# -----------------------------------------------------------------------------
_LOCAL_DATA = Path('data/merged_completevars_balanced_zero_missing.csv')
_LOCAL_ARTIFACTS = Path('artifacts')
_COLAB_PROJECT_ROOT = Path('/content/drive/MyDrive/panar-x-reproducibility')

if _LOCAL_DATA.exists():
    DATA_PATH = str(_LOCAL_DATA)
    _LOCAL_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = _LOCAL_ARTIFACTS / 'vdem_v1h_postanalysis.pkl'
elif _COLAB_PROJECT_ROOT.exists() or os.path.exists('/content/drive'):
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception:
        pass
    PROJECT_ROOT = _COLAB_PROJECT_ROOT
    VDEM_DIR = PROJECT_ROOT / 'results' / 'regime_panar' / 'vdem_v1h'
    DATA_PATH = str(PROJECT_ROOT / 'data' / 'merged_completevars_balanced_zero_missing.csv')
    SAVE_PATH = VDEM_DIR / 'vdem_v1h_postanalysis.pkl'
else:
    raise FileNotFoundError(
        f"Could not find data: tried {_LOCAL_DATA} (relative) and "
        f"/content/drive (Colab). Run from the reproducibility package root."
    )

print(f"DATA_PATH: {DATA_PATH}")
print(f"SAVE_PATH: {SAVE_PATH}")

# -----------------------------------------------------------------------------
# Configuration (must match the v1h sweep exactly)
# -----------------------------------------------------------------------------
Y_VAR = 'v2xel_frefair'
X_VARS = ['v2csantimv', 'v2jureform', 'v2psorgs', 'v2mecorrpt']
L_LAG = 5
TEST_YEARS_PER_COUNTRY = 10
N_X = len(X_VARS)
TAU_INIT = 0.5
LAM_PHI_DEFAULT = 0.005
BASELINE_EPOCHS_POOLED = 300
BASELINE_EPOCHS_STRATIFIED = 900
M2_EPOCHS = 300
SEEDS = [0, 1, 2, 3, 4]

# -----------------------------------------------------------------------------
# Load data + build tensors (same as v1h)
# -----------------------------------------------------------------------------
print("Loading V-Dem dataset...")
df = pd.read_csv(DATA_PATH).sort_values(['country_id', 'year']).reset_index(drop=True)

def build_panel_tensors(df, y_var, x_vars, L, test_years_per_country):
    train_data, test_data = [], []
    for cid in df['country_id'].unique():
        sub = df[df['country_id'] == cid].sort_values('year').reset_index(drop=True)
        if len(sub) < L + test_years_per_country + 1:
            continue
        country_test_threshold = sub['year'].iloc[-test_years_per_country]
        y_arr = sub[y_var].values
        x_arr = sub[x_vars].values
        years = sub['year'].values
        country_name = sub['country_name'].iloc[0]
        for i in range(L, len(sub)):
            y_window = y_arr[i-L:i+1]
            x_window = x_arr[i-L:i]
            if np.isnan(y_window).any() or np.isnan(x_window).any():
                continue
            y_lag = y_window[-2]
            row = {
                'y_hist': y_window[:-1],
                'x_hist': x_window,
                'y_target': y_window[-1] - y_lag,   # ΔY
                'y_lag': y_lag,
                'country_id': cid,
                'country_name': country_name,
                'year': years[i],
            }
            if years[i] < country_test_threshold:
                train_data.append(row)
            else:
                test_data.append(row)
    return train_data, test_data

train_data, test_data = build_panel_tensors(df, Y_VAR, X_VARS, L_LAG, TEST_YEARS_PER_COUNTRY)

def to_arrays(data):
    Y_hist = np.stack([r['y_hist'] for r in data])[:, :, None]
    X_hist = np.stack([r['x_hist'] for r in data])
    Y_target = np.array([r['y_target'] for r in data])
    Y_lag = np.array([r['y_lag'] for r in data])
    cid = np.array([r['country_id'] for r in data])
    yrs = np.array([r['year'] for r in data])
    return Y_hist, X_hist, Y_target, Y_lag, cid, yrs

Y_h_tr, X_h_tr, dY_tr, Y_lag_tr, c_tr, yr_tr = to_arrays(train_data)
Y_h_te, X_h_te, dY_te, Y_lag_te, c_te, yr_te = to_arrays(test_data)
print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# Standardize X
X_flat_tr = X_h_tr.reshape(-1, N_X)
x_mean = X_flat_tr.mean(axis=0)
x_std = X_flat_tr.std(axis=0)
x_std = np.where(x_std < 1e-8, 1.0, x_std)
X_h_tr = (X_h_tr - x_mean) / x_std
X_h_te = (X_h_te - x_mean) / x_std

# Regime boundaries (training Y_{t-1} tertiles)
regime_centers = (
    float(np.percentile(Y_lag_tr, 100/6)),
    float(np.percentile(Y_lag_tr, 50)),
    float(np.percentile(Y_lag_tr, 100*5/6)),
)
b1, b2 = (float(np.percentile(Y_lag_tr, 100/3)),
          float(np.percentile(Y_lag_tr, 100*2/3)))

def assign_regime_hard(y_lag1):
    r = np.zeros(len(y_lag1), dtype=int)
    r[y_lag1 < b1] = 0
    r[(y_lag1 >= b1) & (y_lag1 < b2)] = 1
    r[y_lag1 >= b2] = 2
    return r

r_tr_hard = assign_regime_hard(Y_lag_tr)
r_te_hard = assign_regime_hard(Y_lag_te)

# -----------------------------------------------------------------------------
# Baselines: predictions per observation (not just MSE)
# -----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("RETRAINING BASELINES TO RECOVER PER-OBSERVATION PREDICTIONS")
print("=" * 78)

# RW = predict 0
pred_rw = np.zeros(len(dY_te))

# Pooled linear M1.5 (closed form; no seed needed)
def fit_predict_lin(X_train, y_train, X_test):
    A_tr = np.column_stack([np.ones(len(X_train)), X_train])
    beta, *_ = np.linalg.lstsq(A_tr, y_train, rcond=None)
    A_te = np.column_stack([np.ones(len(X_test)), X_test])
    return A_te @ beta

Y_lags_tr_2d = Y_h_tr[:, :, 0]
Y_lags_te_2d = Y_h_te[:, :, 0]
X_lags_tr_2d = X_h_tr.reshape(len(X_h_tr), -1)
X_lags_te_2d = X_h_te.reshape(len(X_h_te), -1)

YX_tr = np.hstack([Y_lags_tr_2d, X_lags_tr_2d])
YX_te = np.hstack([Y_lags_te_2d, X_lags_te_2d])

pred_lin_pooled = fit_predict_lin(YX_tr, dY_tr, YX_te)
print(f"  Pooled linear M1.5 trained.")

# Stratified linear M1.5
pred_lin_stratified = np.full(len(dY_te), np.nan)
for r in range(3):
    tr_mask = r_tr_hard == r
    te_mask = r_te_hard == r
    if tr_mask.sum() < 50 or te_mask.sum() < 5:
        pred_lin_stratified[te_mask] = pred_lin_pooled[te_mask]
        continue
    pred_lin_stratified[te_mask] = fit_predict_lin(
        YX_tr[tr_mask], dY_tr[tr_mask], YX_te[te_mask])
print(f"  Stratified linear M1.5 trained.")

# NN architectures
class M1_YNet(nn.Module):
    def __init__(self, L, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(L, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, Y_hist):
        return self.bias + self.net(Y_hist[:, :, 0]).squeeze(-1)

class M15_PANAR(nn.Module):
    def __init__(self, L, n_x, hidden=32):
        super().__init__()
        self.f_yy = nn.Sequential(nn.Linear(L, hidden), nn.ReLU(),
                                   nn.Linear(hidden, 1))
        self.f_xy = nn.ModuleList([
            nn.Sequential(nn.Linear(L, hidden), nn.ReLU(),
                          nn.Linear(hidden, 1))
            for _ in range(n_x)
        ])
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, Y_hist, X_hist):
        out = self.bias + self.f_yy(Y_hist[:, :, 0]).squeeze(-1)
        for j, fx in enumerate(self.f_xy):
            out = out + fx(X_hist[:, :, j]).squeeze(-1)
        return out

def train_simple(model, Y_h, X_h, target, epochs=300, lr=3e-4,
                  weight_decay=1e-4, batch=128, use_x=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Yh = torch.tensor(Y_h, dtype=torch.float32)
    Xh = torch.tensor(X_h, dtype=torch.float32) if use_x else None
    Yt = torch.tensor(target, dtype=torch.float32)
    N = len(Yh)
    for ep in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, N, batch):
            idx = perm[s:s+batch]
            opt.zero_grad()
            pred = model(Yh[idx], Xh[idx]) if use_x else model(Yh[idx])
            loss = F.mse_loss(pred, Yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model

# Pooled NN M1 (single seed=0; deterministic comparison baseline)
torch.manual_seed(0); np.random.seed(0)
m1_nn = train_simple(M1_YNet(L_LAG), Y_h_tr, X_h_tr, dY_tr,
                      epochs=BASELINE_EPOCHS_POOLED, use_x=False)
with torch.no_grad():
    pred_m1_nn = m1_nn(torch.tensor(Y_h_te, dtype=torch.float32)).numpy()
print(f"  M1 NN (Y-only) trained.")

# Pooled NN M1.5
torch.manual_seed(0); np.random.seed(0)
m15_nn = train_simple(M15_PANAR(L_LAG, N_X), Y_h_tr, X_h_tr, dY_tr,
                       epochs=BASELINE_EPOCHS_POOLED, use_x=True)
with torch.no_grad():
    pred_m15_nn = m15_nn(torch.tensor(Y_h_te, dtype=torch.float32),
                          torch.tensor(X_h_te, dtype=torch.float32)).numpy()
print(f"  M1.5 NN (pooled, Y+X) trained.")

# Stratified NN M1.5 (3x epochs per regime)
pred_strat_nn = np.full(len(dY_te), np.nan)
for r in range(3):
    tr_mask = r_tr_hard == r
    te_mask = r_te_hard == r
    if tr_mask.sum() < 50 or te_mask.sum() < 5:
        pred_strat_nn[te_mask] = pred_m15_nn[te_mask]
        continue
    torch.manual_seed(0); np.random.seed(0)
    m15_r = train_simple(M15_PANAR(L_LAG, N_X),
                          Y_h_tr[tr_mask], X_h_tr[tr_mask], dY_tr[tr_mask],
                          epochs=BASELINE_EPOCHS_STRATIFIED, use_x=True)
    with torch.no_grad():
        pred_strat_nn[te_mask] = m15_r(
            torch.tensor(Y_h_te[te_mask], dtype=torch.float32),
            torch.tensor(X_h_te[te_mask], dtype=torch.float32)).numpy()
print(f"  Stratified NN M1.5 trained.")

# -----------------------------------------------------------------------------
# M2 (soft_only, lam_M=0) across 5 seeds; save per-observation predictions
# -----------------------------------------------------------------------------
class PerEdgeNetM2(nn.Module):
    def __init__(self, L, hidden=32):
        super().__init__()
        self.lin1 = nn.Linear(L, hidden)
        self.lin2 = nn.Linear(hidden, 1)
    def forward(self, x):
        return self.lin2(F.relu(self.lin1(x))).squeeze(-1)

class M2_RegimeAware(nn.Module):
    def __init__(self, L, n_x, hidden=32, n_regimes=3, tau_init=TAU_INIT,
                 frozen_centers=(0.2, 0.5, 0.8)):
        super().__init__()
        self.n_regimes = n_regimes
        self.n_x = n_x
        self.f_yy = nn.ModuleList([PerEdgeNetM2(L, hidden) for _ in range(n_regimes)])
        self.f_xy = nn.ModuleList([
            nn.ModuleList([PerEdgeNetM2(L, hidden) for _ in range(n_x)])
            for _ in range(n_regimes)
        ])
        self.bias = nn.Parameter(torch.zeros(n_regimes))
        self.register_buffer('centers', torch.tensor(frozen_centers,
                                                       dtype=torch.float32))
        self.log_tau = nn.Parameter(torch.tensor(np.log(tau_init),
                                                   dtype=torch.float32))
    def gating(self, psi):
        tau = torch.exp(self.log_tau)
        d2 = (psi.unsqueeze(-1) - self.centers.unsqueeze(0)) ** 2
        return F.softmax(-d2 / tau, dim=-1)
    def per_regime_outputs(self, Y_hist, X_hist):
        regime_outs = []
        for r in range(self.n_regimes):
            o = self.bias[r] + self.f_yy[r](Y_hist[:, :, 0])
            for j in range(self.n_x):
                o = o + self.f_xy[r][j](X_hist[:, :, j])
            regime_outs.append(o)
        return torch.stack(regime_outs, dim=-1)
    def forward(self, Y_hist, X_hist):
        psi = Y_hist[:, -1, 0]
        pi = self.gating(psi)
        per_r = self.per_regime_outputs(Y_hist, X_hist)
        return (pi * per_r).sum(dim=-1)

def train_m2_softonly_lam0(model, Y_h, X_h, target, epochs=M2_EPOCHS,
                             lr=3e-4, weight_decay=1e-4, batch=128,
                             lam_phi=LAM_PHI_DEFAULT,
                             lam_Y=0.20, target_LY_soft=0.55):
    """Soft routing, lam_M=0, lam_Y=0.20 (matches v1h sweep config)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Yh_t = torch.tensor(Y_h, dtype=torch.float32)
    Xh_t = torch.tensor(X_h, dtype=torch.float32)
    tgt = torch.tensor(target, dtype=torch.float32)
    N = len(Yh_t)
    
    def per_regime_predictor_fn(model, r):
        def fn(Y_h, X_h):
            o = model.bias[r] + model.f_yy[r](Y_h[:, :, 0])
            for j in range(model.n_x):
                o = o + model.f_xy[r][j](X_h[:, :, j])
            return o
        return fn
    
    def jacobian_softmax_norm_Y(model_fn, Y_hist_batch, X_hist_batch,
                                  tau_softmax=5.0):
        Y_h_in = Y_hist_batch.clone().requires_grad_(True)
        out = model_fn(Y_h_in, X_hist_batch)
        grad = torch.autograd.grad(out.sum(), Y_h_in, create_graph=True)[0]
        norms = grad.flatten(1).norm(dim=1)
        B = norms.shape[0]
        return (torch.logsumexp(tau_softmax * norms, dim=0) - np.log(B)) / tau_softmax
    
    for ep in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, N, batch):
            idx = perm[s:s+batch]
            opt.zero_grad()
            pred = model(Yh_t[idx], Xh_t[idx])
            loss = F.mse_loss(pred, tgt[idx])
            
            spec_pen_Y = torch.zeros((), device=Yh_t.device)
            for r in range(model.n_regimes):
                fn_r = per_regime_predictor_fn(model, r)
                sigma_Y = jacobian_softmax_norm_Y(
                    fn_r, Yh_t[idx], Xh_t[idx], tau_softmax=5.0)
                spec_pen_Y = spec_pen_Y + F.relu(sigma_Y - target_LY_soft) ** 2
            loss = loss + lam_Y * spec_pen_Y
            
            inv_tau = torch.exp(-model.log_tau)
            loss = loss + lam_phi * inv_tau ** 2
            
            loss.backward()
            opt.step()
    model.eval()
    return model

print("\n" + "=" * 78)
print("TRAINING M2 (soft_only, lam_M=0) ACROSS 5 SEEDS")
print("=" * 78)
m2_preds_per_seed = []
for seed in SEEDS:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    m2 = M2_RegimeAware(L_LAG, N_X, n_regimes=3, tau_init=TAU_INIT,
                         frozen_centers=regime_centers)
    m2 = train_m2_softonly_lam0(m2, Y_h_tr, X_h_tr, dY_tr)
    with torch.no_grad():
        pred = m2(torch.tensor(Y_h_te, dtype=torch.float32),
                   torch.tensor(X_h_te, dtype=torch.float32)).numpy()
    m2_preds_per_seed.append(pred)
    mse = float(((pred - dY_te) ** 2).mean())
    print(f"  seed {seed}: MSE = {mse:.5f}  ({time.time()-t0:.0f}s)")

# Mean prediction across seeds (this is what we'll use for headline metrics)
pred_m2_mean = np.mean(m2_preds_per_seed, axis=0)
m2_mse_mean = float(((pred_m2_mean - dY_te) ** 2).mean())
print(f"  Mean-of-seed-predictions ensemble MSE: {m2_mse_mean:.5f}")

# =============================================================================
# METRIC 1: SKILL SCORES vs RW
# =============================================================================
print("\n" + "=" * 78)
print("METRIC 1: SKILL SCORES (1 - MSE_model / MSE_RW); higher is better")
print("=" * 78)
mse_rw = float((dY_te ** 2).mean())

def skill_score(pred):
    mse = float(((pred - dY_te) ** 2).mean())
    return 1.0 - mse / mse_rw, mse

models = {
    'Linear in Y_{t-1} only': fit_predict_lin(Y_lag_tr.reshape(-1,1), dY_tr,
                                                Y_lag_te.reshape(-1,1)),
    'Pooled linear M1.5': pred_lin_pooled,
    'Stratified linear M1.5': pred_lin_stratified,
    'M1 NN (Y-only)': pred_m1_nn,
    'Pooled NN M1.5': pred_m15_nn,
    'Stratified NN M1.5': pred_strat_nn,
    'M2 (mean of 5 seeds)': pred_m2_mean,
}

print(f"  {'Model':<30s}  {'MSE':>10}  {'Skill':>10}")
print(f"  {'-'*30}  {'-'*10}  {'-'*10}")
print(f"  {'RW (reference)':<30s}  {mse_rw:>10.5f}  {0:>10.5f}")
skills = {}
for name, pred in models.items():
    skill, mse = skill_score(pred)
    skills[name] = skill
    print(f"  {name:<30s}  {mse:>10.5f}  {skill:>+10.5f}")

# =============================================================================
# METRIC 2: DIRECTIONAL ACCURACY
# =============================================================================
print("\n" + "=" * 78)
print("METRIC 2: DIRECTIONAL ACCURACY (sign prediction)")
print("=" * 78)
print("Class balance in test ΔY:")
print(f"  Positive: {(dY_te > 0).sum()} ({(dY_te > 0).mean()*100:.1f}%)")
print(f"  Negative: {(dY_te < 0).sum()} ({(dY_te < 0).mean()*100:.1f}%)")
print(f"  Exact zero: {(dY_te == 0).sum()} ({(dY_te == 0).mean()*100:.1f}%)")

# Drop exact zeros for sign metrics (no defined sign)
nonzero_mask = dY_te != 0
y_sign = np.sign(dY_te[nonzero_mask])
print(f"\n  Evaluating on N={nonzero_mask.sum()} non-zero observations")

print(f"\n  {'Model':<30s}  {'Acc':>8}  {'Bal Acc':>8}  {'Pos Recall':>10}  {'Neg Recall':>10}")
print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
for name, pred in models.items():
    pred_sign = np.sign(pred[nonzero_mask])
    # Accuracy
    acc = float((pred_sign == y_sign).mean())
    # Per-class recall
    pos_mask = y_sign > 0
    neg_mask = y_sign < 0
    pos_rec = float((pred_sign[pos_mask] > 0).mean()) if pos_mask.sum() > 0 else float('nan')
    neg_rec = float((pred_sign[neg_mask] < 0).mean()) if neg_mask.sum() > 0 else float('nan')
    bal_acc = 0.5 * (pos_rec + neg_rec) if not np.isnan(pos_rec + neg_rec) else float('nan')
    print(f"  {name:<30s}  {acc:>8.3f}  {bal_acc:>8.3f}  {pos_rec:>10.3f}  {neg_rec:>10.3f}")

# =============================================================================
# METRIC 3: SPEARMAN RANK CORRELATION
# =============================================================================
print("\n" + "=" * 78)
print("METRIC 3: SPEARMAN RANK CORRELATION (predicted vs actual ΔY)")
print("=" * 78)
print(f"  {'Model':<30s}  {'Spearman ρ':>12}  {'Kendall τ':>12}  {'p-value':>10}")
print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*10}")
for name, pred in models.items():
    sp_rho, sp_p = sp_stats.spearmanr(pred, dY_te)
    k_tau, k_p = sp_stats.kendalltau(pred, dY_te)
    print(f"  {name:<30s}  {sp_rho:>+12.4f}  {k_tau:>+12.4f}  {sp_p:>10.2e}")

# =============================================================================
# METRIC 4: DIEBOLD-MARIANO TEST
# =============================================================================
print("\n" + "=" * 78)
print("METRIC 4: DIEBOLD-MARIANO TEST (paired comparison of squared errors)")
print("=" * 78)

def diebold_mariano(pred_a, pred_b, y, h=1):
    """Test whether forecast A has smaller MSE than forecast B.
    Positive DM stat with low p means A is significantly better than B.
    h=1 for one-step-ahead forecast (V-Dem ΔY is one-step).
    Uses Newey-West HAC variance estimator."""
    e_a = (pred_a - y) ** 2
    e_b = (pred_b - y) ** 2
    d = e_b - e_a   # positive = A better than B
    n = len(d)
    d_bar = d.mean()
    
    # Newey-West variance with lag = h-1 = 0 for one-step
    if h == 1:
        var_d = d.var(ddof=1) / n
    else:
        gamma_0 = ((d - d_bar) ** 2).mean()
        var_sum = gamma_0
        for lag in range(1, h):
            gamma_k = ((d[lag:] - d_bar) * (d[:-lag] - d_bar)).mean()
            var_sum += 2 * (1 - lag/h) * gamma_k
        var_d = var_sum / n
    
    if var_d <= 0:
        return float('nan'), float('nan'), d_bar
    dm_stat = d_bar / np.sqrt(var_d)
    p_two_sided = 2 * (1 - sp_stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_two_sided, d_bar

# Pairs of interest:
# - M2 vs Pooled NN M1.5 (headline comparison)
# - M2 vs Stratified linear M1.5 (vs the linear floor)
# - M2 vs Pooled linear M1.5 (vs the simplest competitive baseline)
# - Stratified linear vs Pooled linear (the diagnostic finding)
pairs = [
    ('M2 (mean)', pred_m2_mean, 'Pooled NN M1.5', pred_m15_nn),
    ('M2 (mean)', pred_m2_mean, 'Stratified linear M1.5', pred_lin_stratified),
    ('M2 (mean)', pred_m2_mean, 'Pooled linear M1.5', pred_lin_pooled),
    ('Stratified linear M1.5', pred_lin_stratified, 'Pooled linear M1.5', pred_lin_pooled),
    ('Stratified linear M1.5', pred_lin_stratified, 'Pooled NN M1.5', pred_m15_nn),
    ('M2 (mean)', pred_m2_mean, 'Stratified NN M1.5', pred_strat_nn),
]
print(f"  {'A':<25s} {'B':<25s}  {'mean(d)':>10}  {'DM stat':>10}  {'p (2-sided)':>12}  {'verdict':>20}")
print(f"  {'-'*25} {'-'*25}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*20}")
for a_name, a_pred, b_name, b_pred in pairs:
    dm, p, d_bar = diebold_mariano(a_pred, b_pred, dY_te)
    verdict = ("A wins" if dm > 0 and p < 0.05 else
               ("B wins" if dm < 0 and p < 0.05 else "tie"))
    print(f"  {a_name:<25s} {b_name:<25s}  {d_bar:>+10.6f}  {dm:>+10.3f}  "
          f"{p:>12.4f}  {verdict:>20s}")

# =============================================================================
# METRIC 5: REGIME-CONDITIONAL MSE
# =============================================================================
print("\n" + "=" * 78)
print("METRIC 5: REGIME-CONDITIONAL MSE")
print("=" * 78)
print(f"Test set regime occupancy:")
for r in range(3):
    n_r = (r_te_hard == r).sum()
    print(f"  Regime {r}: N={n_r} ({n_r/len(dY_te)*100:.1f}%)")

print(f"\n  {'Model':<30s}  {'Reg 0':>10}  {'Reg 1':>10}  {'Reg 2':>10}  {'Overall':>10}")
print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
for name, pred in models.items():
    err = (pred - dY_te) ** 2
    line = f"  {name:<30s}  "
    for r in range(3):
        mask = r_te_hard == r
        if mask.sum() == 0:
            line += f"{'N/A':>10s}  "
        else:
            line += f"{err[mask].mean():>10.5f}  "
    line += f"{err.mean():>10.5f}"
    print(line)

# Per-regime improvement of M2 over pooled NN M1.5
print(f"\n  M2 improvement over Pooled NN M1.5, by regime:")
for r in range(3):
    mask = r_te_hard == r
    if mask.sum() == 0: continue
    err_m2 = ((pred_m2_mean[mask] - dY_te[mask]) ** 2).mean()
    err_pool = ((pred_m15_nn[mask] - dY_te[mask]) ** 2).mean()
    rel = (err_pool - err_m2) / err_pool * 100
    # DM test within regime
    dm_r, p_r, _ = diebold_mariano(pred_m2_mean[mask], pred_m15_nn[mask], dY_te[mask])
    print(f"    Regime {r} (N={mask.sum()}): "
          f"M2={err_m2:.5f}, Pool={err_pool:.5f}  ({rel:+.2f}%)  "
          f"DM={dm_r:+.2f}, p={p_r:.4f}")

# =============================================================================
# METRIC 6: TAIL PERFORMANCE (substantively important large changes)
# =============================================================================
print("\n" + "=" * 78)
print("METRIC 6: TAIL PERFORMANCE (|ΔY_actual| > threshold)")
print("=" * 78)
for thresh in [0.05, 0.10, 0.20]:
    mask = np.abs(dY_te) > thresh
    n = mask.sum()
    if n < 10:
        print(f"\n  |ΔY| > {thresh}: only N={n}, skipping")
        continue
    print(f"\n  |ΔY_actual| > {thresh}: N={n} ({n/len(dY_te)*100:.1f}% of test)")
    print(f"    {'Model':<30s}  {'MSE':>10}  {'Skill vs RW':>12}")
    mse_rw_tail = float((dY_te[mask] ** 2).mean())
    for name, pred in models.items():
        mse = float(((pred[mask] - dY_te[mask]) ** 2).mean())
        skill = 1.0 - mse / mse_rw_tail
        print(f"    {name:<30s}  {mse:>10.5f}  {skill:>+12.5f}")
    # DM test on tail subset for headline comparison
    dm_t, p_t, _ = diebold_mariano(pred_m2_mean[mask], pred_m15_nn[mask], dY_te[mask])
    print(f"    M2 vs Pooled NN M1.5 DM stat: {dm_t:+.2f}, p={p_t:.4f}")

# =============================================================================
# Save predictions for later analysis
# =============================================================================
print("\n" + "=" * 78)
print("SAVING PREDICTIONS")
print("=" * 78)
with open(SAVE_PATH, 'wb') as f:
    pickle.dump({
        'predictions': {
            'rw': pred_rw,
            'linear_Y_tm1_only': fit_predict_lin(Y_lag_tr.reshape(-1,1), dY_tr,
                                                   Y_lag_te.reshape(-1,1)),
            'linear_pooled_M1.5': pred_lin_pooled,
            'linear_stratified_M1.5': pred_lin_stratified,
            'NN_pooled_M1': pred_m1_nn,
            'NN_pooled_M1.5': pred_m15_nn,
            'NN_stratified_M1.5': pred_strat_nn,
            'M2_per_seed': m2_preds_per_seed,
            'M2_mean': pred_m2_mean,
        },
        'targets': dY_te,
        'regime_assignment': r_te_hard,
        'y_lag': Y_lag_te,
        'country_id': c_te,
        'year': yr_te,
    }, f)
print(f"  Saved to {SAVE_PATH}")

print("\nDone.")
