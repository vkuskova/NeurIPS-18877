# =============================================================================
# V-Dem v1h: regime-aware PANAR-X on V-Dem clean elections (ΔY target)
#
# Architectural changes from previous v1h:
#   1. Sharper gating: tau_init = 0.5 (was 3.0); lam_phi = 0.005 (was 0.02).
#      The diagnostic showed regime-conditional signal recoverable via HARD
#      tertile stratification. Soft gating with broad tau averages this away.
#   2. New 'hard_routing' enforcement mode: route each obs to argmin-distance
#      regime. Direct M2 analog of stratified linear M1.5.
#   3. Reports stratified linear M1.5 floor alongside pooled M1.5 baseline.
#      Stratified NN baselines trained for 3x epochs (~900) to roughly
#      compensate for smaller per-regime sample sizes; validation-based early
#      stopping is deferred.
#
# IMPORTANT: hard_routing breaks the soft-gating Lipschitz audit (the gating
# function is discontinuous, so empirical_L_phi is grid-dependent and rho is
# undefined). Hard-routing runs are predictive baselines ONLY:
#   - audited only on per-regime L_Y and L_X (still well-defined)
#   - L_phi, L_phi_eq, B_trans, alpha_star, rho all set to NaN
#   - excluded from the theorem-audited Pareto winner check
#   - run with lam_M=0 only (lam_M penalty is mechanically incoherent under
#     hard routing because unused regime outputs get pulled toward the active
#     regime regardless of routing)
#
# Sweep grid:
#   hard_routing  x lam_M=0   x 5 seeds = 5 runs (predictive baseline)
#   soft_only     x 3 lam_M  x 5 seeds = 15 runs (audited)
#   projection_every_epoch  x 3 lam_M x 5 seeds = 15 runs (audited)
#   projection_every_batch  x 3 lam_M x 5 seeds = 15 runs (audited)
#   Total = 50 runs.
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
    # Running from the reproducibility package root
    DATA_PATH = str(_LOCAL_DATA)
    _LOCAL_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUT_PATH = _LOCAL_ARTIFACTS / 'vdem_v1h_frontier.pkl'
elif _COLAB_PROJECT_ROOT.exists() or os.path.exists('/content/drive'):
    # Running on Colab with the project Drive mounted
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception:
        pass
    PROJECT_ROOT = _COLAB_PROJECT_ROOT
    VDEM_DIR = PROJECT_ROOT / 'results' / 'regime_panar' / 'vdem_v1h'
    VDEM_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH = str(PROJECT_ROOT / 'data' / 'merged_completevars_balanced_zero_missing.csv')
    OUT_PATH = VDEM_DIR / 'vdem_v1h_frontier.pkl'
else:
    raise FileNotFoundError(
        f"Could not find data: tried {_LOCAL_DATA} (relative) and "
        f"/content/drive (Colab). Run from the reproducibility package root, "
        f"or set DATA_PATH and OUT_PATH manually before running this script."
    )

print(f"DATA_PATH: {DATA_PATH}")
print(f"OUT_PATH:  {OUT_PATH}")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
Y_VAR = 'v2xel_frefair'
X_VARS = ['v2csantimv', 'v2jureform', 'v2psorgs', 'v2mecorrpt']
L_LAG = 5
TEST_YEARS_PER_COUNTRY = 10
N_X = len(X_VARS)
PREDICT_CHANGE = True

# Sharper gating defaults
TAU_INIT = 0.5
LAM_PHI_DEFAULT = 0.005

# Stratified NN baseline training: 3x epochs to compensate for smaller samples
BASELINE_EPOCHS_POOLED = 300
BASELINE_EPOCHS_STRATIFIED = 900   # 3x

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
print("Loading V-Dem dataset (continuous, complete, balanced)...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Y: {Y_VAR}  (n_unique = {df[Y_VAR].nunique()})")
print(f"  X: {X_VARS}")
df = df.sort_values(['country_id', 'year']).reset_index(drop=True)

# -----------------------------------------------------------------------------
# Build lag-window tensors with per-country split
# -----------------------------------------------------------------------------
def build_panel_tensors(df, y_var, x_vars, L, test_years_per_country):
    train_data, test_data = [], []
    skipped = []
    for cid in df['country_id'].unique():
        sub = df[df['country_id'] == cid].sort_values('year').reset_index(drop=True)
        if len(sub) < L + test_years_per_country + 1:
            skipped.append((cid, sub['country_name'].iloc[0], len(sub)))
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
            y_target = y_window[-1] - y_lag if PREDICT_CHANGE else y_window[-1]
            row = {
                'y_hist': y_window[:-1],
                'x_hist': x_window,
                'y_target': y_target,
                'y_lag': y_lag,
                'country_id': cid,
                'country_name': country_name,
                'year': years[i],
            }
            if years[i] < country_test_threshold:
                train_data.append(row)
            else:
                test_data.append(row)
    return train_data, test_data, skipped

train_data, test_data, skipped = build_panel_tensors(
    df, Y_VAR, X_VARS, L_LAG, TEST_YEARS_PER_COUNTRY)
print(f"\nTrain: {len(train_data)}, Test: {len(test_data)}, Skipped: {len(skipped)}")

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

print(f"\nΔY target distribution:")
print(f"  Train: mean={dY_tr.mean():+.5f}, std={dY_tr.std():.5f}")
print(f"  Test:  mean={dY_te.mean():+.5f}, std={dY_te.std():.5f}")

# Standardize X (training-set z-scores)
X_flat_tr = X_h_tr.reshape(-1, N_X)
x_mean = X_flat_tr.mean(axis=0)
x_std = X_flat_tr.std(axis=0)
x_std = np.where(x_std < 1e-8, 1.0, x_std)
X_h_tr = (X_h_tr - x_mean) / x_std
X_h_te = (X_h_te - x_mean) / x_std

# Country index, regime centers
unique_c = np.unique(np.concatenate([c_tr, c_te]))
cid_to_idx = {c: i for i, c in enumerate(unique_c)}
u_tr = np.array([cid_to_idx[c] for c in c_tr])
u_te = np.array([cid_to_idx[c] for c in c_te])
U = len(unique_c)
print(f"\nUnique countries: {U}")

psi_train = Y_lag_tr
regime_centers = (
    float(np.percentile(psi_train, 100/6)),
    float(np.percentile(psi_train, 50)),
    float(np.percentile(psi_train, 100*5/6)),
)
regime_boundaries = (
    float(np.percentile(psi_train, 100/3)),
    float(np.percentile(psi_train, 100*2/3)),
)
print(f"Regime centers (Y_{{t-1}} tertile midpoints): "
      f"{tuple(round(x, 3) for x in regime_centers)}")
print(f"Regime boundaries: {tuple(round(x, 3) for x in regime_boundaries)}")

# -----------------------------------------------------------------------------
# Baselines: linear (pooled, stratified) and NN (pooled, stratified)
# -----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BASELINES (target: ΔY)")
print("=" * 78)

rw_mse = float((dY_te ** 2).mean())
print(f"RW (predict 0 change):              test MSE = {rw_mse:.5f}")

# Linear-in-Y_{t-1}
A = np.column_stack([np.ones_like(Y_lag_tr), Y_lag_tr])
beta_lin, _, _, _ = np.linalg.lstsq(A, dY_tr, rcond=None)
lin_mse = float(((dY_te - (beta_lin[0] + beta_lin[1] * Y_lag_te)) ** 2).mean())
print(f"Linear in Y_{{t-1}}:                    test MSE = {lin_mse:.5f}")

# Pooled linear M1 and M1.5
def fit_predict_lin(X_train, y_train, X_test):
    A_tr = np.column_stack([np.ones(len(X_train)), X_train])
    beta, *_ = np.linalg.lstsq(A_tr, y_train, rcond=None)
    A_te = np.column_stack([np.ones(len(X_test)), X_test])
    return A_te @ beta, beta

Y_lags_tr_2d = Y_h_tr[:, :, 0]
Y_lags_te_2d = Y_h_te[:, :, 0]
X_lags_tr_2d = X_h_tr.reshape(len(X_h_tr), -1)
X_lags_te_2d = X_h_te.reshape(len(X_h_te), -1)

pred_lin_m1, _ = fit_predict_lin(Y_lags_tr_2d, dY_tr, Y_lags_te_2d)
linm1_mse = float(((dY_te - pred_lin_m1) ** 2).mean())
print(f"Pooled linear M1 (Y lags):          test MSE = {linm1_mse:.5f}")

YX_tr = np.hstack([Y_lags_tr_2d, X_lags_tr_2d])
YX_te = np.hstack([Y_lags_te_2d, X_lags_te_2d])
pred_lin_m15, _ = fit_predict_lin(YX_tr, dY_tr, YX_te)
linm15_mse = float(((dY_te - pred_lin_m15) ** 2).mean())
print(f"Pooled linear M1.5 (Y+X lags):      test MSE = {linm15_mse:.5f}")

# Stratified linear M1.5 (matches diagnostic)
def assign_regime_hard(y_lag1, b1, b2):
    r = np.zeros(len(y_lag1), dtype=int)
    r[y_lag1 < b1] = 0
    r[(y_lag1 >= b1) & (y_lag1 < b2)] = 1
    r[y_lag1 >= b2] = 2
    return r

r_tr_hard = assign_regime_hard(Y_lag_tr, regime_boundaries[0], regime_boundaries[1])
r_te_hard = assign_regime_hard(Y_lag_te, regime_boundaries[0], regime_boundaries[1])

stratified_pred_te = np.full(len(dY_te), np.nan)
for r in range(3):
    tr_mask = r_tr_hard == r
    te_mask = r_te_hard == r
    if tr_mask.sum() < 50 or te_mask.sum() < 5:
        stratified_pred_te[te_mask] = pred_lin_m15[te_mask]
        continue
    pred_r, _ = fit_predict_lin(YX_tr[tr_mask], dY_tr[tr_mask], YX_te[te_mask])
    stratified_pred_te[te_mask] = pred_r
strat_lin_m15_mse = float(((dY_te - stratified_pred_te) ** 2).mean())
print(f"Stratified linear M1.5 (the floor): test MSE = {strat_lin_m15_mse:.5f}")
print(f"  Stratified vs pooled linear M1.5: "
      f"{(linm15_mse - strat_lin_m15_mse)/linm15_mse*100:+.2f}%")
print(f"  Stratified vs RW:                  "
      f"{(rw_mse - strat_lin_m15_mse)/rw_mse*100:+.2f}%")

# Pooled NN M1 and M1.5
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

def train_simple(model, Y_h_tr, X_h_tr, Y_t_tr, epochs=300, lr=3e-4,
                  weight_decay=1e-4, batch=128, use_x=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Yh = torch.tensor(Y_h_tr, dtype=torch.float32)
    Xh = torch.tensor(X_h_tr, dtype=torch.float32) if use_x else None
    Yt = torch.tensor(Y_t_tr, dtype=torch.float32)
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

torch.manual_seed(0); np.random.seed(0)
m1 = train_simple(M1_YNet(L_LAG), Y_h_tr, X_h_tr, dY_tr,
                   epochs=BASELINE_EPOCHS_POOLED, use_x=False)
with torch.no_grad():
    m1_pred = m1(torch.tensor(Y_h_te, dtype=torch.float32)).numpy()
m1_mse = float(((m1_pred - dY_te) ** 2).mean())
print(f"\nM1 NN (Y-only, pooled):             test MSE = {m1_mse:.5f}")

torch.manual_seed(0); np.random.seed(0)
m15 = train_simple(M15_PANAR(L_LAG, N_X), Y_h_tr, X_h_tr, dY_tr,
                    epochs=BASELINE_EPOCHS_POOLED, use_x=True)
with torch.no_grad():
    m15_pred = m15(torch.tensor(Y_h_te, dtype=torch.float32),
                    torch.tensor(X_h_te, dtype=torch.float32)).numpy()
m15_mse = float(((m15_pred - dY_te) ** 2).mean())
print(f"M1.5 NN (pooled, Y+X):              test MSE = {m15_mse:.5f}")

# Stratified NN M1.5: 3x epochs per regime (note: validation-based early
# stopping deferred)
print(f"\nStratified NN M1.5 (3 separate networks, {BASELINE_EPOCHS_STRATIFIED} epochs each):")
print(f"  Note: 3x epochs to roughly compensate for smaller per-regime samples;")
print(f"  validation-based early stopping deferred.")
strat_nn_pred_te = np.full(len(dY_te), np.nan)
for r in range(3):
    tr_mask = r_tr_hard == r
    te_mask = r_te_hard == r
    if tr_mask.sum() < 50 or te_mask.sum() < 5:
        strat_nn_pred_te[te_mask] = m15_pred[te_mask]
        continue
    torch.manual_seed(0); np.random.seed(0)
    m15_r = train_simple(M15_PANAR(L_LAG, N_X),
                          Y_h_tr[tr_mask], X_h_tr[tr_mask], dY_tr[tr_mask],
                          epochs=BASELINE_EPOCHS_STRATIFIED, use_x=True)
    with torch.no_grad():
        pred_r = m15_r(torch.tensor(Y_h_te[te_mask], dtype=torch.float32),
                        torch.tensor(X_h_te[te_mask], dtype=torch.float32)).numpy()
    strat_nn_pred_te[te_mask] = pred_r
strat_nn_m15_mse = float(((strat_nn_pred_te - dY_te) ** 2).mean())
print(f"  test MSE = {strat_nn_m15_mse:.5f}")
print(f"  Stratified NN vs pooled NN M1.5:  "
      f"{(m15_mse - strat_nn_m15_mse)/m15_mse*100:+.2f}%")

print(f"\n{'='*78}")
print(f"BASELINE FLOOR SUMMARY:")
print(f"  RW (predict 0):                      {rw_mse:.5f}")
print(f"  Pooled NN M1.5:                      {m15_mse:.5f}  "
      f"(comparison baseline for M2)")
print(f"  Stratified linear M1.5 (linear floor): {strat_lin_m15_mse:.5f}")
print(f"  Stratified NN M1.5 (NN regime floor): {strat_nn_m15_mse:.5f}")
print(f"{'='*78}")

# -----------------------------------------------------------------------------
# M2 architecture
# -----------------------------------------------------------------------------
class PerEdgeNetM2(nn.Module):
    """Two-layer MLP per regime per channel. Distinct from baseline M15_PANAR's
    sequential definition to allow direct access to .lin1 and .lin2 weights for
    spectral norm projection."""
    def __init__(self, L, hidden=32):
        super().__init__()
        self.lin1 = nn.Linear(L, hidden)
        self.lin2 = nn.Linear(hidden, 1)
    def forward(self, x):
        return self.lin2(F.relu(self.lin1(x))).squeeze(-1)

class M2_RegimeAware(nn.Module):
    def __init__(self, L, n_x, hidden=32, n_regimes=3, tau_init=TAU_INIT,
                 frozen_centers=(0.2, 0.5, 0.8), routing='soft'):
        super().__init__()
        self.n_regimes = n_regimes
        self.n_x = n_x
        self.routing = routing
        self.f_yy = nn.ModuleList([PerEdgeNetM2(L, hidden)
                                    for _ in range(n_regimes)])
        self.f_xy = nn.ModuleList([
            nn.ModuleList([PerEdgeNetM2(L, hidden) for _ in range(n_x)])
            for _ in range(n_regimes)
        ])
        self.bias = nn.Parameter(torch.zeros(n_regimes))
        self.register_buffer('centers', torch.tensor(frozen_centers,
                                                       dtype=torch.float32))
        self.log_tau = nn.Parameter(torch.tensor(np.log(tau_init),
                                                   dtype=torch.float32))
    
    def gating_soft(self, psi):
        tau = torch.exp(self.log_tau)
        d2 = (psi.unsqueeze(-1) - self.centers.unsqueeze(0)) ** 2
        return F.softmax(-d2 / tau, dim=-1)
    
    def gating_hard(self, psi):
        # argmin over distance: pick the regime whose center is closest
        d2 = (psi.unsqueeze(-1) - self.centers.unsqueeze(0)) ** 2
        idx = d2.argmin(dim=-1)
        return F.one_hot(idx, num_classes=self.n_regimes).float()
    
    def gating(self, psi):
        if self.routing == 'hard':
            return self.gating_hard(psi)
        return self.gating_soft(psi)
    
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

# -----------------------------------------------------------------------------
# Hard projection (Y-channel only) and soft Lipschitz penalty
# -----------------------------------------------------------------------------
def project_layer(layer, target_norm):
    with torch.no_grad():
        sigma = torch.linalg.svdvals(layer.weight)[0].item()
        if sigma > target_norm:
            layer.weight.mul_(target_norm / sigma)

def project_per_edge_net(net, target_lipschitz):
    per_layer = float(np.sqrt(target_lipschitz))
    project_layer(net.lin1, per_layer)
    project_layer(net.lin2, per_layer)

def project_m2_y_only(model, target_LY):
    for r in range(model.n_regimes):
        project_per_edge_net(model.f_yy[r], target_LY)

def per_regime_predictor_fn(model, r):
    def fn(Y_h, X_h):
        o = model.bias[r] + model.f_yy[r](Y_h[:, :, 0])
        for j in range(model.n_x):
            o = o + model.f_xy[r][j](X_h[:, :, j])
        return o
    return fn

def jacobian_softmax_norm_Y(model_fn, Y_hist_batch, X_hist_batch, tau_softmax=5.0):
    Y_h_in = Y_hist_batch.clone().requires_grad_(True)
    out = model_fn(Y_h_in, X_hist_batch)
    grad = torch.autograd.grad(out.sum(), Y_h_in, create_graph=True)[0]
    norms = grad.flatten(1).norm(dim=1)
    B = norms.shape[0]
    return (torch.logsumexp(tau_softmax * norms, dim=0) - np.log(B)) / tau_softmax

# -----------------------------------------------------------------------------
# Training modes
# -----------------------------------------------------------------------------
def train_m2_unified(model, Y_h_tr, X_h_tr, target, mode,
                      epochs=300, lr=3e-4, lam_M=0.10, lam_phi=LAM_PHI_DEFAULT,
                      weight_decay=1e-4, batch=128):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Yh_t = torch.tensor(Y_h_tr, dtype=torch.float32)
    Xh_t = torch.tensor(X_h_tr, dtype=torch.float32)
    tgt = torch.tensor(target, dtype=torch.float32)
    N = len(Yh_t)
    
    if mode == 'hard_routing':
        # Pure predictive baseline; no Lipschitz constraint, no lam_phi
        # (gating is hard, so log_tau is unused). lam_M is forced to 0 by
        # the sweep grid; if a caller passes lam_M > 0, train_m2_unified
        # will still apply it but the result is mechanically incoherent.
        lam_Y = 0.0
        do_pb, do_pe = False, False
        model.routing = 'hard'
        lam_phi_eff = 0.0
    elif mode == 'soft_only':
        lam_Y = 0.20; target_LY_soft = 0.55
        do_pb, do_pe = False, False
        model.routing = 'soft'
        lam_phi_eff = lam_phi
    elif mode == 'projection_every_epoch':
        lam_Y = 0.05; target_LY_soft = 0.85; target_LY_hard = 0.90
        do_pb, do_pe = False, True
        model.routing = 'soft'
        lam_phi_eff = lam_phi
        project_m2_y_only(model, target_LY_hard)
    elif mode == 'projection_every_batch':
        lam_Y = 0.0; target_LY_hard = 0.85
        do_pb, do_pe = True, False
        model.routing = 'soft'
        lam_phi_eff = lam_phi
        project_m2_y_only(model, target_LY_hard)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    for ep in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, N, batch):
            idx = perm[s:s+batch]
            opt.zero_grad()
            pred = model(Yh_t[idx], Xh_t[idx])
            loss = F.mse_loss(pred, tgt[idx])
            
            if lam_Y > 0:
                spec_pen_Y = torch.zeros((), device=Yh_t.device)
                for r in range(model.n_regimes):
                    fn_r = per_regime_predictor_fn(model, r)
                    sigma_Y = jacobian_softmax_norm_Y(
                        fn_r, Yh_t[idx], Xh_t[idx], tau_softmax=5.0)
                    spec_pen_Y = spec_pen_Y + F.relu(sigma_Y - target_LY_soft) ** 2
                loss = loss + lam_Y * spec_pen_Y
            
            if lam_phi_eff > 0:
                inv_tau = torch.exp(-model.log_tau)
                loss = loss + lam_phi_eff * inv_tau ** 2
            
            if lam_M > 0:
                per_r = model.per_regime_outputs(Yh_t[idx], Xh_t[idx])
                disagree = torch.zeros((), device=Yh_t.device)
                for r1 in range(model.n_regimes):
                    for r2 in range(r1+1, model.n_regimes):
                        disagree = disagree + ((per_r[:, r1] - per_r[:, r2]) ** 2).mean()
                loss = loss + lam_M * disagree
            
            loss.backward()
            opt.step()
            if do_pb:
                project_m2_y_only(model, target_LY_hard)
        if do_pe:
            project_m2_y_only(model, target_LY_hard)
    
    model.eval()
    return model

# -----------------------------------------------------------------------------
# Audit: per-regime Lipschitz applies to ALL routings; gating-related quantities
# only apply to soft routing
# -----------------------------------------------------------------------------
def audit_per_regime_Lipschitz(model, Y_h_t, X_h_t, n_samples=200):
    """Per-regime function gradient norms. Well-defined for both hard and soft
    routing because each regime's per-edge networks are continuous."""
    L_Y = np.zeros(model.n_regimes)
    L_X = np.zeros(model.n_regimes)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(Y_h_t), size=min(n_samples, len(Y_h_t)),
                             replace=False)
    for r in range(model.n_regimes):
        max_Y, max_X = 0.0, 0.0
        for i in sample_idx:
            Y_h = Y_h_t[i:i+1].clone().requires_grad_(True)
            X_h = X_h_t[i:i+1].clone().requires_grad_(True)
            pred_r = model.bias[r] + model.f_yy[r](Y_h[:, :, 0])
            for j in range(model.n_x):
                pred_r = pred_r + model.f_xy[r][j](X_h[:, :, j])
            gY = torch.autograd.grad(pred_r.sum(), Y_h, retain_graph=True)[0]
            gX = torch.autograd.grad(pred_r.sum(), X_h)[0]
            ny = float(gY.flatten().norm())
            nx = float(gX.flatten().norm())
            if ny > max_Y: max_Y = ny
            if nx > max_X: max_X = nx
        L_Y[r] = max_Y
        L_X[r] = max_X
    return L_Y, L_X

def empirical_L_phi(model, psi_lo, psi_hi, n_grid=500):
    """ONLY valid for soft routing. Hard routing produces a step function whose
    finite-difference slope is grid-dependent (artifact). Caller must check."""
    psi_grid = torch.tensor(np.linspace(psi_lo, psi_hi, n_grid), dtype=torch.float32)
    with torch.no_grad():
        pi = model.gating(psi_grid).numpy()
    max_slope = 0.0
    for i in range(n_grid - 1):
        d_pi = np.abs(pi[i+1] - pi[i]).sum()
        d_psi = float(psi_grid[i+1] - psi_grid[i])
        if d_psi > 1e-8 and d_pi/d_psi > max_slope:
            max_slope = d_pi/d_psi
    return max_slope

def empirical_L_phi_eq(model, psi_test, c1, c2):
    """ONLY valid for soft routing."""
    L_phi_eq = np.zeros(3)
    masks = [psi_test < c1, (psi_test >= c1) & (psi_test < c2), psi_test >= c2]
    for r, mask in enumerate(masks):
        if mask.sum() < 10:
            L_phi_eq[r] = float('nan'); continue
        psi_r = psi_test[mask]
        psi_lo_r, psi_hi_r = np.percentile(psi_r, [25, 75])
        psi_sweep = torch.tensor(np.linspace(psi_lo_r, psi_hi_r, 50),
                                  dtype=torch.float32)
        with torch.no_grad():
            pi = model.gating(psi_sweep).numpy()
        max_slope = 0.0
        for i in range(len(psi_sweep) - 1):
            d_pi = np.abs(pi[i+1] - pi[i]).sum()
            d_psi = float(psi_sweep[i+1] - psi_sweep[i])
            if d_psi > 1e-8 and d_pi/d_psi > max_slope:
                max_slope = d_pi/d_psi
        L_phi_eq[r] = max_slope
    return L_phi_eq

def get_M(model, Yh_t, Xh_t):
    with torch.no_grad():
        per_r = model.per_regime_outputs(Yh_t, Xh_t).numpy()
    M = 0.0
    for r1 in range(model.n_regimes):
        for r2 in range(r1+1, model.n_regimes):
            d = float(np.abs(per_r[:, r1] - per_r[:, r2]).max())
            if d > M: M = d
    return M

def compute_alpha_emp(model, Y_h_te, u_te, U):
    psi_te = Y_h_te[:, -1, 0]
    psi_te_t = torch.tensor(psi_te, dtype=torch.float32)
    with torch.no_grad():
        pi_te = model.gating(psi_te_t).numpy()
    r_pred = np.argmax(pi_te, axis=1)
    per_unit_alpha = []
    for u in range(U):
        mask = u_te == u
        if mask.sum() < 2: continue
        r_u = r_pred[mask]
        transitions = (r_u[1:] != r_u[:-1]).sum()
        per_unit_alpha.append(transitions / (len(r_u) - 1))
    return float(np.mean(per_unit_alpha)) if per_unit_alpha else 0.0

def compute_rho(L_eq_max, B_trans, alpha_emp):
    if L_eq_max >= 1:
        return None, None
    if B_trans <= 1:
        alpha_star = float('inf')
    else:
        alpha_star = float(-np.log(L_eq_max) / (np.log(B_trans) - np.log(L_eq_max)))
    rho = float(L_eq_max ** (1 - alpha_emp) * B_trans ** alpha_emp)
    return alpha_star, rho

def full_audit(model, Y_h_te, X_h_te, target_te, u_te, U, L, regime_boundaries):
    """Audit dispatcher.
    
    For soft routing: full theorem-audit including L_phi, L_phi_eq, B_trans,
    alpha_star, rho.
    
    For hard routing: only per-regime L_Y, L_X, M, and test_mse are valid.
    L_phi-related quantities are set to NaN; rho/alpha_star are None.
    """
    Yh_te_t = torch.tensor(Y_h_te, dtype=torch.float32)
    Xh_te_t = torch.tensor(X_h_te, dtype=torch.float32)
    psi_te = Y_h_te[:, -1, 0]
    
    L_Y, L_X = audit_per_regime_Lipschitz(model, Yh_te_t, Xh_te_t)
    L_max = float(L_Y.max())
    LX_max = float(L_X.max())
    M_emp = get_M(model, Yh_te_t, Xh_te_t)
    
    with torch.no_grad():
        pred = model(Yh_te_t, Xh_te_t)
        test_mse = float(((pred - torch.tensor(target_te, dtype=torch.float32)) ** 2).mean())
    
    if model.routing == 'hard':
        # Theorem audit not applicable
        result = {
            'routing': 'hard',
            'L_Y_per_regime': L_Y.tolist(), 'L_X_per_regime': L_X.tolist(),
            'L_max': L_max, 'LX_max': LX_max, 'M': M_emp,
            'tau': float('nan'),
            'L_phi': float('nan'),
            'L_phi_eq_max': float('nan'),
            'L_phi_eq_M': float('nan'),
            'L_phi_eq_per_regime': [float('nan')] * 3,
            'L_eq_per_regime': [float('nan')] * 3,
            'L_eq_max': float('nan'),
            'B_trans': float('nan'),
            'alpha_star': None, 'alpha_emp': float('nan'),
            'rho': None,
            'test_mse': test_mse,
        }
        return result
    
    # Soft routing: full audit
    tau_emp = float(torch.exp(model.log_tau).item())
    L_phi = empirical_L_phi(model, float(psi_te.min()), float(psi_te.max()))
    L_phi_eq = empirical_L_phi_eq(model, psi_te,
                                    regime_boundaries[0], regime_boundaries[1])
    
    L_eq_per_regime = L_Y + L_phi_eq * M_emp
    L_eq_max = float(np.nanmax(L_eq_per_regime))
    L_phi_eq_max = float(np.nanmax(L_phi_eq))
    L_phi_eq_M = L_phi_eq_max * M_emp
    B_trans = L_max + L_phi * M_emp
    
    alpha_emp = compute_alpha_emp(model, Y_h_te, u_te, U)
    alpha_star, rho = compute_rho(L_eq_max, B_trans, alpha_emp)
    
    return {
        'routing': 'soft',
        'L_Y_per_regime': L_Y.tolist(), 'L_X_per_regime': L_X.tolist(),
        'L_max': L_max, 'LX_max': LX_max,
        'M': M_emp, 'L_phi_eq_max': L_phi_eq_max, 'L_phi_eq_M': L_phi_eq_M,
        'tau': tau_emp, 'L_phi': L_phi,
        'L_phi_eq_per_regime': L_phi_eq.tolist(),
        'L_eq_per_regime': L_eq_per_regime.tolist(),
        'L_eq_max': L_eq_max, 'B_trans': B_trans,
        'alpha_star': alpha_star, 'alpha_emp': alpha_emp, 'rho': rho,
        'test_mse': test_mse,
    }

# -----------------------------------------------------------------------------
# Sweep grid
# -----------------------------------------------------------------------------
# hard_routing: lam_M=0 only (lam_M penalty is mechanically incoherent under
# hard routing, see header comment). 5 seeds.
# Three audited modes: 3 lam_M x 5 seeds each = 15 runs each.
# Total: 5 + 45 = 50 runs.

SWEEP_CELLS = []
# Hard routing: predictive baseline only
for seed in [0, 1, 2, 3, 4]:
    SWEEP_CELLS.append(('hard_routing', 0.00, seed))
# Soft routing modes: full sweep
for mode in ['soft_only', 'projection_every_epoch', 'projection_every_batch']:
    for lam_M in [0.00, 0.05, 0.10]:
        for seed in [0, 1, 2, 3, 4]:
            SWEEP_CELLS.append((mode, lam_M, seed))

MODE_SHORT = {'hard_routing': 'hard', 'soft_only': 'soft',
              'projection_every_epoch': 'proj_ep',
              'projection_every_batch': 'proj_ba'}

results = {}
total_runs = len(SWEEP_CELLS)
t_sweep_start = time.time()

print("\n" + "=" * 78)
print(f"V-DEM v1h FRONTIER SWEEP: {total_runs} runs (target: ΔY)")
print("=" * 78)
print(f"  Hard routing: lam_M=0 x 5 seeds = 5 runs (predictive baseline only)")
print(f"  Soft modes: 3 modes x 3 lam_M x 5 seeds = 45 runs (audited)")
print(f"  tau_init: {TAU_INIT}, lam_phi_default: {LAM_PHI_DEFAULT}")
print(f"  Pooled NN M1.5 baseline:    {m15_mse:.5f}")
print(f"  Stratified linear M1.5:     {strat_lin_m15_mse:.5f}")
print(f"  Stratified NN M1.5:         {strat_nn_m15_mse:.5f}")
print()

for run_idx, (mode, lam_M, seed) in enumerate(SWEEP_CELLS, 1):
    cell_key = (mode, lam_M)
    if cell_key not in results:
        results[cell_key] = []
    
    t_run = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    m2 = M2_RegimeAware(L_LAG, N_X, n_regimes=3, tau_init=TAU_INIT,
                         frozen_centers=regime_centers,
                         routing='hard' if mode == 'hard_routing' else 'soft')
    m2 = train_m2_unified(m2, Y_h_tr, X_h_tr, dY_tr, mode=mode,
                           epochs=300, lam_M=lam_M, lam_phi=LAM_PHI_DEFAULT)
    audit = full_audit(m2, Y_h_te, X_h_te, dY_te, u_te, U, L_LAG,
                        regime_boundaries)
    audit['seed'] = seed
    audit['mode'] = mode
    audit['lam_M'] = lam_M
    results[cell_key].append(audit)
    
    elapsed = time.time() - t_run
    sweep_elapsed = time.time() - t_sweep_start
    eta = sweep_elapsed * (total_runs - run_idx) / run_idx
    
    if audit['routing'] == 'hard':
        rho_str = "N/A (hard)"
        L_eq_str = "  N/A"
    else:
        rho_str = f"{audit['rho']:.3f}" if audit['rho'] is not None else "  N/A"
        L_eq_str = f"{audit['L_eq_max']:.3f}"
    
    gap_pooled = m15_mse - audit['test_mse']
    gap_strat = strat_nn_m15_mse - audit['test_mse']
    print(f"[{run_idx}/{total_runs}] mode={MODE_SHORT[mode]}, lam_M={lam_M}, seed={seed}: "
          f"L_eq={L_eq_str}, MSE={audit['test_mse']:.5f}, "
          f"vs_pool={gap_pooled:+.5f}, vs_strat={gap_strat:+.5f}, "
          f"M={audit['M']:.3f}, rho={rho_str}  "
          f"({elapsed:.0f}s, ETA {eta/60:.0f}min)")
    
    # Save after every run
    with open(OUT_PATH, 'wb') as f:
        pickle.dump({
            'results': results,
            'baselines': {
                'RW': rw_mse, 'linear_Y_t-1': lin_mse,
                'pooled_linear_M1': linm1_mse,
                'pooled_linear_M1.5': linm15_mse,
                'stratified_linear_M1.5': strat_lin_m15_mse,
                'M1_NN': m1_mse,
                'pooled_NN_M1.5': m15_mse,
                'stratified_NN_M1.5': strat_nn_m15_mse,
            },
            'config': {
                'Y_var': Y_VAR, 'X_vars': X_VARS,
                'predict_change': PREDICT_CHANGE,
                'L': L_LAG,
                'test_years_per_country': TEST_YEARS_PER_COUNTRY,
                'tau_init': TAU_INIT,
                'lam_phi_default': LAM_PHI_DEFAULT,
                'baseline_epochs_pooled': BASELINE_EPOCHS_POOLED,
                'baseline_epochs_stratified': BASELINE_EPOCHS_STRATIFIED,
                'regime_centers': regime_centers,
                'regime_boundaries': regime_boundaries,
                'n_train': len(Y_h_tr), 'n_test': len(Y_h_te),
                'n_countries': U,
            },
        }, f)

print(f"\nSweep complete. Total time: {(time.time() - t_sweep_start)/60:.1f} minutes")

# -----------------------------------------------------------------------------
# Summary tables (separate hard-routing from audited modes)
# -----------------------------------------------------------------------------
def t_ci(values, conf=0.95):
    n = len(values)
    if n < 2: return float('nan'), float('nan')
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(n)
    t_crit = sp_stats.t.ppf((1 + conf) / 2, df=n-1)
    return mean - t_crit * se, mean + t_crit * se

def bootstrap_ci(values, conf=0.95, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boot_means = np.array([rng.choice(values, size=len(values), replace=True).mean()
                            for _ in range(n_boot)])
    alpha = (1 - conf) / 2
    return float(np.percentile(boot_means, 100*alpha)), \
           float(np.percentile(boot_means, 100*(1-alpha)))

print("\n" + "=" * 130)
print("HARD ROUTING ROW (predictive baseline; NOT theorem-audited)")
print("=" * 130)
print(f"  Pooled NN M1.5 = {m15_mse:.5f} | Stratified NN M1.5 = {strat_nn_m15_mse:.5f}")
print()
hard_cell = results.get(('hard_routing', 0.00), [])
if hard_cell:
    mse = [r['test_mse'] for r in hard_cell]
    M_vals = [r['M'] for r in hard_cell]
    gap_pool = [m15_mse - m for m in mse]
    gap_strat = [strat_nn_m15_mse - m for m in mse]
    print(f"  hard_routing, lam_M=0.00:")
    print(f"    test MSE:           {np.mean(mse):.5f} ± {np.std(mse):.5f}")
    print(f"    M:                  {np.mean(M_vals):.3f} ± {np.std(M_vals):.3f}")
    print(f"    vs pooled M1.5:     {np.mean(gap_pool):+.5f}  CI={t_ci(gap_pool)}")
    print(f"    vs stratified NN:   {np.mean(gap_strat):+.5f}  CI={t_ci(gap_strat)}")

print("\n" + "=" * 140)
print("SOFT-ROUTING AUDITED CELLS")
print(f"  Pooled NN M1.5 = {m15_mse:.5f} | Stratified NN M1.5 = {strat_nn_m15_mse:.5f}")
print("=" * 140)
print(f"{'mode':>22} {'lam_M':>6}  "
      f"{'L_eq,max':>14}  {'M':>14}  "
      f"{'M2 vs pool (t-CI)':>26}  {'M2 vs strat (t-CI)':>26}  {'L_eq<1':>7}")
print("-" * 140)

AUDITED_MODES = ['soft_only', 'projection_every_epoch', 'projection_every_batch']
for mode in AUDITED_MODES:
    for lam_M in [0.00, 0.05, 0.10]:
        cell = results.get((mode, lam_M), [])
        if not cell:
            continue
        L_eq = [r['L_eq_max'] for r in cell]
        M_vals = [r['M'] for r in cell]
        mse = [r['test_mse'] for r in cell]
        gap_pool = [m15_mse - m for m in mse]
        gap_strat = [strat_nn_m15_mse - m for m in mse]
        gap_pool_ci = t_ci(gap_pool)
        gap_strat_ci = t_ci(gap_strat)
        bound_holds = sum(1 for r in cell if r['L_eq_max'] < 1)
        gp_str = f"{np.mean(gap_pool):+.5f} [{gap_pool_ci[0]:+.5f},{gap_pool_ci[1]:+.5f}]"
        gs_str = f"{np.mean(gap_strat):+.5f} [{gap_strat_ci[0]:+.5f},{gap_strat_ci[1]:+.5f}]"
        print(f"{mode:>22} {lam_M:>6.2f}  "
              f"{np.mean(L_eq):>5.3f}±{np.std(L_eq):>5.3f}  "
              f"{np.mean(M_vals):>5.3f}±{np.std(M_vals):>5.3f}  "
              f"{gp_str:>26}  {gs_str:>26}  "
              f"{bound_holds:>3}/5")

# Pareto analysis ONLY over audited (soft-routing) cells
print("\n" + "=" * 140)
print("PARETO ANALYSIS (audited cells only; hard_routing excluded as not theorem-applicable)")
print("Criteria: L_eq,max < 1 in 5/5 seeds AND CI on M2-pooled excludes 0 above")
print("=" * 140)
robust_winners = []
for (mode, lam_M), cell in results.items():
    if mode == 'hard_routing':
        continue
    L_eq = [r['L_eq_max'] for r in cell]
    mse = [r['test_mse'] for r in cell]
    gap = [m15_mse - m for m in mse]
    bound_holds_all = all(le < 1 for le in L_eq)
    gap_ci = t_ci(gap)
    if bound_holds_all and gap_ci[0] > 0:
        robust_winners.append((mode, lam_M, np.mean(gap), gap_ci, np.mean(L_eq)))

if robust_winners:
    print("\nRobust winners (CI-significant beat over pooled M1.5 with stability):")
    for mode, lam_M, mean_gap, ci, mean_L in robust_winners:
        print(f"  {mode:>22}, lam_M={lam_M}: gap={mean_gap:+.5f} "
              f"CI=[{ci[0]:+.5f}, {ci[1]:+.5f}], L_eq,max={mean_L:.3f}")
else:
    print("\nNo audited cell has both 5/5 stability AND CI-significant beat over pooled M1.5.")
    print()
    print("Audited cells with 5/5 stability (L_eq < 1):")
    found_any = False
    for (mode, lam_M), cell in results.items():
        if mode == 'hard_routing':
            continue
        L_eq = [r['L_eq_max'] for r in cell]
        if all(le < 1 for le in L_eq):
            found_any = True
            mse = [r['test_mse'] for r in cell]
            gap = [m15_mse - m for m in mse]
            ci = t_ci(gap)
            verdict = ("wins" if ci[0] > 0 else
                       ("loses" if ci[1] < 0 else "ties"))
            print(f"  {mode:>22}, lam_M={lam_M}: gap={np.mean(gap):+.5f} "
                  f"CI=[{ci[0]:+.5f}, {ci[1]:+.5f}] -> {verdict} pooled M1.5")
    if not found_any:
        print("  (none)")

# Best unconstrained predictive cell across all modes
print("\nBest unconstrained M2 cell (lam_M=0):")
unconstr_keys = [('hard_routing', 0.00), ('soft_only', 0.00)]
for key in unconstr_keys:
    if key in results:
        cell = results[key]
        mse = [r['test_mse'] for r in cell]
        gap_pool = np.mean([m15_mse - m for m in mse])
        gap_strat = np.mean([strat_nn_m15_mse - m for m in mse])
        L_eq_arr = [r['L_eq_max'] for r in cell if not np.isnan(r['L_eq_max'])]
        L_eq_str = f"{np.mean(L_eq_arr):.3f}" if L_eq_arr else "N/A"
        print(f"  {key[0]}: mean MSE = {np.mean(mse):.5f}")
        print(f"    vs pooled NN M1.5:     {gap_pool:+.5f}  ({gap_pool/m15_mse*100:+.2f}%)")
        print(f"    vs stratified NN M1.5: {gap_strat:+.5f}  ({gap_strat/strat_nn_m15_mse*100:+.2f}%)")
        print(f"    L_eq,max mean: {L_eq_str}")

print("\nDone.")
