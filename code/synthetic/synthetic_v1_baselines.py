"""
synthetic_v1_baselines.py

Port of the original `synthetic_v1` notebook cell from the development run.
Produces `artifacts/synthetic_v1_results.pkl`, which contains:
  - M1, M1.5, M2 trained baselines and their test MSE
  - RW and AR(1) reference MSEs
  - Theorem-constant audit (L_max, M, L_eq_max, B_trans, alpha_star, rho)
  - Regime alignment metrics

The Pareto-frontier sweep (`synthetic_v1f_sweep.py`) loads this pickle to
reuse the M1/M1.5 baselines as reference; running this script before that
one is therefore required.

This is a faithful port of the original Colab cell (single seed = 0,
single set of hyperparameters). Multi-seed M2 sweeps are in v1f.

Path-portable: assumes you run from the package root, writing to
artifacts/. No Colab Drive mounts.

Expected runtime: ~5 minutes on CPU, ~2 minutes on GPU.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score, confusion_matrix


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ARTIFACTS_DIR = Path('artifacts')
ARTIFACTS_DIR.mkdir(exist_ok=True)
OUT_PATH = ARTIFACTS_DIR / 'synthetic_v1_results.pkl'
print(f'OUT_PATH: {OUT_PATH}')


# -----------------------------------------------------------------------------
# 1. Synthetic data generation
# -----------------------------------------------------------------------------
def generate_synthetic_panel(U=30, T=50, seed=42):
    rng = np.random.default_rng(seed)
    c1, c2 = -0.6, 0.6
    rho_y = {0: 0.5, 1: 0.8, 2: 0.7}
    beta_x = {0: 0.05, 1: 0.20, 2: 0.40}
    sigma_y = 0.3
    sigma_x_innov = 0.5

    Y = np.zeros((U, T))
    X = np.zeros((U, T))
    R_true = np.zeros((U, T), dtype=int)
    intercepts = rng.normal(0, 0.5, size=U)

    for u in range(U):
        Y[u, 0] = rng.normal(intercepts[u], 0.3)
        X[u, 0] = rng.normal(0, 1)
        R_true[u, 0] = 0 if Y[u, 0] < c1 else (1 if Y[u, 0] < c2 else 2)
        for t in range(1, T):
            X[u, t] = 0.9 * X[u, t-1] + rng.normal(0, sigma_x_innov)
            r_prev = 0 if Y[u, t-1] < c1 else (1 if Y[u, t-1] < c2 else 2)
            R_true[u, t] = r_prev
            Y[u, t] = (intercepts[u] + rho_y[r_prev] * Y[u, t-1] +
                       beta_x[r_prev] * X[u, t-1] + rng.normal(0, sigma_y))

    return Y, X, R_true, intercepts, {'c1': c1, 'c2': c2, 'rho_y': rho_y, 'beta_x': beta_x}


print('Generating synthetic panel...')
Y_data, X_data, R_true, intercepts_true, true_params = generate_synthetic_panel(U=30, T=50, seed=42)
U, T = Y_data.shape
print(f'  Shape: {U} units x {T} years')
print(f'  Regime occupancy (true): '
      f'{(R_true == 0).mean():.2%} / {(R_true == 1).mean():.2%} / {(R_true == 2).mean():.2%}')
true_transitions = (R_true[:, 1:] != R_true[:, :-1]).sum(axis=1)
true_alpha = float(true_transitions.mean() / (T - 1))
print(f'  True punctuation rate: {true_alpha:.4f}')


# -----------------------------------------------------------------------------
# 2. Within-unit temporal split (last 10 years held out per unit)
# -----------------------------------------------------------------------------
L = 5
T_TEST = 10


def build_tensors_temporal_split(Y_data, X_data, R_true, L, T_test):
    U, T = Y_data.shape
    sets = {'tr': [], 'te': []}
    for u in range(U):
        for t in range(L, T):
            split = 'tr' if t < T - T_test else 'te'
            sets[split].append((Y_data[u, t-L:t], X_data[u, t-L:t],
                                Y_data[u, t], R_true[u, t], u, t))
    out = {}
    for k in ('tr', 'te'):
        Yh, Xh, Yt, R, uu, tt = zip(*sets[k])
        out[k] = (np.stack(Yh)[:, :, None], np.stack(Xh)[:, :, None],
                  np.array(Yt), np.array(R), np.array(uu), np.array(tt))
    return out['tr'], out['te']


(Y_h_tr, X_h_tr, Y_t_tr, R_tr, u_tr, t_tr), \
(Y_h_te, X_h_te, Y_t_te, R_te, u_te, t_te) = build_tensors_temporal_split(
    Y_data, X_data, R_true, L, T_TEST)
print(f'\nTemporal split: train={len(Y_h_tr)}, test={len(Y_h_te)}')


# -----------------------------------------------------------------------------
# 3. Architectures (M2 has frozen oracle centers)
# -----------------------------------------------------------------------------
class PerEdgeNet(nn.Module):
    def __init__(self, L, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(L, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class M1_NavarYOnly(nn.Module):
    def __init__(self, L, hidden=32):
        super().__init__()
        self.f_yy = PerEdgeNet(L, hidden)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, Y_hist, X_hist):
        return self.bias + self.f_yy(Y_hist[:, :, 0])


class M15_PanarNoRegimes(nn.Module):
    def __init__(self, L, hidden=32):
        super().__init__()
        self.f_yy = PerEdgeNet(L, hidden)
        self.f_xy = PerEdgeNet(L, hidden)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, Y_hist, X_hist):
        return self.bias + self.f_yy(Y_hist[:, :, 0]) + self.f_xy(X_hist[:, :, 0])


class M2_RegimeAware(nn.Module):
    def __init__(self, L, hidden=32, n_regimes=3, tau_init=0.5,
                 frozen_centers=(-0.6, 0.0, 0.6)):
        super().__init__()
        self.n_regimes = n_regimes
        self.f_yy = nn.ModuleList([PerEdgeNet(L, hidden) for _ in range(n_regimes)])
        self.f_xy = nn.ModuleList([PerEdgeNet(L, hidden) for _ in range(n_regimes)])
        self.bias = nn.Parameter(torch.zeros(n_regimes))
        self.register_buffer('centers', torch.tensor(frozen_centers, dtype=torch.float32))
        self.log_tau = nn.Parameter(torch.tensor(np.log(tau_init), dtype=torch.float32))

    def gating(self, psi):
        tau = torch.exp(self.log_tau)
        d2 = (psi.unsqueeze(-1) - self.centers.unsqueeze(0)) ** 2
        return F.softmax(-d2 / tau, dim=-1)

    def per_regime_outputs(self, Y_hist, X_hist):
        return torch.stack([
            self.bias[r] + self.f_yy[r](Y_hist[:, :, 0]) + self.f_xy[r](X_hist[:, :, 0])
            for r in range(self.n_regimes)
        ], dim=-1)

    def forward(self, Y_hist, X_hist):
        psi = Y_hist[:, -1, 0]
        pi = self.gating(psi)
        per_r = self.per_regime_outputs(Y_hist, X_hist)
        return (pi * per_r).sum(dim=-1)


# -----------------------------------------------------------------------------
# 4. Honest Jacobian gradient-norm estimator (used during training)
# -----------------------------------------------------------------------------
def jacobian_row_norm_estimate(model_fn, Y_hist_batch, X_hist_batch, channel='Y'):
    """Mean L2 grad norm over the batch — proxy for Jacobian spectral norm
    in scalar-output case. Audit version uses MAX over samples instead."""
    if channel == 'Y':
        Y_h_in = Y_hist_batch.clone().requires_grad_(True)
        out = model_fn(Y_h_in, X_hist_batch)
        grad = torch.autograd.grad(out.sum(), Y_h_in, create_graph=True)[0]
    else:
        X_h_in = X_hist_batch.clone().requires_grad_(True)
        out = model_fn(Y_hist_batch, X_h_in)
        grad = torch.autograd.grad(out.sum(), X_h_in, create_graph=True)[0]
    return grad.flatten(1).norm(dim=1).mean()


def per_regime_predictor_fn(model, r):
    def fn(Y_h, X_h):
        return model.bias[r] + model.f_yy[r](Y_h[:, :, 0]) + model.f_xy[r](X_h[:, :, 0])
    return fn


# -----------------------------------------------------------------------------
# 5. Training loop with optional Y-, X-channel and disagreement penalties
# -----------------------------------------------------------------------------
def train_model(model, Y_h_tr, X_h_tr, Y_t_tr, epochs=300, lr=3e-4,
                lam_Y=0.0, lam_X=0.0, lam_phi=0.0, lam_M=0.0,
                target_LY=0.95, target_LX=0.95,
                weight_decay=1e-4, batch=128, verbose=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Yh_t = torch.tensor(Y_h_tr, dtype=torch.float32)
    Xh_t = torch.tensor(X_h_tr, dtype=torch.float32)
    Yt_t = torch.tensor(Y_t_tr, dtype=torch.float32)
    N = len(Yh_t)

    losses = []
    for ep in range(epochs):
        perm = torch.randperm(N)
        ep_loss = 0.0
        for s in range(0, N, batch):
            idx = perm[s:s+batch]
            opt.zero_grad()
            pred = model(Yh_t[idx], Xh_t[idx])
            mse = F.mse_loss(pred, Yt_t[idx])
            loss = mse

            if isinstance(model, M2_RegimeAware) and (lam_Y > 0 or lam_X > 0):
                spec_pen_Y = 0.0
                spec_pen_X = 0.0
                for r in range(model.n_regimes):
                    fn_r = per_regime_predictor_fn(model, r)
                    if lam_Y > 0:
                        sigma_Y = jacobian_row_norm_estimate(
                            fn_r, Yh_t[idx], Xh_t[idx], channel='Y')
                        spec_pen_Y = spec_pen_Y + F.relu(sigma_Y - target_LY) ** 2
                    if lam_X > 0:
                        sigma_X = jacobian_row_norm_estimate(
                            fn_r, Yh_t[idx], Xh_t[idx], channel='X')
                        spec_pen_X = spec_pen_X + F.relu(sigma_X - target_LX) ** 2
                if lam_Y > 0:
                    loss = loss + lam_Y * spec_pen_Y
                if lam_X > 0:
                    loss = loss + lam_X * spec_pen_X

            if isinstance(model, M2_RegimeAware) and lam_phi > 0:
                inv_tau = torch.exp(-model.log_tau)
                loss = loss + lam_phi * inv_tau ** 2

            if isinstance(model, M2_RegimeAware) and lam_M > 0:
                per_r = model.per_regime_outputs(Yh_t[idx], Xh_t[idx])
                disagree_pen = 0.0
                for r1 in range(model.n_regimes):
                    for r2 in range(r1+1, model.n_regimes):
                        disagree_pen = disagree_pen + ((per_r[:, r1] - per_r[:, r2]) ** 2).mean()
                loss = loss + lam_M * disagree_pen

            loss.backward()
            opt.step()
            ep_loss += mse.item() * len(idx)
        losses.append(ep_loss / N)
        if verbose and (ep + 1) % 50 == 0:
            print(f'    ep {ep+1}: train MSE = {losses[-1]:.4f}')

    model.eval()
    return model, losses


def eval_test_mse(model, Y_h_te, X_h_te, Y_t_te):
    with torch.no_grad():
        pred = model(torch.tensor(Y_h_te, dtype=torch.float32),
                     torch.tensor(X_h_te, dtype=torch.float32))
        return float(((pred - torch.tensor(Y_t_te, dtype=torch.float32)) ** 2).mean())


# -----------------------------------------------------------------------------
# 6. Train all baselines
# -----------------------------------------------------------------------------
torch.manual_seed(0)
np.random.seed(0)

print('\n' + '=' * 78)
print('Training baselines')
print('=' * 78)

print('M1 (Y-only NAVAR)...')
m1 = M1_NavarYOnly(L)
m1, _ = train_model(m1, Y_h_tr, X_h_tr, Y_t_tr, epochs=300)
m1_test_mse = eval_test_mse(m1, Y_h_te, X_h_te, Y_t_te)
print(f'  M1 test MSE: {m1_test_mse:.4f}')

print('M1.5 (PANAR-X no regimes)...')
m15 = M15_PanarNoRegimes(L)
m15, _ = train_model(m15, Y_h_tr, X_h_tr, Y_t_tr, epochs=300)
m15_test_mse = eval_test_mse(m15, Y_h_te, X_h_te, Y_t_te)
print(f'  M1.5 test MSE: {m15_test_mse:.4f}')

print('M2 (regime-aware PANAR-X, frozen centers, Y+X penalties)...')
m2 = M2_RegimeAware(L, n_regimes=3, tau_init=0.5, frozen_centers=(-0.6, 0.0, 0.6))
m2, _ = train_model(m2, Y_h_tr, X_h_tr, Y_t_tr, epochs=300,
                    lam_Y=0.005, lam_X=0.005, lam_phi=0.001, lam_M=0.005,
                    target_LY=0.95, target_LX=0.95)
m2_test_mse = eval_test_mse(m2, Y_h_te, X_h_te, Y_t_te)
print(f'  M2 test MSE: {m2_test_mse:.4f}')


# -----------------------------------------------------------------------------
# 7. RW + AR(1)
# -----------------------------------------------------------------------------
rw_test_mse = float(((Y_h_te[:, -1, 0] - Y_t_te) ** 2).mean())
print(f'\nRW test MSE: {rw_test_mse:.4f}')

y_lag_tr = Y_h_tr[:, -1, 0]
X_ar = np.column_stack([np.ones_like(y_lag_tr), y_lag_tr])
ar_coef, *_ = np.linalg.lstsq(X_ar, Y_t_tr, rcond=None)
ar_intercept, ar_rho = float(ar_coef[0]), float(ar_coef[1])
ar_pred_te = ar_intercept + ar_rho * Y_h_te[:, -1, 0]
ar1_test_mse = float(((ar_pred_te - Y_t_te) ** 2).mean())
print(f'AR(1) w/intercept test MSE: {ar1_test_mse:.4f}  '
      f'(intercept={ar_intercept:.4f}, rho={ar_rho:.4f})')


# -----------------------------------------------------------------------------
# 8. Theorem constant audit on trained M2 (MAX over samples — conservative)
# -----------------------------------------------------------------------------
print('\n' + '=' * 78)
print('EMPIRICAL THEOREM CONSTANTS')
print('=' * 78)

m2.eval()
Yh_te_t = torch.tensor(Y_h_te, dtype=torch.float32)
Xh_te_t = torch.tensor(X_h_te, dtype=torch.float32)


def audit_per_regime_Lipschitz(model, Y_h_t, X_h_t, n_samples=200):
    L_Y = np.zeros(model.n_regimes)
    L_X = np.zeros(model.n_regimes)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(Y_h_t), size=min(n_samples, len(Y_h_t)), replace=False)
    for r in range(model.n_regimes):
        max_Y, max_X = 0.0, 0.0
        for i in sample_idx:
            Y_h = Y_h_t[i:i+1].clone().requires_grad_(True)
            X_h = X_h_t[i:i+1].clone().requires_grad_(True)
            pred_r = model.bias[r] + model.f_yy[r](Y_h[:, :, 0]) + model.f_xy[r](X_h[:, :, 0])
            gY = torch.autograd.grad(pred_r.sum(), Y_h, retain_graph=True)[0]
            gX = torch.autograd.grad(pred_r.sum(), X_h)[0]
            ny = float(gY.flatten().norm())
            nx = float(gX.flatten().norm())
            if ny > max_Y: max_Y = ny
            if nx > max_X: max_X = nx
        L_Y[r] = max_Y
        L_X[r] = max_X
    return L_Y, L_X


L_Y_per_regime, L_X_per_regime = audit_per_regime_Lipschitz(m2, Yh_te_t, Xh_te_t)
print(f'Per-regime L_Y: {L_Y_per_regime}')
print(f'Per-regime L_X: {L_X_per_regime}')
L_max = float(L_Y_per_regime.max())
LX_max = float(L_X_per_regime.max())
print(f'L_max = {L_max:.4f}, LX_max = {LX_max:.4f}')

with torch.no_grad():
    per_r_preds = m2.per_regime_outputs(Yh_te_t, Xh_te_t).numpy()
M_emp = 0.0
for r1 in range(m2.n_regimes):
    for r2 in range(r1+1, m2.n_regimes):
        d = float(np.abs(per_r_preds[:, r1] - per_r_preds[:, r2]).max())
        if d > M_emp: M_emp = d
print(f'M (regime disagreement, max on test): {M_emp:.4f}')

tau_emp = float(torch.exp(m2.log_tau).item())
psi_test = Y_h_te[:, -1, 0]
psi_lo, psi_hi = float(psi_test.min()), float(psi_test.max())


def empirical_L_phi(model, psi_lo, psi_hi, n_grid=500):
    psi_grid = torch.tensor(np.linspace(psi_lo, psi_hi, n_grid), dtype=torch.float32)
    with torch.no_grad():
        pi = model.gating(psi_grid).numpy()
    max_slope = 0.0
    for i in range(n_grid - 1):
        d_pi = np.abs(pi[i+1] - pi[i]).sum()
        d_psi = float(psi_grid[i+1] - psi_grid[i])
        if d_psi > 1e-8:
            slope = d_pi / d_psi
            if slope > max_slope: max_slope = slope
    return max_slope


L_phi_empirical = empirical_L_phi(m2, psi_lo, psi_hi)
centers_emp = m2.centers.numpy()
max_dist = max(abs(p - c) for p in [psi_lo, psi_hi] for c in centers_emp)
L_phi_analytic = float(2 * max_dist / tau_emp)
L_phi = L_phi_empirical
print(f'\ntau = {tau_emp:.4f}')
print(f'L_phi (empirical, dense grid sweep): {L_phi_empirical:.4f}')
print(f'L_phi (analytic conservative bound): {L_phi_analytic:.4f}')


def estimate_L_phi_eq(model, psi_test, c1=-0.6, c2=0.6):
    L_phi_eq = np.zeros(3)
    masks = [psi_test < c1, (psi_test >= c1) & (psi_test < c2), psi_test >= c2]
    for r, mask in enumerate(masks):
        if mask.sum() < 10:
            L_phi_eq[r] = float('nan')
            continue
        psi_r = psi_test[mask]
        psi_lo_r, psi_hi_r = np.percentile(psi_r, [25, 75])
        psi_sweep = torch.tensor(np.linspace(psi_lo_r, psi_hi_r, 50), dtype=torch.float32)
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


L_phi_eq_per_regime = estimate_L_phi_eq(m2, psi_test)
print(f'L_phi_eq per regime: {L_phi_eq_per_regime}')

L_eq_per_regime = L_Y_per_regime + L_phi_eq_per_regime * M_emp
L_eq_max = float(L_eq_per_regime.max())
B_trans = L_max + L_phi * M_emp
print(f'\nL_eq per regime: {L_eq_per_regime}')
print(f'L_eq,max = {L_eq_max:.4f}')
print(f'B_trans = {B_trans:.4f}')

if L_eq_max < 1 and B_trans > 1:
    alpha_star = -np.log(L_eq_max) / (np.log(B_trans) - np.log(L_eq_max))
    print(f'alpha* = {alpha_star:.4f}')
else:
    alpha_star = None
    print(f'alpha* UNDEFINED: L_eq_max={L_eq_max:.4f}, B_trans={B_trans:.4f}')

# Empirical alpha(h)
psi_te_t = torch.tensor(psi_test, dtype=torch.float32)
with torch.no_grad():
    pi_te = m2.gating(psi_te_t).numpy()
r_predicted_te = np.argmax(pi_te, axis=1)

per_unit_alpha = []
for u in range(U):
    u_mask = u_te == u
    if u_mask.sum() < 2: continue
    r_u = r_predicted_te[u_mask]
    transitions = (r_u[1:] != r_u[:-1]).sum()
    per_unit_alpha.append(transitions / (len(r_u) - 1))
alpha_emp = float(np.mean(per_unit_alpha))
print(f'\nEmpirical alpha(h) on test: {alpha_emp:.4f}  (true alpha: {true_alpha:.4f})')

if alpha_star is not None:
    rho = float(L_eq_max ** (1 - alpha_emp) * B_trans ** alpha_emp)
    rate_holds = (alpha_emp < alpha_star)
    print(f'Empirical rho: {rho:.4f}')
    print(f'alpha < alpha*: {rate_holds}')
else:
    rho = None
    rate_holds = False

# C_Y constant
C_Y = L_max + L_phi * M_emp


# -----------------------------------------------------------------------------
# 9. Regime alignment
# -----------------------------------------------------------------------------
print('\n' + '=' * 78)
print('REGIME ALIGNMENT (oracle-aligned centers)')
print('=' * 78)

ari_te = float(adjusted_rand_score(R_te, r_predicted_te))
ari_tr = float(adjusted_rand_score(R_tr, np.argmax(
    m2.gating(torch.tensor(Y_h_tr[:, -1, 0], dtype=torch.float32)).detach().numpy(), axis=1)))
print(f'ARI (train): {ari_tr:.4f}')
print(f'ARI (test):  {ari_te:.4f}')
cm_te = confusion_matrix(R_te, r_predicted_te)


# -----------------------------------------------------------------------------
# 10. Save
# -----------------------------------------------------------------------------
print('\n' + '=' * 78)
print('BASELINE SUMMARY')
print('=' * 78)
print(f'  RW            : {rw_test_mse:.4f}')
print(f'  AR(1) +intcpt : {ar1_test_mse:.4f}')
print(f'  M1   (Y NN)   : {m1_test_mse:.4f}')
print(f'  M1.5 (Y+X NN) : {m15_test_mse:.4f}')
print(f'  M2   (regime) : {m2_test_mse:.4f}')

out = {
    'true_params': true_params,
    'true_alpha': true_alpha,
    'baselines': {
        'RW': rw_test_mse, 'AR1': ar1_test_mse,
        'AR1_intercept': ar_intercept, 'AR1_rho': ar_rho,
        'M1': m1_test_mse, 'M1.5': m15_test_mse, 'M2': m2_test_mse,
    },
    'theorem_constants': {
        'L_Y_per_regime': L_Y_per_regime.tolist(),
        'L_X_per_regime': L_X_per_regime.tolist(),
        'L_max': L_max, 'LX_max': LX_max,
        'M': M_emp, 'tau': tau_emp,
        'L_phi_empirical': L_phi_empirical, 'L_phi_analytic': L_phi_analytic,
        'L_phi_eq_per_regime': L_phi_eq_per_regime.tolist(),
        'L_eq_per_regime': L_eq_per_regime.tolist(),
        'L_eq_max': L_eq_max, 'B_trans': B_trans,
        'alpha_star': alpha_star, 'alpha_emp': alpha_emp,
        'rho': rho, 'rate_holds': rate_holds, 'C_Y': C_Y,
    },
    'regime_alignment': {
        'ari_train': ari_tr, 'ari_test': ari_te,
        'confusion_matrix_test': cm_te.tolist(),
    },
}
with open(OUT_PATH, 'wb') as f:
    pickle.dump(out, f)
print(f'\nSaved to {OUT_PATH}')
