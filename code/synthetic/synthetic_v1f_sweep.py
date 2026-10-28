"""
synthetic_v1f_sweep.py

Port of the original synthetic v1f frontier sweep cell. Maps the
stability-accuracy Pareto frontier across:
  - 3 enforcement modes: soft_only, projection_every_epoch, projection_every_batch
  - 3 disagreement-penalty values: lam_M in {0.00, 0.05, 0.10}
  - 5 seeds: {0, 1, 2, 3, 4}
= 45 runs total.

Loads `artifacts/synthetic_v1_results.pkl` (from synthetic_v1_baselines.py)
to reuse the M1 and M1.5 baseline test-MSE references for gap computation.

Path-portable: assumes you run from the package root, reading from and
writing to artifacts/. No Colab Drive mounts.

Expected runtime: ~30 minutes on T4 GPU, ~4.5 hours on CPU.
"""
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp_stats


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ARTIFACTS_DIR = Path('artifacts')
ARTIFACTS_DIR.mkdir(exist_ok=True)
V1_RESULTS_PATH = ARTIFACTS_DIR / 'synthetic_v1_results.pkl'
OUT_PATH = ARTIFACTS_DIR / 'synthetic_v1f_frontier.pkl'

print(f'V1 results: {V1_RESULTS_PATH}')
print(f'OUT_PATH:   {OUT_PATH}')

if not V1_RESULTS_PATH.exists():
    raise FileNotFoundError(
        f'{V1_RESULTS_PATH} not found. Run synthetic_v1_baselines.py first.')

print('\nLoading v1 baselines...')
with open(V1_RESULTS_PATH, 'rb') as f:
    v1 = pickle.load(f)
m15_test_mse = v1['baselines']['M1.5']
m1_test_mse = v1['baselines']['M1']
print(f"  M1   baseline:  {m1_test_mse:.4f}")
print(f"  M1.5 baseline:  {m15_test_mse:.4f}")


# -----------------------------------------------------------------------------
# Data generation (identical to v1)
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
    return Y, X, R_true


Y_data, X_data, R_true = generate_synthetic_panel()
U, T = Y_data.shape
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
print(f"\nData: train={len(Y_h_tr)}, test={len(Y_h_te)}\n")


# -----------------------------------------------------------------------------
# Architecture (note: PerEdgeNet uses lin1, lin2 explicitly for projection)
# -----------------------------------------------------------------------------
class PerEdgeNet(nn.Module):
    def __init__(self, L, hidden=32):
        super().__init__()
        self.lin1 = nn.Linear(L, hidden)
        self.lin2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.lin2(F.relu(self.lin1(x))).squeeze(-1)


class M2_RegimeAware(nn.Module):
    def __init__(self, L, hidden=32, n_regimes=3, tau_init=3.0,
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
# Hard-projection helpers (Y-channel only per v1f design)
# -----------------------------------------------------------------------------
def project_layer(layer, target_norm):
    with torch.no_grad():
        sigma = torch.linalg.svdvals(layer.weight)[0].item()
        if sigma > target_norm:
            layer.weight.mul_(target_norm / sigma)


def project_per_edge_net(net, target_lipschitz):
    per_layer_target = float(np.sqrt(target_lipschitz))
    project_layer(net.lin1, per_layer_target)
    project_layer(net.lin2, per_layer_target)


def project_m2_y_only(model, target_LY):
    for r in range(model.n_regimes):
        project_per_edge_net(model.f_yy[r], target_LY)


# -----------------------------------------------------------------------------
# Soft penalty (smooth max via log-sum-exp)
# -----------------------------------------------------------------------------
def per_regime_predictor_fn(model, r):
    def fn(Y_h, X_h):
        return model.bias[r] + model.f_yy[r](Y_h[:, :, 0]) + model.f_xy[r](X_h[:, :, 0])
    return fn


def jacobian_softmax_norm(model_fn, Y_hist_batch, X_hist_batch, channel='Y',
                          tau_softmax=5.0):
    if channel == 'Y':
        Y_h_in = Y_hist_batch.clone().requires_grad_(True)
        out = model_fn(Y_h_in, X_hist_batch)
        grad = torch.autograd.grad(out.sum(), Y_h_in, create_graph=True)[0]
    else:
        X_h_in = X_hist_batch.clone().requires_grad_(True)
        out = model_fn(Y_hist_batch, X_h_in)
        grad = torch.autograd.grad(out.sum(), X_h_in, create_graph=True)[0]
    norms = grad.flatten(1).norm(dim=1)
    B = norms.shape[0]
    return (torch.logsumexp(tau_softmax * norms, dim=0) - np.log(B)) / tau_softmax


# -----------------------------------------------------------------------------
# Training: three enforcement modes
# -----------------------------------------------------------------------------
def train_m2_unified(model, Y_h_tr, X_h_tr, Y_t_tr, mode,
                     epochs=300, lr=3e-4, lam_M=0.10, lam_phi=0.02,
                     weight_decay=1e-4, batch=128):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Yh_t = torch.tensor(Y_h_tr, dtype=torch.float32)
    Xh_t = torch.tensor(X_h_tr, dtype=torch.float32)
    Yt_t = torch.tensor(Y_t_tr, dtype=torch.float32)
    N = len(Yh_t)

    if mode == 'soft_only':
        lam_Y = 0.20
        target_LY_soft = 0.55
        do_project_batch = False
        do_project_epoch = False
        target_LY_hard = None
    elif mode == 'projection_every_epoch':
        lam_Y = 0.05
        target_LY_soft = 0.85
        target_LY_hard = 0.90
        do_project_batch = False
        do_project_epoch = True
        project_m2_y_only(model, target_LY_hard)
    elif mode == 'projection_every_batch':
        lam_Y = 0.0
        target_LY_soft = None
        target_LY_hard = 0.85
        do_project_batch = True
        do_project_epoch = False
        project_m2_y_only(model, target_LY_hard)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for ep in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, N, batch):
            idx = perm[s:s+batch]
            opt.zero_grad()
            pred = model(Yh_t[idx], Xh_t[idx])
            loss = F.mse_loss(pred, Yt_t[idx])

            if lam_Y > 0:
                spec_pen_Y = torch.zeros((), device=Yh_t.device)
                for r in range(model.n_regimes):
                    fn_r = per_regime_predictor_fn(model, r)
                    sigma_Y = jacobian_softmax_norm(fn_r, Yh_t[idx], Xh_t[idx],
                                                    channel='Y', tau_softmax=5.0)
                    spec_pen_Y = spec_pen_Y + F.relu(sigma_Y - target_LY_soft) ** 2
                loss = loss + lam_Y * spec_pen_Y

            if lam_phi > 0:
                inv_tau = torch.exp(-model.log_tau)
                loss = loss + lam_phi * inv_tau ** 2

            if lam_M > 0:
                per_r = model.per_regime_outputs(Yh_t[idx], Xh_t[idx])
                disagree = torch.zeros((), device=Yh_t.device)
                for r1 in range(model.n_regimes):
                    for r2 in range(r1+1, model.n_regimes):
                        disagree = disagree + ((per_r[:, r1] - per_r[:, r2]) ** 2).mean()
                loss = loss + lam_M * disagree

            loss.backward()
            opt.step()

            if do_project_batch:
                project_m2_y_only(model, target_LY_hard)

        if do_project_epoch:
            project_m2_y_only(model, target_LY_hard)

    model.eval()
    return model


# -----------------------------------------------------------------------------
# Audit functions (max over samples — the conservative theorem constants)
# -----------------------------------------------------------------------------
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


def empirical_L_phi(model, psi_lo, psi_hi, n_grid=500):
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


def empirical_L_phi_eq(model, psi_test, c1=-0.6, c2=0.6):
    L_phi_eq = np.zeros(3)
    masks = [psi_test < c1, (psi_test >= c1) & (psi_test < c2), psi_test >= c2]
    for r, mask in enumerate(masks):
        if mask.sum() < 10:
            L_phi_eq[r] = float('nan'); continue
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
    return float(np.mean(per_unit_alpha))


def compute_rho(L_eq_max, B_trans, alpha_emp):
    if L_eq_max >= 1:
        return None, None
    if B_trans <= 1:
        alpha_star = float('inf')
    else:
        alpha_star = float(-np.log(L_eq_max) / (np.log(B_trans) - np.log(L_eq_max)))
    rho = float(L_eq_max ** (1 - alpha_emp) * B_trans ** alpha_emp)
    return alpha_star, rho


def full_audit(model, Y_h_te, X_h_te, Y_t_te, u_te, U, L):
    Yh_te_t = torch.tensor(Y_h_te, dtype=torch.float32)
    Xh_te_t = torch.tensor(X_h_te, dtype=torch.float32)
    psi_te = Y_h_te[:, -1, 0]

    L_Y, L_X = audit_per_regime_Lipschitz(model, Yh_te_t, Xh_te_t)
    L_max = float(L_Y.max())
    LX_max = float(L_X.max())
    M_emp = get_M(model, Yh_te_t, Xh_te_t)
    tau_emp = float(torch.exp(model.log_tau).item())
    L_phi = empirical_L_phi(model, float(psi_te.min()), float(psi_te.max()))
    L_phi_eq = empirical_L_phi_eq(model, psi_te)

    L_eq_per_regime = L_Y + L_phi_eq * M_emp
    L_eq_max = float(L_eq_per_regime.max())
    L_phi_eq_max = float(L_phi_eq.max())
    L_phi_eq_M = L_phi_eq_max * M_emp
    B_trans = L_max + L_phi * M_emp

    alpha_emp = compute_alpha_emp(model, Y_h_te, u_te, U)
    alpha_star, rho = compute_rho(L_eq_max, B_trans, alpha_emp)

    with torch.no_grad():
        pred = model(Yh_te_t, Xh_te_t)
        test_mse = float(((pred - torch.tensor(Y_t_te, dtype=torch.float32)) ** 2).mean())

    return {
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
# Sweep
# -----------------------------------------------------------------------------
ENFORCEMENT_MODES = ['soft_only', 'projection_every_epoch', 'projection_every_batch']
LAM_M_GRID = [0.00, 0.05, 0.10]
SEEDS = [0, 1, 2, 3, 4]
MODE_SHORT = {'soft_only': 'soft', 'projection_every_epoch': 'proj_ep',
              'projection_every_batch': 'proj_ba'}

results = {}
total_runs = len(ENFORCEMENT_MODES) * len(LAM_M_GRID) * len(SEEDS)
run_idx = 0
t_sweep_start = time.time()

print(f"Starting v1f frontier sweep: {total_runs} runs")
print(f"  {len(ENFORCEMENT_MODES)} enforcement modes x {len(LAM_M_GRID)} lam_M values x {len(SEEDS)} seeds\n")

for mode in ENFORCEMENT_MODES:
    for lam_M in LAM_M_GRID:
        cell_key = (mode, lam_M)
        results[cell_key] = []
        for seed in SEEDS:
            run_idx += 1
            t_run = time.time()
            torch.manual_seed(seed); np.random.seed(seed)
            m2 = M2_RegimeAware(L, n_regimes=3, tau_init=3.0)
            m2 = train_m2_unified(m2, Y_h_tr, X_h_tr, Y_t_tr, mode=mode,
                                  epochs=300, lam_M=lam_M, lam_phi=0.02)
            audit = full_audit(m2, Y_h_te, X_h_te, Y_t_te, u_te, U, L)
            audit['seed'] = seed
            audit['mode'] = mode
            audit['lam_M'] = lam_M
            results[cell_key].append(audit)

            elapsed = time.time() - t_run
            sweep_elapsed = time.time() - t_sweep_start
            eta = sweep_elapsed * (total_runs - run_idx) / run_idx
            rho_str = f"{audit['rho']:.3f}" if audit['rho'] is not None else "N/A"
            print(f"[{run_idx}/{total_runs}] mode={MODE_SHORT[mode]}, lam_M={lam_M}, seed={seed}: "
                  f"L_eq,max={audit['L_eq_max']:.3f}, MSE={audit['test_mse']:.4f}, "
                  f"M={audit['M']:.3f}, rho={rho_str}  "
                  f"({elapsed:.0f}s, ETA {eta/60:.0f}min)")

            with open(OUT_PATH, 'wb') as f:
                pickle.dump({'results': results, 'm15_test_mse': m15_test_mse,
                             'm1_test_mse': m1_test_mse}, f)

print(f"\nSweep complete. Total time: {(time.time() - t_sweep_start)/60:.1f} minutes")


# -----------------------------------------------------------------------------
# Per-cell summary with t-CIs
# -----------------------------------------------------------------------------
def t_ci(values, conf=0.95):
    n = len(values)
    if n < 2:
        return float('nan'), float('nan')
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(n)
    t_crit = sp_stats.t.ppf((1 + conf) / 2, df=n-1)
    return mean - t_crit * se, mean + t_crit * se


print("\n" + "=" * 130)
print("PER-CELL SUMMARY (n=5 seeds; positive gap = M2 better than M1.5)")
print("=" * 130)
print(f"{'mode':>22} {'lam_M':>6}  {'L_eq,max':>10}  {'M':>8}  "
      f"{'L_phi_eq*M':>12}  {'M1.5-M2 gap':>14}  {'L_eq<1':>8}")
print("-" * 130)

for mode in ENFORCEMENT_MODES:
    for lam_M in LAM_M_GRID:
        cell = results[(mode, lam_M)]
        L_eq = [r['L_eq_max'] for r in cell]
        M_vals = [r['M'] for r in cell]
        L_phi_eq_M = [r['L_phi_eq_M'] for r in cell]
        mse = [r['test_mse'] for r in cell]
        gap = [m15_test_mse - m for m in mse]
        gap_ci = t_ci(gap)
        bound_holds = sum(1 for r in cell if r['L_eq_max'] < 1)
        print(f"{mode:>22} {lam_M:>6.2f}  {np.mean(L_eq):>10.3f}  {np.mean(M_vals):>8.3f}  "
              f"{np.mean(L_phi_eq_M):>12.3f}  {np.mean(gap):>+8.4f}  {bound_holds}/5")

print(f"\nSaved to {OUT_PATH}")
