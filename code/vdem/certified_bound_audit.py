"""
Streamlined certified-vs-probe audit on V-Dem PANAR-X (M2).

Trains M2 with vanilla MSE for 100 epochs (no per-batch Jacobian penalty)
and computes both the certified spectral-norm upper bound L̃_Y^(r) =
σ(W_1)·σ(W_2) and the gradient-probe sample max L̂_Y^(r) on the same
trained model. The comparison is architecture-driven: certified bound
upper-bounds gradient-probe by construction. The full v1h sweep numbers
(with per-batch Lipschitz penalty) are unchanged and reported separately.
"""
import pickle, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

DATA_PATH = 'data/merged_completevars_balanced_zero_missing.csv'
Y_VAR = 'v2xel_frefair'
X_VARS = ['v2csantimv', 'v2jureform', 'v2psorgs', 'v2mecorrpt']
L_LAG = 5
TEST_YEARS_PER_COUNTRY = 10
N_X = len(X_VARS)
TAU_INIT = 0.5

def build_panel_tensors(df, y_var, x_vars, L, test_years_per_country):
    Y_h_tr, X_h_tr, Y_t_tr = [], [], []
    Y_h_te, X_h_te, Y_t_te = [], [], []
    for cid, g in df.groupby('country_id'):
        g = g.sort_values('year').reset_index(drop=True)
        T = len(g)
        if T < L + test_years_per_country + 1: continue
        cutoff = T - test_years_per_country
        for t in range(L, T):
            y_lag = g[y_var].iloc[t-L:t].values
            x_lag = g[x_vars].iloc[t-L:t].values
            target = float(g[y_var].iloc[t]) - float(g[y_var].iloc[t-1])
            Y_h_full = y_lag.reshape(L, 1).astype(np.float32)
            X_h_full = x_lag.astype(np.float32)
            target = np.float32(target)
            if t < cutoff:
                Y_h_tr.append(Y_h_full); X_h_tr.append(X_h_full); Y_t_tr.append(target)
            else:
                Y_h_te.append(Y_h_full); X_h_te.append(X_h_full); Y_t_te.append(target)
    return (np.stack(Y_h_tr), np.stack(X_h_tr), np.array(Y_t_tr),
            np.stack(Y_h_te), np.stack(X_h_te), np.array(Y_t_te))


class PerEdge(nn.Module):
    def __init__(self, L, hidden=32):
        super().__init__()
        self.lin1 = nn.Linear(L, hidden); self.lin2 = nn.Linear(hidden, 1)
    def forward(self, x): return self.lin2(F.relu(self.lin1(x))).squeeze(-1)


class M2(nn.Module):
    def __init__(self, L, n_x, hidden=32, n_regimes=3, tau_init=TAU_INIT,
                 frozen_centers=(0.2, 0.5, 0.8)):
        super().__init__()
        self.n_regimes = n_regimes; self.n_x = n_x
        self.f_yy = nn.ModuleList([PerEdge(L, hidden) for _ in range(n_regimes)])
        self.f_xy = nn.ModuleList([
            nn.ModuleList([PerEdge(L, hidden) for _ in range(n_x)])
            for _ in range(n_regimes)])
        self.bias = nn.Parameter(torch.zeros(n_regimes))
        self.register_buffer('centers', torch.tensor(frozen_centers, dtype=torch.float32))
        self.log_tau = nn.Parameter(torch.tensor(np.log(tau_init), dtype=torch.float32))
    def gating(self, psi):
        tau = torch.exp(self.log_tau)
        d2 = (psi.unsqueeze(-1) - self.centers.unsqueeze(0)) ** 2
        return F.softmax(-d2 / tau, dim=-1)
    def forward(self, Y, X):
        psi = Y[:, -1, 0]
        pi = self.gating(psi)
        outs = []
        for r in range(self.n_regimes):
            o = self.bias[r] + self.f_yy[r](Y[:, :, 0])
            for j in range(self.n_x):
                o = o + self.f_xy[r][j](X[:, :, j])
            outs.append(o)
        per_r = torch.stack(outs, dim=-1)
        return (pi * per_r).sum(dim=-1)


def train(m, Y, X, t, epochs=100, lr=3e-3, wd=1e-4, batch=128):
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Yt = torch.tensor(Y, dtype=torch.float32)
    Xt = torch.tensor(X, dtype=torch.float32)
    tt = torch.tensor(t, dtype=torch.float32)
    N = len(Yt)
    for ep in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, N, batch):
            idx = perm[s:s+batch]
            opt.zero_grad()
            loss = F.mse_loss(m(Yt[idx], Xt[idx]), tt[idx])
            loss.backward(); opt.step()
    m.eval(); return m


def probe(m, Y_te, X_te, n=200):
    L = np.zeros(m.n_regimes)
    rng = np.random.default_rng(0)
    idx_s = rng.choice(len(Y_te), size=min(n, len(Y_te)), replace=False)
    for r in range(m.n_regimes):
        mx = 0.0
        for i in idx_s:
            Y = Y_te[i:i+1].clone().requires_grad_(True)
            X = X_te[i:i+1].clone()
            p = m.bias[r] + m.f_yy[r](Y[:, :, 0])
            for j in range(m.n_x): p = p + m.f_xy[r][j](X[:, :, j])
            g = torch.autograd.grad(p.sum(), Y)[0]
            v = float(g.flatten().norm())
            if v > mx: mx = v
        L[r] = mx
    return L


def cert(m):
    L = np.zeros(m.n_regimes)
    for r in range(m.n_regimes):
        net = m.f_yy[r]
        s1 = float(torch.linalg.svdvals(net.lin1.weight.detach())[0])
        s2 = float(torch.linalg.svdvals(net.lin2.weight.detach())[0])
        L[r] = s1 * s2
    return L


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH).sort_values(['country_id', 'year']).reset_index(drop=True)
    Yh_tr, Xh_tr, Yt_tr, Yh_te, Xh_te, Yt_te = build_panel_tensors(
        df, Y_VAR, X_VARS, L_LAG, TEST_YEARS_PER_COUNTRY)
    print(f"Loaded {time.time()-t0:.1f}s. Train {Yh_tr.shape[0]}, Test {Yh_te.shape[0]}")
    Yh_te_t = torch.tensor(Yh_te, dtype=torch.float32)
    Xh_te_t = torch.tensor(Xh_te, dtype=torch.float32)
    yl = Yh_tr[:, -1, 0]
    centers = (float(np.percentile(yl, 100/6)),
               float(np.percentile(yl, 50)),
               float(np.percentile(yl, 500/6)))
    
    results = []
    for seed in [0, 1, 2]:
        ts = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        m = M2(L=L_LAG, n_x=N_X, hidden=32, n_regimes=3,
               tau_init=TAU_INIT, frozen_centers=centers)
        train(m, Yh_tr, Xh_tr, Yt_tr, epochs=100)
        Lp = probe(m, Yh_te_t, Xh_te_t)
        Lc = cert(m)
        print(f"Seed {seed} ({time.time()-ts:.0f}s): probe={Lp}, cert={Lc}")
        results.append(dict(seed=seed, probe=Lp.tolist(), cert=Lc.tolist()))
    
    pmax = np.array([max(r['probe']) for r in results])
    cmax = np.array([max(r['cert'])  for r in results])
    print(f"\n3-seed summary:")
    print(f"  probe max: mean={pmax.mean():.4f} std={pmax.std():.4f}")
    print(f"  cert  max: mean={cmax.mean():.4f} std={cmax.std():.4f}")
    print(f"  ratio cert/probe: mean={(cmax/np.maximum(pmax,1e-9)).mean():.2f}x")
    print(f"  cert >= probe in all seeds, all regimes: {all(all(c>=p for c,p in zip(r['cert'], r['probe'])) for r in results)}")
    print(f"Total: {time.time()-t0:.1f}s")
    
    with open('artifacts/certified_audit_results.pkl', 'wb') as f:
        pickle.dump({'results': results, 'pmax_seeds': pmax.tolist(),
                     'cmax_seeds': cmax.tolist(), 'training': 'vanilla MSE 100ep'}, f)


if __name__ == '__main__':
    main()
