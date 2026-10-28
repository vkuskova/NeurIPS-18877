"""
Diagnostic + recovery figure for V-Dem demonstration. v3.

Layout:
- Row 1: cancellation diagnostic (linear coefficients)
  A. Heatmap of linear coefficients at lag t-4 across {pooled, R0, R1, R2}
  B. Bar chart focused on v2psorgs t-4

- Row 2: per-regime CUMULATIVE squared-error advantage
  Walking through the test set (sorted by absolute paired difference desc),
  show cumulative sum of (e_pool^2 - e_M2^2). Final value = N * mean_diff,
  exactly what the DM test averages. Visually shows where the MSE
  advantage comes from (concentrated in tails or spread evenly).
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression

DATA_PATH = 'data/merged_completevars_balanced_zero_missing.csv'
PRED_PATH = 'artifacts/vdem_v1h_postanalysis.pkl'
Y_VAR = 'v2xel_frefair'
X_VARS = ['v2csantimv', 'v2jureform', 'v2psorgs', 'v2mecorrpt']
X_LABELS = ['anti-system\nmovements', 'judicial\nreform', 'party\norganization', 'media\ncorruption']
L_LAG = 5
TEST_YEARS_PER_COUNTRY = 10

with open(PRED_PATH, 'rb') as f:
    pred = pickle.load(f)
target = pred['targets']
regime = pred['regime_assignment']
nn_pool = pred['predictions']['NN_pooled_M1.5']
m2_mean = pred['predictions']['M2_mean']

df = pd.read_csv(DATA_PATH).sort_values(['country_id', 'year']).reset_index(drop=True)

def build_obs(df):
    Y_h, X_h, t_arr, y1 = [], [], [], []
    for cid, g in df.groupby('country_id'):
        g = g.sort_values('year').reset_index(drop=True)
        T = len(g)
        if T < L_LAG + TEST_YEARS_PER_COUNTRY + 1: continue
        for t in range(L_LAG, T):
            Y_h.append(g[Y_VAR].iloc[t-L_LAG:t].values)
            X_h.append(g[X_VARS].iloc[t-L_LAG:t].values)
            t_arr.append(float(g[Y_VAR].iloc[t] - g[Y_VAR].iloc[t-1]))
            y1.append(float(g[Y_VAR].iloc[t-1]))
    return (np.array(Y_h, dtype=np.float32),
            np.stack(X_h).astype(np.float32),
            np.array(t_arr, dtype=np.float32),
            np.array(y1, dtype=np.float32))

Y_h, X_h, target_full, y_lag1_full = build_obs(df)
N_total = len(target_full)
is_test = np.zeros(N_total, dtype=bool)
ptr = 0
for cid, g in df.groupby('country_id'):
    T = len(g)
    if T < L_LAG + TEST_YEARS_PER_COUNTRY + 1: continue
    n_added = T - L_LAG
    is_test[ptr + n_added - TEST_YEARS_PER_COUNTRY : ptr + n_added] = True
    ptr += n_added

Y_h_tr, X_h_tr, t_tr, y1_tr = Y_h[~is_test], X_h[~is_test], target_full[~is_test], y_lag1_full[~is_test]
b1, b2 = np.percentile(y1_tr, 100/3), np.percentile(y1_tr, 200/3)
regime_tr = np.where(y1_tr < b1, 0, np.where(y1_tr < b2, 1, 2))

X_mat = np.concatenate([Y_h_tr, X_h_tr.reshape(len(X_h_tr), -1)], axis=1)
pooled_lin = LinearRegression().fit(X_mat, t_tr)
strat_lin = {r: LinearRegression().fit(X_mat[regime_tr == r], t_tr[regime_tr == r]) for r in range(3)}

def get_X_coefs(model):
    return model.coef_[5:].reshape(5, 4)

pooled_X = get_X_coefs(pooled_lin)
strat_X = np.stack([get_X_coefs(strat_lin[r]) for r in range(3)])

# === FIGURE ===
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.titlesize'] = 9.5
plt.rcParams['figure.dpi'] = 150

fig = plt.figure(figsize=(7.0, 5.8))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.0],
              hspace=0.55, wspace=0.42,
              top=0.86, bottom=0.10, left=0.10, right=0.97)

regime_colors = ['#d62728', '#7f7f7f', '#1f77b4']
pooled_color = '#2ca02c'
m2_color = '#9467bd'

# ---- Panel A: heatmap
ax_a = fig.add_subplot(gs[0, 0:2])
LAG_IDX = 1  # t-4
heat = np.zeros((4, 4))
heat[:, 0] = pooled_X[LAG_IDX, :]
for r in range(3):
    heat[:, r+1] = strat_X[r, LAG_IDX, :]

vmax = max(abs(heat.min()), abs(heat.max()))
im = ax_a.imshow(heat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
ax_a.set_xticks(range(4))
ax_a.set_xticklabels(['Pooled', 'Regime 0\n(low)', 'Regime 1\n(mid)', 'Regime 2\n(high)'])
ax_a.set_yticks(range(4))
ax_a.set_yticklabels(X_LABELS)
ax_a.set_title('A. Linear coefficients at lag $t{-}4$, by regime\n'
               'Sign reversals across regimes cancel under pooling',
               fontsize=10)
for i in range(4):
    for j in range(4):
        v = heat[i, j]
        color = 'white' if abs(v) > 0.7*vmax else 'black'
        ax_a.text(j, i, f'{v:+.3f}', ha='center', va='center', color=color, fontsize=8)
cbar = plt.colorbar(im, ax=ax_a, shrink=0.7, pad=0.02)
cbar.set_label('coefficient', fontsize=8)
cbar.ax.tick_params(labelsize=7)

# ---- Panel B: party org bar chart
ax_b = fig.add_subplot(gs[0, 2])
J_FOCUS = 2
xpos = np.arange(4)
vals = [pooled_X[LAG_IDX, J_FOCUS]] + [strat_X[r, LAG_IDX, J_FOCUS] for r in range(3)]
colors = [pooled_color] + regime_colors
labels = ['Pooled', 'R0', 'R1', 'R2']
bars = ax_b.bar(xpos, vals, color=colors, edgecolor='black', linewidth=0.5)
ax_b.axhline(0, color='black', lw=0.5)
ax_b.set_xticks(xpos)
ax_b.set_xticklabels(labels, fontsize=8)
ax_b.set_ylabel('coef. on $X_{t-4}$', fontsize=8)
ax_b.set_title('B. Party org.\n(focus from A)', fontsize=10)
for b, v in zip(bars, vals):
    h = b.get_height()
    ax_b.text(b.get_x() + b.get_width()/2,
              h + 0.0008 * (1 if h >= 0 else -1),
              f'{v:+.3f}', ha='center',
              va='bottom' if h >= 0 else 'top', fontsize=7.5)
ax_b.set_ylim(-0.027, 0.027)

# ---- Row 2: cumulative paired squared-error advantage
err_pool = (nn_pool - target) ** 2
err_m2   = (m2_mean - target) ** 2

# Camera-ready relabeling (reviewer kDiP, minor): row-2 panel titles report the
# per-regime RELATIVE MSE reduction of M2 vs. the pooled NN (computed here from
# the stored predictions), not an MSE level. DM statistics are the submitted
# seed-asymmetric per-run test (M2 5-seed mean vs. single-seed pooled NN),
# reported at three decimals as in artifacts/results_18877_R1_table2_precision.csv
# (overall +2.411 p=0.016; regime0 +2.196 p=0.028; regime1 +0.299 p=0.765;
# regime2 +2.407 p=0.016). The seed-fair ensemble comparison is null; see the
# paper's Table 2 and the figure caption.
dm_3dp = {0: ('+2.196', '0.028'), 1: ('+0.299', '0.765'), 2: ('+2.407', '0.016')}
regime_names = {0: 'Regime 0 (low democracy)',
                1: 'Regime 1 (mid democracy)',
                2: 'Regime 2 (high democracy)'}
dm_info = {}
for r in range(3):
    mask = regime == r
    mse_pool_r = err_pool[mask].mean()
    mse_m2_r = err_m2[mask].mean()
    rel_red = 100.0 * (mse_pool_r - mse_m2_r) / mse_pool_r
    dm_s, p_s = dm_3dp[r]
    dm_info[r] = dict(
        name=regime_names[r],
        line1=rf'$N{{=}}{mask.sum()}$,  MSE ${rel_red:.1f}\%$ below pooled',
        line2=rf'per-run DM ${{=}}{dm_s}$,  $p{{=}}{p_s}$',
        wins=(float(p_s) < 0.05))
    print(f'regime {r}: N={mask.sum()} pooled={mse_pool_r:.6f} M2={mse_m2_r:.6f} rel_red={rel_red:.2f}%')

# Find a common y-axis range across the three regimes for cumulative sum
all_cumsums = []
for r in range(3):
    mask = regime == r
    diff = err_pool[mask] - err_m2[mask]
    # Sort by some natural ordering — use observation index (no ordering)
    # But to show "where the advantage comes from", sort by absolute paired diff descending,
    # so the cumulative sum reflects how concentrated the advantage is
    sort_idx = np.argsort(-np.abs(diff))
    diff_sorted = diff[sort_idx]
    cumsum = np.cumsum(diff_sorted)
    all_cumsums.append(cumsum)

ymax = max([abs(c).max() for c in all_cumsums])

for r in range(3):
    ax = fig.add_subplot(gs[1, r])
    mask = regime == r
    diff = err_pool[mask] - err_m2[mask]
    n = mask.sum()
    
    sort_idx = np.argsort(-np.abs(diff))  # descending by |diff|
    diff_sorted = diff[sort_idx]
    cumsum = np.cumsum(diff_sorted)
    final_value = cumsum[-1]  # = N * mean_diff
    
    x_axis = np.arange(1, n+1) / n  # fraction of test set processed
    
    ax.plot(x_axis, cumsum, color='black', lw=1.2)
    ax.fill_between(x_axis, 0, cumsum, where=cumsum >= 0,
                    color=m2_color, alpha=0.5, label='M2 advantage')
    ax.fill_between(x_axis, 0, cumsum, where=cumsum < 0,
                    color=pooled_color, alpha=0.5, label='pool advantage')
    ax.axhline(0, color='black', lw=0.5)
    
    # Mark final value on right edge
    ax.scatter([1.0], [final_value], color='black', s=18, zorder=5)
    # Annotation in lower-left of plot (below zero, where data isn't)
    ax.text(0.03, 0.03,
            f'final:\n${final_value:+.3f}$',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.25',
                       facecolor='white', edgecolor='gray',
                       alpha=0.92, linewidth=0.4))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.06, 0.12)
    ax.set_xlabel('fraction of test set\n(sorted by $|\\Delta|$ desc.)',
                  fontsize=8)
    if r == 0:
        ax.set_ylabel('cumulative $e^2_{\\mathrm{pool}}{-}e^2_{M2}$',
                      fontsize=9)
    
    info = dm_info[r]
    ax.set_title(f'{"CDE"[r]}. {info["name"]}\n{info["line1"]}\n{info["line2"]}',
                 fontsize=8.5)
    
    if info['wins']:
        ax.set_facecolor('#f4faf4')
    else:
        ax.set_facecolor('#fafafa')
    
    if r == 0:
        ax.legend(loc='center right', fontsize=7, frameon=True,
                  handlelength=1.4, handletextpad=0.5,
                  bbox_to_anchor=(1.0, 0.45))
    
    ax.grid(True, alpha=0.3, lw=0.4)
    ax.tick_params(labelsize=7.5)

# No overall figure title: the figure is captioned in the paper's LaTeX
# source (\caption{...} in section 6 of main.tex), so a matplotlib suptitle
# would duplicate the caption. Layout below is sized to fill the figure
# area without a top title band.

import os
os.makedirs('figures', exist_ok=True)
out_pdf = 'figures/figure_diagnostic.pdf'
out_png = 'figures/figure_diagnostic.png'
fig.savefig(out_pdf, bbox_inches='tight', dpi=200)
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
