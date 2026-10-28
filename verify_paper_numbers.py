"""
verify_paper_numbers.py

Loads the artifact pickles in artifacts/ and verifies every numeric claim made
in the paper. Each paper-cited number is computed from a saved artifact and
compared against the value reported in the paper. Reports PASS/FAIL per check.

Run from the package root:
    python verify_paper_numbers.py

Expected runtime: < 30 seconds (no training, only arithmetic).

This script covers the SUBMITTED version's numbers. The numbers added or
changed in the NeurIPS 2026 camera-ready are checked by the companion script
verify_camera_ready_numbers.py against artifacts/rebuttal_* and artifacts/rv_*.

If every check passes, the paper's numbers are reproducible from the archived
artifacts. If a check fails, the failure message names the section and table
of the paper where the disagreeing number appears.
"""
import os
import pickle
import sys

import numpy as np
from scipy import stats

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')

n_pass = 0
n_fail = 0


def check(name, computed, expected, tol_abs=0, tol_rel=0, location=''):
    """Single-value check with absolute or relative tolerance."""
    global n_pass, n_fail
    # Bool comparison: exact
    if isinstance(computed, (bool, np.bool_)) or isinstance(expected, bool):
        ok = (bool(computed) == bool(expected))
    # String comparison: exact match
    elif isinstance(computed, str) or isinstance(expected, str):
        ok = (str(computed) == str(expected))
    else:
        diff = abs(computed - expected)
        if tol_abs > 0 and diff <= tol_abs:
            ok = True
        elif tol_rel > 0 and (expected == 0 or diff / abs(expected) <= tol_rel):
            ok = True
        elif tol_abs == 0 and tol_rel == 0:
            ok = (computed == expected)
        else:
            ok = False
    status = 'PASS' if ok else 'FAIL'
    if ok:
        n_pass += 1
    else:
        n_fail += 1
    loc_str = f' [{location}]' if location else ''
    print(f"  [{status}] {name}{loc_str}: computed={computed!r}, expected={expected!r}")
    return ok


def section(title):
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


# ============================================================================
section('SYNTHETIC v1f frontier — section 6 / table 5.1, 5.2 / section 8')
# ============================================================================

with open(os.path.join(ARTIFACTS, 'synthetic_v1f_frontier.pkl'), 'rb') as f:
    synth = pickle.load(f)

m15_base = synth['m15_test_mse']
m1_base = synth['m1_test_mse']

print('\n--- Table 5.1: baseline test MSE (5-seed mean of soft_only lam_M=0) ---')
check('M1 (Y-only PANAR) test MSE', round(m1_base, 4), 0.1895,
      tol_abs=1e-4, location='Table 5.1, paper §6.1')
check('M1.5 (additive PANAR-X) test MSE', round(m15_base, 4), 0.1497,
      tol_abs=1e-4, location='Table 5.1')

soft_lm0 = synth['results'][('soft_only', 0.0)]
m2_mses = np.array([r['test_mse'] for r in soft_lm0])
gaps_synth = m15_base - m2_mses
m2_mean_mse = m2_mses.mean()

check('M2 (regime-aware) 5-seed mean test MSE', round(m2_mean_mse, 4), 0.1410,
      tol_abs=1e-4, location='Table 5.1')

# Gain decomposition: nonlinear X (M1.5 vs M1) and regime (M2 vs M1.5)
gain_nonlin_X = m1_base - m15_base
gain_regime = gaps_synth.mean()
pct_nonlin_X = 100 * gain_nonlin_X / m1_base
pct_regime = 100 * gain_regime / m15_base

check('Gain decomp: nonlinear X channel (absolute)', round(gain_nonlin_X, 4), 0.0398,
      tol_abs=1e-4, location='Table 5.1')
check('Gain decomp: nonlinear X channel (% of M1)', round(pct_nonlin_X, 2), 20.98,
      tol_abs=0.05, location='Table 5.1')
check('Gain decomp: regime structure (absolute)', round(gain_regime, 4), 0.0087,
      tol_abs=1e-4, location='Table 5.1')
check('Gain decomp: regime structure (% of M1.5)', round(pct_regime, 2), 5.82,
      tol_abs=0.05, location='Table 5.1')

# 95% t-CI on regime-structure gap
gap_se = gaps_synth.std(ddof=1) / np.sqrt(len(gaps_synth))
t_crit = stats.t.ppf(0.975, df=len(gaps_synth) - 1)
ci_lo = gaps_synth.mean() - t_crit * gap_se
ci_hi = gaps_synth.mean() + t_crit * gap_se
check('Regime gap 95% t-CI lower', round(ci_lo, 4), 0.0065,
      tol_abs=1e-4, location='Table 5.1')
check('Regime gap 95% t-CI upper', round(ci_hi, 4), 0.0109,
      tol_abs=1e-4, location='Table 5.1')

print('\n--- §6.1: audit constants (5-seed mean of soft_only lam_M=0) ---')
L_max_arr = np.array([r['L_max'] for r in soft_lm0])
M_arr = np.array([r['M'] for r in soft_lm0])
L_eq_max_arr = np.array([r['L_eq_max'] for r in soft_lm0])
B_trans_arr = np.array([r['B_trans'] for r in soft_lm0])
L_phi_eq_max_arr = np.array([r['L_phi_eq_max'] for r in soft_lm0])

check('L_max (5-seed mean)', round(L_max_arr.mean(), 2), 0.77,
      tol_abs=0.005, location='§6.1')
check('L_phi_eq (5-seed mean)', round(L_phi_eq_max_arr.mean(), 2), 0.27,
      tol_abs=0.005, location='§6.1')
check('M (5-seed mean)', round(M_arr.mean(), 2), 5.94,
      tol_abs=0.005, location='§6.1')
check('L_eq_max (5-seed mean)', round(L_eq_max_arr.mean(), 2), 2.38,
      tol_abs=0.005, location='§6.1')
check('B_trans (5-seed mean)', round(B_trans_arr.mean(), 2), 2.39,
      tol_abs=0.005, location='§6.1')

print('\n--- Table 5.2: 9-cell Pareto frontier (5-seed means) ---')
expected_frontier = {
    ('soft_only', 0.0):                ('2.38', '5.94', '1.62',  '+0.0087', 0),
    ('soft_only', 0.05):               ('1.02', '0.70', '0.19',  '+0.0046', 2),
    ('soft_only', 0.1):                ('0.96', '0.42', '0.11',  '+0.0028', 3),
    ('projection_every_epoch', 0.0):   ('2.56', '6.79', '1.93',  '+0.0071', 0),
    ('projection_every_epoch', 0.05):  ('1.02', '0.71', '0.19',  '-0.0049', 2),
    ('projection_every_epoch', 0.1):   ('0.92', '0.44', '0.12',  '-0.0073', 5),
    ('projection_every_batch', 0.0):   ('2.59', '6.95', '1.98',  '+0.0066', 0),
    ('projection_every_batch', 0.05):  ('0.99', '0.74', '0.20',  '-0.0089', 2),
    ('projection_every_batch', 0.1):   ('0.91', '0.47', '0.12',  '-0.0112', 5),
}
for key, (e_leq, e_M, e_lpm, e_gap, e_sat) in expected_frontier.items():
    runs = synth['results'][key]
    leq = np.array([r['L_eq_max'] for r in runs]).mean()
    Mv = np.array([r['M'] for r in runs]).mean()
    lpm = np.array([r['L_phi_eq_M'] for r in runs]).mean()
    gap = m15_base - np.array([r['test_mse'] for r in runs]).mean()
    sat = sum(1 for r in runs if r['L_eq_max'] < 1)
    cell_label = f"{key[0]}, lam_M={key[1]}"
    check(f'  L_eq_max', f'{leq:.2f}', e_leq,
          location=f'Table 5.2 cell {cell_label}')
    check(f'  M', f'{Mv:.2f}', e_M,
          location=f'Table 5.2 cell {cell_label}')
    check(f'  L_phi_eq*M', f'{lpm:.2f}', e_lpm,
          location=f'Table 5.2 cell {cell_label}')
    check(f'  gap', f'{gap:+.4f}', e_gap,
          location=f'Table 5.2 cell {cell_label}')
    check(f'  L_eq<1 sat', sat, e_sat,
          location=f'Table 5.2 cell {cell_label}')

print('\n--- §6.3: OLS slope claim across 45 (cell, seed) pairs ---')
xs, ys = [], []
for k, runs in synth['results'].items():
    for r in runs:
        xs.append(r['L_eq_max'])
        ys.append(m15_base - r['test_mse'])
xs = np.array(xs); ys = np.array(ys)
slope, intercept, r_val, p_val, std_err = stats.linregress(xs, ys)
check('OLS slope: gap vs L_eq_max', round(slope, 4), 0.0075,
      tol_abs=5e-4, location='§6.3')
check('OLS slope std err', round(std_err, 4), 0.0011,
      tol_abs=5e-4, location='§6.3')
check('OLS R-squared', round(r_val**2, 2), 0.54,
      tol_abs=0.01, location='§6.3')
check('OLS p < 1e-7', p_val < 1e-7, True, location='§6.3')

print('\n--- §8: cross-experiment table, synthetic columns ---')
# At lam_M=0 (synthetic): M=5.94, L_eq=2.38, gap=+0.0087, sat=0/5
# At lam_M=0.10 most-enforced (proj_every_batch): M=0.47, L_eq=0.91, gap=-0.0112, sat=5/5
proj_batch_lm10 = synth['results'][('projection_every_batch', 0.1)]
M_pb = np.array([r['M'] for r in proj_batch_lm10]).mean()
Leq_pb = np.array([r['L_eq_max'] for r in proj_batch_lm10]).mean()
gap_pb = m15_base - np.array([r['test_mse'] for r in proj_batch_lm10]).mean()
check('§8 V-Dem table: synth M @ lam_M=0', round(M_arr.mean(), 2), 5.94,
      tol_abs=0.005, location='§8 Table')
check('§8 V-Dem table: synth L_eq @ lam_M=0', round(L_eq_max_arr.mean(), 2), 2.38,
      tol_abs=0.005, location='§8 Table')
check('§8 V-Dem table: synth M @ lam_M=0.10 enforced', round(M_pb, 2), 0.47,
      tol_abs=0.005, location='§8 Table')
check('§8 V-Dem table: synth L_eq @ lam_M=0.10 enforced', round(Leq_pb, 2), 0.91,
      tol_abs=0.005, location='§8 Table')
check('§8 V-Dem table: synth gap @ lam_M=0.10 enforced', round(gap_pb, 4), -0.0112,
      tol_abs=1e-4, location='§8 Table')


# ============================================================================
section('V-DEM v1h frontier — section 7.4, 7.6 / section 8 / table 5.3')
# ============================================================================

with open(os.path.join(ARTIFACTS, 'vdem_v1h_frontier.pkl'), 'rb') as f:
    v1h = pickle.load(f)

print('\n--- §7.4: baseline test MSEs ---')
b = v1h['baselines']
check('Random walk MSE', round(b['RW'], 5), 0.00340,
      tol_abs=5e-6, location='§7.4')
check('Pooled linear M1.5 MSE', round(b['pooled_linear_M1.5'], 5), 0.00347,
      tol_abs=5e-6, location='§7.4')
check('Stratified linear M1.5 MSE', round(b['stratified_linear_M1.5'], 5), 0.00337,
      tol_abs=5e-6, location='§7.4')
check('Pooled NN M1.5 MSE (archived single seed-0 run; the submitted version mislabeled it a 5-seed mean, corrected in the camera-ready Sec. 6.2)', round(b['pooled_NN_M1.5'], 5), 0.00352,
      tol_abs=5e-6, location='§7.4')

# Stratified vs pooled linear improvement
strat_improvement_pct = 100 * (b['pooled_linear_M1.5'] - b['stratified_linear_M1.5']) / b['pooled_linear_M1.5']
check('§7.3 diagnostic improvement strat vs pool', round(strat_improvement_pct, 2), 3.02,
      tol_abs=0.005, location='§7.3 / Table 5.3')

print('\n--- §7.6: stability conditions, all 9 audited cells ---')
all_sat = True
all_leq = []
for k, runs in v1h['results'].items():
    if k[0] == 'hard_routing':
        continue
    leq_arr = np.array([r['L_eq_max'] for r in runs])
    cell_sat = (leq_arr < 1).sum()
    if cell_sat < 5:
        all_sat = False
    all_leq.extend(leq_arr.tolist())

check('§7.6 all 9 cells satisfy L_eq < 1 in 5/5 seeds', all_sat, True, location='§7.6')

# Range of L_eq_max across audited cells (cell means)
cell_means_leq = []
for k, runs in v1h['results'].items():
    if k[0] == 'hard_routing': continue
    cell_means_leq.append(np.array([r['L_eq_max'] for r in runs]).mean())
check('§7.6 L_eq_max range, lower (most enforced)',
      round(min(cell_means_leq), 3), 0.130,
      tol_abs=0.005, location='§7.6')
check('§7.6 L_eq_max range, upper (unconstrained soft)',
      round(max(cell_means_leq), 3), 0.292,
      tol_abs=0.005, location='§7.6')

# rho < 0.36 across all audited cells (bound-confirming)
all_rho = []
for k, runs in v1h['results'].items():
    if k[0] == 'hard_routing': continue
    rho_arr = np.array([r['rho'] for r in runs if r.get('rho') is not None])
    all_rho.extend(rho_arr.tolist())
max_rho = max(all_rho) if all_rho else None
check('§7.6 max rho across audited cells <= 0.36',
      max_rho is not None and max_rho <= 0.36, True,
      location='§7.6')

print('\n--- §7.5: hard-routing baseline ---')
hr = v1h['results'][('hard_routing', 0.0)]
hr_mse = np.array([r['test_mse'] for r in hr]).mean()
check('§7.5 hard-routing 5-seed-mean MSE', round(hr_mse, 5), 0.00344,
      tol_abs=5e-6, location='§7.5')

print('\n--- §8: cross-experiment table, V-Dem columns ---')
soft_lm0_v = v1h['results'][('soft_only', 0.0)]
M_v_lm0 = np.array([r['M'] for r in soft_lm0_v]).mean()
Leq_v_lm0 = np.array([r['L_eq_max'] for r in soft_lm0_v]).mean()
check('§8 Table: V-Dem M @ lam_M=0', round(M_v_lm0, 2), 0.87,
      tol_abs=0.005, location='§8 Table')
check('§8 Table: V-Dem L_eq @ lam_M=0', round(Leq_v_lm0, 2), 0.29,
      tol_abs=0.005, location='§8 Table')

# Most enforced V-Dem cell (use proj_every_epoch lam_M=0.10)
pee_lm10 = v1h['results'][('projection_every_epoch', 0.1)]
M_v_pee = np.array([r['M'] for r in pee_lm10]).mean()
Leq_v_pee = np.array([r['L_eq_max'] for r in pee_lm10]).mean()
check('§8 Table: V-Dem M @ lam_M=0.10', round(M_v_pee, 2), 0.05,
      tol_abs=0.005, location='§8 Table')
check('§8 Table: V-Dem L_eq @ lam_M=0.10', round(Leq_v_pee, 2), 0.13,
      tol_abs=0.005, location='§8 Table')


# ============================================================================
section('V-DEM postanalysis — paired DM tests (§7.5, table 5.4)')
# ============================================================================

with open(os.path.join(ARTIFACTS, 'vdem_v1h_postanalysis.pkl'), 'rb') as f:
    pa = pickle.load(f)

target = pa['targets']
regime = pa['regime_assignment']
nn_pool = pa['predictions']['NN_pooled_M1.5']
m2_mean = pa['predictions']['M2_mean']

def paired_dm(pred_a, pred_b, target):
    """Paired DM. positive DM => pred_b is better."""
    err_a = (pred_a - target) ** 2
    err_b = (pred_b - target) ** 2
    diff = err_a - err_b
    n = len(diff)
    DM = diff.mean() / (diff.std(ddof=1) / np.sqrt(n))
    p = 2 * (1 - stats.norm.cdf(abs(DM)))
    return DM, p

print('\n--- §7.5 / Table 5.4: per-regime paired DM, M2 vs pooled NN M1.5 ---')
expected_dm = {
    0: ('+2.20', '0.0281'),
    1: ('+0.30', '0.7647'),
    2: ('+2.41', '0.0161'),
}
for r in [0, 1, 2]:
    mask = regime == r
    DM, p = paired_dm(nn_pool[mask], m2_mean[mask], target[mask])
    e_DM, e_p = expected_dm[r]
    check(f'Regime {r} DM stat', f'{DM:+.2f}', e_DM,
          location=f'Table 5.4 regime {r}')
    check(f'Regime {r} p-value', f'{p:.4f}', e_p,
          location=f'Table 5.4 regime {r}')

# Overall DM
DM_all, p_all = paired_dm(nn_pool, m2_mean, target)
check('§7.5 submitted headline DM (seed-asymmetric per-run test; superseded by Table 2 of the camera-ready)', f'{DM_all:+.2f}', '+2.41',
      location='§7.5')
check('§7.5 headline overall p-value', f'{p_all:.4f}', '0.0159',
      location='§7.5')


# ============================================================================
section('Certified bound audit — appendix D.2')
# ============================================================================

with open(os.path.join(ARTIFACTS, 'certified_audit_results.pkl'), 'rb') as f:
    cert = pickle.load(f)

print('\n--- App D.2: certified spectral-norm bound vs gradient-probe ---')
results = cert['results']
all_cert_ge_probe = all(
    all(c >= p for c, p in zip(r['cert'], r['probe']))
    for r in results
)
check('Cert >= probe in all 9 (seed, regime) cells', all_cert_ge_probe, True,
      location='App D.2')

pmax = np.array([max(r['probe']) for r in results])
cmax = np.array([max(r['cert']) for r in results])
ratio = (cmax / np.maximum(pmax, 1e-9)).mean()
check('Cert/probe ratio mean', round(ratio, 2), 1.07,
      tol_abs=0.02, location='App D.2')


# ============================================================================
section('Prose-level claims (paper §1, §4, §6, §7, §8 narrative)')
# ============================================================================
# These checks cover claims made in the paper's prose that do not appear
# verbatim in any table. They were added after a narrative-table consistency
# audit identified several places where the saved tabular values were
# correct but the surrounding text disagreed with them.

# --- §4 anchor paragraph: V-Dem 5-seed audit constants ---
print('\n--- §4 anchor: V-Dem unconstrained 5-seed audit constants ---')
vdem_unc = v1h['results'][('soft_only', 0.0)]
L_eq_unc = np.mean([s['L_eq_max'] for s in vdem_unc])
B_trans_unc = np.mean([s['B_trans'] for s in vdem_unc])
M_unc = np.mean([s['M'] for s in vdem_unc])
check('§4 V-Dem L_eq_max ~0.29 (5-seed mean)', round(L_eq_unc, 2), 0.29,
      tol_abs=0.005, location='§4')
check('§4 V-Dem B_trans ~0.30 (5-seed mean)', round(B_trans_unc, 2), 0.30,
      tol_abs=0.01, location='§4')
check('§4 V-Dem L_eq_max < 1 (trivial-contraction precondition)',
      L_eq_unc < 1.0, True, location='§4')
check('§4 V-Dem B_trans <= 1 (A3 vacuously satisfied)',
      B_trans_unc <= 1.0, True, location='§4')
check('§6.4 V-Dem M ~0.87 at lam_M=0', round(M_unc, 2), 0.87,
      tol_abs=0.01, location='§6.4')

# --- §6.4 max rho across all 9 audited V-Dem cells <= 0.36 ---
print('\n--- §6.4 / §1 abstract: max rho across 9 audited V-Dem cells ---')
audited_modes = ['soft_only', 'projection_every_epoch', 'projection_every_batch']
audited_lams = [0.0, 0.05, 0.10]
audited_rhos = []
for mode in audited_modes:
    for lam in audited_lams:
        cell = v1h['results'][(mode, lam)]
        for s in cell:
            r = s.get('rho', None)
            if r is not None and not np.isnan(r):
                audited_rhos.append(r)
audited_rhos = np.array(audited_rhos)
check('§6.4 max rho across 9 audited cells (5 seeds each) <= 0.36',
      audited_rhos.max() <= 0.36, True, location='§6.4')
check('§6.4 N audited (cell, seed) rho values = 45',
      len(audited_rhos), 45, location='§6.4')

# --- §6.4 L_eq,max range across audited cells ---
print('\n--- §6.4: L_eq_max range across 9 audited V-Dem cells ---')
all_L_eq = []
for mode in audited_modes:
    for lam in audited_lams:
        cell = v1h['results'][(mode, lam)]
        all_L_eq.append(np.mean([s['L_eq_max'] for s in cell]))
all_L_eq = np.array(all_L_eq)
check('§6.4 L_eq_max range lower (most enforced) ~0.130',
      round(all_L_eq.min(), 3), 0.130, tol_abs=0.005, location='§6.4')
check('§6.4 L_eq_max range upper (unconstrained soft) ~0.292',
      round(all_L_eq.max(), 3), 0.292, tol_abs=0.005, location='§6.4')

# --- §6.4 / §8: 3.3% overall MSE reduction (raw, not from rounded table) ---
print('\n--- §6.4 / §8: M2 vs pool overall MSE reduction (raw) ---')
m2_mse = ((np.asarray(m2_mean) - target) ** 2).mean()
pool_mse = ((np.asarray(nn_pool) - target) ** 2).mean()
pct_reduction = (1 - m2_mse / pool_mse) * 100
check('§6.4 / §8 overall MSE reduction (raw) ~3.3%',
      round(pct_reduction, 1), 3.3, tol_abs=0.05, location='§6.4 / §8')

# --- §6.3: per-regime % MSE reductions (figure subtitles also use these) ---
print('\n--- §6.3 / Figure 1C-E: per-regime % MSE reductions ---')
expected_pct = {0: 7.3, 1: 0.5, 2: 6.3}  # negative-of-figure subtitle, %
for r in [0, 1, 2]:
    mask = regime == r
    m2_r = ((np.asarray(m2_mean)[mask] - target[mask]) ** 2).mean()
    pool_r = ((np.asarray(nn_pool)[mask] - target[mask]) ** 2).mean()
    pct = (1 - m2_r / pool_r) * 100
    check(f'§6.3 regime {r} MSE reduction ~{expected_pct[r]}%',
          round(pct, 1), expected_pct[r], tol_abs=0.05,
          location=f'§6.3 / Fig1{["C","D","E"][r]}')

# --- §6.3: V-Dem regime counts ---
print('\n--- §6.3 / Tables 1, 2: V-Dem test-set regime occupancy ---')
counts = np.bincount(regime).tolist()
check('§6.3 regime 0 count', counts[0], 252, location='Tables 1, 2')
check('§6.3 regime 1 count', counts[1], 621, location='Tables 1, 2')
check('§6.3 regime 2 count', counts[2], 637, location='Tables 1, 2')
check('§6.3 total test N', len(target), 1510, location='Tables 1, 2')

# --- Figure 1 panels C-E final cumulative values ---
print('\n--- Figure 1C-E: cumulative paired-error final values ---')
expected_final = {0: 0.102, 1: 0.015, 2: 0.059}  # +N * mean_diff per regime
for r in [0, 1, 2]:
    mask = regime == r
    pool_sq = (np.asarray(nn_pool)[mask] - target[mask]) ** 2
    m2_sq = (np.asarray(m2_mean)[mask] - target[mask]) ** 2
    cum_final = (pool_sq - m2_sq).sum()
    check(f'Figure 1{["C","D","E"][r]} cumulative final ~+{expected_final[r]}',
          round(cum_final, 3), expected_final[r], tol_abs=0.001,
          location=f'Figure 1{["C","D","E"][r]}')

# --- §7 / App B.1: synthetic DGP regime occupancy and alpha ---
print('\n--- §7 / App B.1: synthetic DGP regime structure ---')
rng = np.random.default_rng(42)
U_dgp, T_dgp = 30, 50
betas_dgp = rng.normal(0, 0.5, U_dgp)
Y_dgp = np.zeros((U_dgp, T_dgp))
X_dgp = np.zeros((U_dgp, T_dgp))
Y_dgp[:, 0] = rng.normal(0, 0.3, U_dgp)
X_dgp[:, 0] = rng.normal(0, 0.3, U_dgp)
nu_dgp = rng.normal(0, 0.5, (U_dgp, T_dgp))
eps_dgp = rng.normal(0, 0.3, (U_dgp, T_dgp))
rho_Y_dgp = (0.5, 0.8, 0.7)
beta_X_dgp = (0.05, 0.20, 0.40)

def regime_dgp(y):
    if y < -0.6:
        return 0
    if y > 0.6:
        return 2
    return 1

for u in range(U_dgp):
    for t in range(1, T_dgp):
        X_dgp[u, t] = 0.9 * X_dgp[u, t-1] + nu_dgp[u, t]
        r = regime_dgp(Y_dgp[u, t-1])
        Y_dgp[u, t] = (betas_dgp[u] + rho_Y_dgp[r] * Y_dgp[u, t-1]
                       + beta_X_dgp[r] * X_dgp[u, t-1] + eps_dgp[u, t])

T_train = T_dgp - 10
regimes_dgp = np.array([[regime_dgp(Y_dgp[u, t]) for t in range(T_dgp)]
                        for u in range(U_dgp)])
train_regs = regimes_dgp[:, :T_train]
occ = np.bincount(train_regs.flatten()) / train_regs.size * 100
check('App B.1 synth train regime 0 occupancy ~28.2%',
      round(occ[0], 1), 28.2, tol_abs=0.05, location='App B.1')
check('App B.1 synth train regime 1 occupancy ~37.5%',
      round(occ[1], 1), 37.5, tol_abs=0.05, location='App B.1')
check('App B.1 synth train regime 2 occupancy ~34.2%',
      round(occ[2], 1), 34.2, tol_abs=0.05, location='App B.1')

trans_train = sum(regimes_dgp[u, t] != regimes_dgp[u, t-1]
                  for u in range(U_dgp) for t in range(1, T_train))
alpha_train = trans_train / (U_dgp * (T_train - 1))
trans_full = sum(regimes_dgp[u, t] != regimes_dgp[u, t-1]
                 for u in range(U_dgp) for t in range(1, T_dgp))
alpha_full = trans_full / (U_dgp * (T_dgp - 1))
check('§7 / App B.1 synth alpha_train ~0.217',
      round(alpha_train, 3), 0.217, tol_abs=0.001, location='App B.1')
check('§7 synth alpha (full panel) ~0.20',
      round(alpha_full, 2), 0.20, tol_abs=0.005, location='§7')

# --- §7: synthetic L_phi_eq * M = 1.6 at soft penalty lam_M=0 ---
print('\n--- §7: synthetic gating-correction term L_phi_eq * M ~1.6 ---')
synth_unc = synth['results'][('soft_only', 0.0)]
LpM = np.mean([s['L_phi_eq_M'] for s in synth_unc])
check('§7 synth L_phi_eq * M ~1.6 at soft, lam_M=0',
      round(LpM, 1), 1.6, tol_abs=0.05, location='§7')

# --- App D.2 table: per-regime probe and cert means/stds ---
print('\n--- App D.2 table: per-regime probe and cert (3 seeds) ---')
probes = np.array([r['probe'] for r in results])  # shape (3, 3)
certs = np.array([r['cert'] for r in results])
expected = [
    # (probe_mean, probe_std, cert_mean, cert_std)
    (0.129, 0.011, 0.136, 0.014),
    (0.152, 0.022, 0.157, 0.023),
    (0.024, 0.015, 0.030, 0.012),
]
for r_idx, (pm, ps, cm, cs) in enumerate(expected):
    check(f'App D.2 R{r_idx} probe mean',
          round(probes[:, r_idx].mean(), 3), pm, tol_abs=0.001,
          location='App D.2')
    check(f'App D.2 R{r_idx} probe std',
          round(probes[:, r_idx].std(), 3), ps, tol_abs=0.001,
          location='App D.2')
    check(f'App D.2 R{r_idx} cert mean',
          round(certs[:, r_idx].mean(), 3), cm, tol_abs=0.001,
          location='App D.2')
    check(f'App D.2 R{r_idx} cert std',
          round(certs[:, r_idx].std(), 3), cs, tol_abs=0.001,
          location='App D.2')

# Max-over-regime aggregate
pmax_seeds = np.array([max(r['probe']) for r in results])
cmax_seeds = np.array([max(r['cert']) for r in results])
check('App D.2 max-over-regimes probe mean ~0.155',
      round(pmax_seeds.mean(), 3), 0.155, tol_abs=0.001, location='App D.2')
check('App D.2 max-over-regimes cert mean ~0.164',
      round(cmax_seeds.mean(), 3), 0.164, tol_abs=0.001, location='App D.2')

# --- §6.1 / Figure 1 panel B: party-org coefficients at lag t-4 ---
print('\n--- §6.1 / Figure 1B: party-organization coefficients at lag t-4 ---')
# Reproduce figure builder's regression layout
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'data', 'merged_completevars_balanced_zero_missing.csv')
try:
    import pandas as pd
    from sklearn.linear_model import LinearRegression
except ImportError:
    print('  [SKIP] pandas/sklearn not installed; skipping coefficient checks')
else:
    Y_VAR_FIG = 'v2xel_frefair'
    X_VARS_FIG = ['v2csantimv', 'v2jureform', 'v2psorgs', 'v2mecorrpt']
    L_LAG_FIG = 5
    TEST_YR_FIG = 10
    df_fig = (pd.read_csv(DATA_PATH)
              .sort_values(['country_id', 'year']).reset_index(drop=True))

    Y_h_list, X_h_list, t_list, y1_list = [], [], [], []
    for cid, g in df_fig.groupby('country_id'):
        g = g.sort_values('year').reset_index(drop=True)
        T = len(g)
        if T < L_LAG_FIG + TEST_YR_FIG + 1:
            continue
        for t_idx in range(L_LAG_FIG, T):
            Y_h_list.append(g[Y_VAR_FIG].iloc[t_idx-L_LAG_FIG:t_idx].values)
            X_h_list.append(g[X_VARS_FIG].iloc[t_idx-L_LAG_FIG:t_idx].values)
            t_list.append(float(g[Y_VAR_FIG].iloc[t_idx]
                                - g[Y_VAR_FIG].iloc[t_idx-1]))
            y1_list.append(float(g[Y_VAR_FIG].iloc[t_idx-1]))
    Y_h_arr = np.asarray(Y_h_list, dtype=np.float64)
    X_h_arr = np.stack(X_h_list).astype(np.float64)
    t_arr = np.asarray(t_list, dtype=np.float64)
    y1_arr = np.asarray(y1_list, dtype=np.float64)

    # Train/test mask matches build_figure.py
    N_total = len(t_arr)
    is_test = np.zeros(N_total, dtype=bool)
    ptr = 0
    for cid, g in df_fig.groupby('country_id'):
        T = len(g)
        if T < L_LAG_FIG + TEST_YR_FIG + 1:
            continue
        n_added = T - L_LAG_FIG
        is_test[ptr + n_added - TEST_YR_FIG: ptr + n_added] = True
        ptr += n_added

    Y_h_tr = Y_h_arr[~is_test]
    X_h_tr = X_h_arr[~is_test]
    t_tr = t_arr[~is_test]
    y1_tr = y1_arr[~is_test]
    b1, b2 = np.percentile(y1_tr, 100/3), np.percentile(y1_tr, 200/3)
    reg_tr = np.where(y1_tr < b1, 0, np.where(y1_tr < b2, 1, 2))

    X_mat = np.concatenate(
        [Y_h_tr, X_h_tr.reshape(len(X_h_tr), -1)], axis=1)
    party_idx_in_X = X_VARS_FIG.index('v2psorgs')  # = 2
    LAG_T_MINUS_4 = 1  # index 1 in 5-vector: 0=t-5, 1=t-4, ..., 4=t-1

    expected_party = {0: -0.019, 1: +0.018, 2: +0.016}  # at lag t-4
    for r in [0, 1, 2]:
        lr = LinearRegression().fit(X_mat[reg_tr == r], t_tr[reg_tr == r])
        coef_X_r = lr.coef_[L_LAG_FIG:].reshape(L_LAG_FIG, len(X_VARS_FIG))
        c = coef_X_r[LAG_T_MINUS_4, party_idx_in_X]
        check(f'§6.1 / Fig1B party-org R{r} coef at t-4 ~{expected_party[r]:+.3f}',
              round(c, 3), expected_party[r], tol_abs=0.0005,
              location='§6.1 / Figure 1B')

    pooled_lr = LinearRegression().fit(X_mat, t_tr)
    coef_X_pool = pooled_lr.coef_[L_LAG_FIG:].reshape(
        L_LAG_FIG, len(X_VARS_FIG))
    c_pool = coef_X_pool[LAG_T_MINUS_4, party_idx_in_X]
    check('§6.1 / Fig1B party-org pooled coef at t-4 ~-0.000',
          round(c_pool, 3), 0.000, tol_abs=0.0005,
          location='§6.1 / Figure 1B')

    # 4-of-4 sign-reversal claim: at least one lag in {t-2,...,t-5} where
    # the three regime coefficients do not all share the same sign.
    print('\n--- Abstract / §6.1: 4-of-4 covariates show sign reversal ---')
    n_with_reversal = 0
    for j, xv in enumerate(X_VARS_FIG):
        had_reversal = False
        for lag_idx in range(L_LAG_FIG):  # 0=t-5, 1=t-4, 2=t-3, 3=t-2, 4=t-1
            signs = set()
            for r in [0, 1, 2]:
                lr_r = LinearRegression().fit(
                    X_mat[reg_tr == r], t_tr[reg_tr == r])
                cr = lr_r.coef_[L_LAG_FIG:].reshape(
                    L_LAG_FIG, len(X_VARS_FIG))[lag_idx, j]
                if abs(cr) > 1e-4:
                    signs.add(int(np.sign(cr)))
            if len(signs) > 1:
                had_reversal = True
                break
        if had_reversal:
            n_with_reversal += 1
    check('Abstract / §6.1 N covariates with sign reversal at some lag = 4',
          n_with_reversal, 4, location='Abstract / §6.1')

# --- §6.2 / Table 1: data-prep sanity checks ---
print('\n--- §6.2 / Table 1: V-Dem panel sizing ---')
try:
    import pandas as _pd_only
except ImportError:
    print('  [SKIP] pandas not installed; skipping panel-sizing checks')
else:
    df_sz = _pd_only.read_csv(DATA_PATH)
    check('§6.2 V-Dem unique countries = 151',
          int(df_sz['country_id'].nunique()), 151, location='§6.2')
    check('§6.2 V-Dem year range start = 1970',
          int(df_sz['year'].min()), 1970, location='§6.2')
    check('§6.2 V-Dem year range end = 2025',
          int(df_sz['year'].max()), 2025, location='§6.2')
    # Lag window L=5 drops 5 obs/country; train+test = total - 151*5
    expected_tt = len(df_sz) - 151 * 5
    check('§6.2 train+test size after lag drop = 7,701',
          expected_tt, 6191 + 1510, location='§6.2')


# ============================================================================
section('FINAL VERDICT')
# ============================================================================
print()
print(f'Total checks:  {n_pass + n_fail}')
print(f'Passed:        {n_pass}')
print(f'Failed:        {n_fail}')
print()
if n_fail == 0:
    print('All paper numbers reproduce from the archived artifacts.')
    sys.exit(0)
else:
    print(f'{n_fail} numerical disagreement(s) — see [FAIL] entries above.')
    sys.exit(1)
