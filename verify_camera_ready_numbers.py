"""
verify_camera_ready_numbers.py

Companion to verify_paper_numbers.py for the NeurIPS 2026 camera-ready
version of "Stability-Constrained Regime-Aware Forecasting for Heterogeneous
Panel Time Series" (submission 18877).

verify_paper_numbers.py checks the numbers of the submitted version against
the four original artifact pickles. This script checks every number ADDED or
CHANGED in the camera-ready (rebuttal analyses, sensitivity study,
realized-volatility panel, corrected Lemma 2 constants, relabeled Figure 1)
against the rebuttal artifacts in artifacts/rebuttal_R1, artifacts/rebuttal_S2,
artifacts/rv_screen_S1, artifacts/rv_f1 and the original frontier /
postanalysis pickles.

Run from the package root:
    python verify_camera_ready_numbers.py

Expected runtime: a few seconds (no training; CSV/JSON/pickle arithmetic).
Every check names the section, table, or figure of the camera-ready where the
number appears.
"""
import csv
import json
import math
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(ROOT, 'artifacts')
R1 = os.path.join(ART, 'rebuttal_R1')
S2 = os.path.join(ART, 'rebuttal_S2')
RVS1 = os.path.join(ART, 'rv_screen_S1')
RVF1 = os.path.join(ART, 'rv_f1')

n_pass = 0
n_fail = 0


def check(name, computed, expected, tol_abs=0, tol_rel=0, location=''):
    global n_pass, n_fail
    if isinstance(computed, (bool, np.bool_)) or isinstance(expected, bool):
        ok = (bool(computed) == bool(expected))
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
    if ok:
        n_pass += 1
    else:
        n_fail += 1
    loc = f' [{location}]' if location else ''
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{loc}: computed={computed!r}, expected={expected!r}")
    return ok


def section(title):
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


def rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def fl(x):
    return float(x) if x not in ('', None) else float('nan')


# ============================================================================
section('R1: panel-appropriate inference on V-Dem (Table 2, Sec. 6.2)')
# ============================================================================
dm = rows(os.path.join(R1, 'results_18877_R1_dm_tests.csv'))
a3 = rows(os.path.join(S2, 'results_S2_A3_stratification_tests.csv'))


def row(table, **kw):
    out = [r for r in table if all(r[k] == v for k, v in kw.items())]
    assert len(out) == 1, (kw, len(out))
    return out[0]


r = row(dm, family='C2_M2_vs_rw', scope='overall')
check('Table 2 row 1: mean d x1e4', round(fl(r['mean_d']) * 1e4, 3), -0.056, tol_abs=5e-4, location='Table 2')
check('Table 2 row 1: cluster t', round(fl(r['t_cluster']), 3), -0.077, tol_abs=5e-4, location='Table 2')
check('Table 2 row 1: cluster p', round(fl(r['p_cluster']), 3), 0.938, tol_abs=5e-4, location='Table 2')
check('Table 2 row 1: bootstrap p', round(fl(r['p_boot']), 3), 0.940, tol_abs=5e-4, location='Table 2')

r = row(dm, family='C1_M2_vs_stratlin', scope='overall')
check('Table 2 row 2: mean d x1e4', round(fl(r['mean_d']) * 1e4, 3), -0.343, tol_abs=5e-4, location='Table 2')
check('Table 2 row 2: cluster t', round(fl(r['t_cluster']), 3), -0.413, tol_abs=5e-4, location='Table 2')
check('Table 2 row 2: cluster p', round(fl(r['p_cluster']), 3), 0.680, tol_abs=5e-4, location='Table 2')
check('Table 2 row 2: bootstrap p', round(fl(r['p_boot']), 3), 0.690, tol_abs=5e-4, location='Table 2')

r = row(dm, family='C3_stratlin_vs_rw', scope='overall')
check('Table 2 row 3: mean d x1e4', round(fl(r['mean_d']) * 1e4, 3), 0.287, tol_abs=5e-4, location='Table 2')
check('Table 2 row 3: cluster t', round(fl(r['t_cluster']), 3), 0.215, tol_abs=5e-4, location='Table 2')
check('Table 2 row 3: cluster p', round(fl(r['p_cluster']), 3), 0.830, tol_abs=5e-4, location='Table 2, Sec. 6.2')
check('Table 2 row 3: bootstrap p', round(fl(r['p_boot']), 3), 0.828, tol_abs=5e-4, location='Table 2')

r = row(a3, scope='overall')
check('Table 2 row 4: mean d x1e4 (stratified vs pooled linear)', round(fl(r['mean_d']) * 1e4, 3), 1.050, tol_abs=5e-4, location='Table 2')
check('Table 2 row 4: bootstrap p', round(fl(r['p_boot']), 3), 0.273, tol_abs=5e-4, location='Table 2, Sec. 6.1, Sec. 6.4')
strat_gain = 100 * (fl(r['mse_pooled']) - fl(r['mse_stratified'])) / fl(r['mse_pooled'])
check('Sec. 6.1: stratified-vs-pooled linear gain +3.02%', round(strat_gain, 2), 3.02, tol_abs=0.005, location='Sec. 6.1')

r = row(dm, family='C1_M2_vs_stratlin', scope='regime2')
check('Table 2 caption: regime-2 uncorrected cluster p = 0.051', round(fl(r['p_cluster']), 3), 0.051, tol_abs=5e-4, location='Table 2 caption')
check('Table 2 caption: regime-2 bootstrap p = 0.202', round(fl(r['p_boot']), 3), 0.202, tol_abs=5e-4, location='Table 2 caption')
check('Table 2 caption: regime-2 Holm p = 0.606', round(fl(r['p_holm']), 3), 0.606, tol_abs=5e-4, location='Table 2 caption')
per_regime_null = all(fl(x['p_holm']) > 0.05 for x in dm
                      if x['family'] in ('C1_M2_vs_stratlin', 'C2_M2_vs_rw') and x['scope'] != 'overall')
check('Table 2 caption: no per-regime M2 test significant after Holm', per_regime_null, True, location='Table 2 caption')
check('Table 2: N = 1,510', int(row(dm, family='C2_M2_vs_rw', scope='overall')['N_obs']), 1510, location='Table 2')
check('Table 2: 151 country clusters', int(row(dm, family='C2_M2_vs_rw', scope='overall')['n_countries']), 151, location='Table 2')

print('\n--- Submitted seed-asymmetric test at three decimals (Fig. 1 C-E, Sec. 6.2) ---')
prec = rows(os.path.join(R1, 'results_18877_R1_table2_precision.csv'))
for scope, e_dm, e_p in [('overall', 2.411, 0.016), ('regime0', 2.196, 0.028),
                         ('regime1', 0.299, 0.765), ('regime2', 2.407, 0.016)]:
    r = row(prec, scope=scope)
    check(f'{scope}: DM at 3 dp', round(fl(r['dm_stat_full']), 3), e_dm, tol_abs=5e-4, location='Fig. 1 / Sec. 6.2')
    check(f'{scope}: p at 3 dp', round(fl(r['p_full']), 3), e_p, tol_abs=5e-4, location='Fig. 1 / Sec. 6.2')
check('Sec. 6.2: submitted headline DM +2.41, p = 0.016', f"{fl(row(prec, scope='overall')['dm_stat_full']):+.2f}", '+2.41', location='Sec. 6.2')

print('\n--- Step 0 reconciliation (all printed submission numbers recompute) ---')
rec = rows(os.path.join(R1, 'results_18877_R1_step0_reconciliation.csv'))
mism = [x for x in rec if x['match'] not in ('True', '')]
check('Step 0: zero mismatches between printed and recomputed values', len(mism), 0, location='Sec. 6.2')

# ============================================================================
section('S2: sensitivity to the regime construction (Sec. 6.4, Appendix)')
# ============================================================================
grid = rows(os.path.join(S2, 'results_S2_A1_diagnostic_grid.csv'))
variants = sorted(set(x['variant'] for x in grid))
check('24 diagnostic variants', len(variants), 24, location='Sec. 6.4')
check('PC1 state summary present in the grid', any(v.startswith('psi3_pc1') for v in variants), True, location='Sec. 6.4')
check('Variants include R in {2,3,4,5} and +/-5pp cut shifts',
      all(any(f'_R{R}' in v for v in variants) for R in (2, 3, 4, 5)) and any('cut' in v for v in variants), True,
      location='Sec. 6.4')


def betas(x):
    out = []
    for k in ('beta_R0', 'beta_R1', 'beta_R2', 'beta_R3', 'beta_R4'):
        if x[k] not in ('', None):
            out.append(fl(x[k]))
    return out


def ses(x):
    out = []
    for k in ('se_R0', 'se_R1', 'se_R2', 'se_R3', 'se_R4'):
        if x[k] not in ('', None):
            out.append(fl(x[k]))
    return out


mech = neg = sig = 0
for v in variants:
    x = row(grid, variant=v, covariate='v2psorgs', lag='4')
    b = betas(x)
    straddle = min(b) < 0 < max(b)
    collapse = abs(fl(x['pooled'])) < max(abs(t) for t in b)
    mech += int(straddle and collapse)
    neg += int(b[0] < 0)
    sig += int(abs(b[0]) > 2 * ses(x)[0])
check('Cancellation mechanism (pooled collapses, regimes straddle 0) in 24/24 variants', mech, 24, location='Sec. 6.4')
check('Low-stratum party-org t-4 coefficient negative in 24/24', neg, 24, location='Sec. 6.4')
check('Low-stratum party-org t-4 |beta| > 2 se in 13/24', sig, 13, location='Sec. 6.4')
rev = sum(1 for v in variants if any(x['reversal_outer'] == 'True' for x in grid if x['variant'] == v))
check('Significance-filtered outer reversal in 0/24 variants', rev, 0, location='Sec. 6.4')
canon = [x for x in grid if x['variant'] == 'psi1_ylag1_R3']
covs = sorted(set(x['covariate'] for x in canon))
n_strad = sum(1 for c in covs if any(min(betas(x)) < 0 < max(betas(x)) for x in canon if x['covariate'] == c))
check('Canonical construction: all four covariates straddle 0 at one or more lag', n_strad, 4, location='Sec. 6.4 / Appendix')

stab = rows(os.path.join(S2, 'results_S2_A2_stability.csv'))
r = row(stab, check='jackknife', sample='v2psorgs_lag4')
check('Jackknife: low-stratum sign retained in 94.5% of 200', round(100 * fl(r['frac_sign_R0']), 1), 94.5, tol_abs=0.05, location='Sec. 6.4')
check('Jackknife: n = 200 half-country subsamples', int(r['n']), 200, location='Sec. 6.4')
r = row(stab, check='jackknife', sample='psorgs_l4_full_pattern')
check('Jackknife: complete symmetric pattern retained in 1.5%', round(100 * fl(r['frac_pattern']), 1), 1.5, tol_abs=0.05, location='Sec. 6.4')
eras = {x['sample']: fl(x['psorgs_l4_R0']) for x in stab if x['check'] == 'era'}
negs = [k for k, v in eras.items() if v < 0]
check('Era windows: low-stratum coefficient negative in 4 of 5', len(negs), 4, location='Sec. 6.4 / Appendix')
check('Era windows: negative range lower bound -0.024', round(min(eras.values()), 3), -0.024, tol_abs=5e-4, location='Appendix')
check('Era windows: negative range upper bound -0.015', round(max(v for v in eras.values() if v < 0), 3), -0.015, tol_abs=5e-4, location='Appendix')
check('Era windows: most recent half of training ~ +0.001', round(eras['train_second_half'], 3), 0.001, tol_abs=5e-4, location='Sec. 6.4 / Appendix')
loco = [x for x in stab if x['check'] == 'loco']
check('Leave-one-covariate-out: 4 refits, no significance-filtered reversal in any', len(loco) == 4 and all(fl(x['n_reversals']) == 0 for x in loco), True, location='Appendix')

b = rows(os.path.join(S2, 'results_S2_B_m2_variants.csv'))
rw = [fl(x['p_boot']) for x in b if x['family'] == 'M2_vs_rw']
sl = [fl(x['p_holm']) for x in b if x['family'] == 'M2_vs_stratlin']
check('M2 retrained under 4 variants', len(rw), 4, location='Sec. 6.4')
check('M2 vs RW: min bootstrap p rounds to >= 0.80', round(min(rw), 2) >= 0.80, True, location='Sec. 6.4')
check('M2 vs stratified linear: min Holm p rounds to >= 0.16', round(min(sl), 2) >= 0.16, True, location='Sec. 6.4')
with open(os.path.join(S2, 'results_S2_metaA.json')) as f:
    metaA = json.load(f)
check('Design: 24 variants in meta', int(metaA['n_variants']), 24, location='Sec. 6.4')
check('Design: 200 jackknife draws in meta', int(metaA['n_jk']), 200, location='Sec. 6.4')
check('Design: B = 9,999 bootstrap draws', int(metaA['B']), 9999, location='Sec. 6.4')

print('\n--- Eight-configuration stratified-linear table (independent reimplementation, s2_grid.csv) ---')
grid8 = rows(os.path.join(S2, 's2_grid.csv'))
tab = {
    'R3 tertiles (SUBMITTED)': (3.02, [-.028, .027, .024]),
    'R2 median': (-0.19, [-.021, .040]),
    'R3 quartile cuts (25/75)': (3.84, [-.027, .030, -.030]),
    'R3 quintile cuts (20/80)': (1.30, [-.020, .009, -.005]),
    'R4 quartiles': (2.41, [-.027, -.005, .063, -.030]),
    'R5 quintiles': (-0.54, None),
    'psi = Y_{t-2}, tertiles': (0.33, [-.030, .035, .014]),
    'psi = MA3, tertiles': (0.07, [-.025, .021, .020]),
}
check('Eight configurations present', len(grid8), 8, location='Appendix (sensitivity table)')
for x in grid8:
    e_gain, e_co = tab[x['config']]
    co = json.loads(x['headline_coefs'])
    check(f"{x['config']}: gain vs pooled {e_gain:+.2f}%", round(fl(x['gain_pct']), 2), e_gain, tol_abs=5e-3, location='Appendix (sensitivity table)')
    if e_co is not None:
        for j, (c, e) in enumerate(zip(co, e_co)):
            check(f"{x['config']}: party-org t-4 regime {j} (std.) {e:+.3f}", round(c, 3), e, tol_abs=5e-4, location='Appendix (sensitivity table)')
    check(f"{x['config']}: regime coefficients straddle 0", min(co) < 0 < max(co), True, location='Appendix (sensitivity table)')
_b = json.loads(row(grid8, config='R3 tertiles (SUBMITTED)')['bounds'])
check('Canonical tertile cut points (0.168, 0.674) in s2_grid.csv',
      f"{_b[0]:.3f}, {_b[1]:.3f}", '0.168, 0.674', location='Appendix')

# ============================================================================
section('RV-S1: realized-volatility screen (Sec. 8, Appendix)')
# ============================================================================
g1 = rows(os.path.join(RVS1, 'results_RVS1_gate1_headroom.csv'))
with open(os.path.join(RVS1, 'results_RVS1_meta.json')) as f:
    ms1 = json.load(f)
r = row(g1, comparison='pooled_lin_vs_rw')
headroom = 100 * (fl(r['mse_base']) - fl(r['mse_model'])) / fl(r['mse_base'])
check('Pooled linear beats RW by 28.2%', round(headroom, 1), 28.2, tol_abs=0.05, location='Sec. 8')
check('RW MSE 0.1039', round(fl(r['mse_base']), 4), 0.1039, tol_abs=5e-5, location='Sec. 8 / Appendix')
check('Pooled linear MSE 0.0746', round(fl(r['mse_model']), 4), 0.0746, tol_abs=5e-5, location='Sec. 8 / Appendix')
check('Gate 1 bootstrap p at the 1e-4 resolution floor', fl(r['p_boot']), 0.0001, tol_abs=1e-9, location='Sec. 8')
check('N test observations 4,184', int(r['n_obs']), 4184, location='Sec. 8')
check('Test days 523', int(r['n_days']), 523, location='Sec. 8')
check('Trading days 2,615', int(ms1['T_days']), 2615, location='Sec. 8 / Appendix')
check('Usable days 2,611', int(ms1['usable_days']), 2611, location='Appendix')
check('Eight equity indices', len(ms1['assets']), 8, location='Sec. 8')
check('Block length 25', int(ms1['block_len']), 25, location='Sec. 8 / Appendix')
check('B = 9,999', int(ms1['B']), 9999, location='Sec. 8 / Appendix')
check('p = 4 lags', int(ms1['p_lags']), 4, location='Appendix')
check('Wald W = 59.9', round(float(ms1['wald_W']), 1), 59.9, tol_abs=0.05, location='Sec. 8')
check('Wald p = 5.4e-7', round(float(ms1['wald_p']) * 1e7, 1), 5.4, tol_abs=0.05, location='Sec. 8')
g2 = rows(os.path.join(RVS1, 'results_RVS1_gate2_diagnostic.csv'))
check('0 of 8 slopes change sign across regimes', sum(1 for x in g2 if x['sign_reversal_R0_R2'] == 'True'), 0, location='Sec. 8')
check('8 slopes tested (4 own + 4 spillover lags)', len(g2), 8, location='Sec. 8 / Appendix')

# ============================================================================
section('RV-F1: certificate audit and predictive results (Sec. 8, Table 4)')
# ============================================================================
au = [x for x in rows(os.path.join(RVF1, 'results_RVF1_audit_constants.csv')) if x['panel'] == 'RV']
with open(os.path.join(RVF1, 'results_RVF1_meta.json')) as f:
    mf1 = json.load(f)
expected = {  # seed: (L_max, L_phi_eq*M, L_eq_max, corr, B_trans, certified)
    0: (0.570, 0.499, 1.067, 1.068, 1.085, False),
    1: (0.578, 0.676, 1.252, 1.254, 1.277, False),
    2: (0.577, 0.534, 1.110, 1.110, 1.128, False),
    3: (0.569, 0.560, 1.120, 1.130, 1.148, False),
    4: (0.570, 0.426, 0.984, 0.996, 1.010, True),
}
lpm_all, leq_all = [], []
for x in au:
    s = int(x['seed'])
    Lmax, M, lpe, leq, bt = fl(x['L_max']), fl(x['M']), fl(x['L_phi_eq_max']), fl(x['L_eq_max']), fl(x['B_trans'])
    lpm = lpe * M
    corr = Lmax + lpm
    e = expected[s]
    print(f'\n--- Table 4, seed {s} ---')
    check('max_r L_Y', round(Lmax, 3), e[0], tol_abs=5e-4, location='Table 4')
    check('L_phi,eq * M', round(lpm, 3), e[1], tol_abs=5e-4, location='Table 4')
    # The audit CSV stores constants at 4 dp, so a derived or half-way value can
    # sit exactly on a 3-dp rounding boundary (seed 1 B_trans = 1.2765; seed 3
    # Corr. = 0.5694 + 0.1362 * 4.112). Compare unrounded within the storage
    # precision (6e-4 for sums of two 4-dp inputs) rather than after rounding.
    check('L_eq,max', round(leq, 3), e[2], tol_abs=5e-4, location='Table 4')
    check('Corr. (eps = 1 substitution)', corr, e[3], tol_abs=6e-4, location='Table 4')
    check('B_trans', bt, e[4], tol_abs=5.1e-4, location='Table 4')
    check('Certified (L_eq,max < 1)', leq < 1, e[5], location='Table 4')
    lpm_all.append(lpm)
    leq_all.append(leq)
print('\n--- Sec. 8 ranges ---')
check('Per-regime L_Y range 0.55-0.58 (min)', round(min(fl(v) for x in au for v in json.loads(x['L_Y'])), 2), 0.55, tol_abs=5e-3, location='Sec. 8')
check('Per-regime L_Y range 0.55-0.58 (max)', round(max(fl(v) for x in au for v in json.loads(x['L_Y'])), 2), 0.58, tol_abs=5e-3, location='Sec. 8')
check('Gating term range 0.43-0.68 (min)', round(min(lpm_all), 2), 0.43, tol_abs=5e-3, location='Sec. 8')
check('Gating term range 0.43-0.68 (max)', round(max(lpm_all), 2), 0.68, tol_abs=5e-3, location='Sec. 8')
unc = sorted(l for l in leq_all if l >= 1)
check('4 of 5 seeds uncertified', len(unc), 4, location='Sec. 8')
check('Uncertified L_eq,max range 1.07-1.25 (min)', round(unc[0], 2), 1.07, tol_abs=5e-3, location='Sec. 8')
check('Uncertified L_eq,max range 1.07-1.25 (max)', round(unc[-1], 2), 1.25, tol_abs=5e-3, location='Sec. 8')
s4 = row(au, seed='4')
check('Seed 4: alpha_emp = 0.271', round(fl(s4['alpha_emp']), 3), 0.271, tol_abs=5e-4, location='Sec. 8 / Table 4')
check('Seed 4: alpha* = 0.619', round(fl(s4['alpha_star']), 3), 0.619, tol_abs=5e-4, location='Sec. 8 / Table 4')
check('Seed 4: rho = 0.991', round(fl(s4['rho']), 3), 0.991, tol_abs=5e-4, location='Sec. 8 / Table 4')
leq_c = fl(s4['L_max']) + fl(s4['L_phi_eq_max']) * fl(s4['M'])
bt4 = fl(s4['B_trans'])
a_star_c = -math.log(leq_c) / (math.log(bt4) - math.log(leq_c))
rho_c = leq_c ** (1 - fl(s4['alpha_emp'])) * bt4 ** fl(s4['alpha_emp'])
check('Seed 4 corrected: L_eq,max = 0.996', round(leq_c, 3), 0.996, tol_abs=5e-4, location='Sec. 8')
check('Seed 4 corrected: alpha* = 0.305 > 0.271', round(a_star_c, 3), 0.305, tol_abs=5e-4, location='Sec. 8')
check('Seed 4 corrected: rho = 0.9995', round(rho_c, 4), 0.9995, tol_abs=5e-5, location='Sec. 8')
vref = [x for x in rows(os.path.join(RVF1, 'results_RVF1_audit_constants.csv')) if x['panel'].startswith('VDem_ref')]
check('Table 4 caption: V-Dem unconstrained reference L_eq,max = 0.292', round(fl(vref[0]['L_eq_max']), 3), 0.292, tol_abs=5e-4, location='Table 4 caption')

print('\n--- Sec. 8 predictive results ---')
dmr = rows(os.path.join(RVF1, 'results_RVF1_dm_tests.csv'))
r = row(dmr, family='F1_M2_vs_B2interaction', scope='overall')
check('M2 vs interaction linear: t = -3.82', round(fl(r['t_boot']), 2), -3.82, tol_abs=5e-3, location='Sec. 8')
check('M2 vs interaction linear: p = 0.0014', round(fl(r['p_boot']), 4), 0.0014, tol_abs=5e-5, location='Sec. 8')
r0 = row(dmr, family='F1_M2_vs_B2interaction', scope='regime0')
check('Calm regime: t = -4.85', round(fl(r0['t_boot']), 2), -4.85, tol_abs=5e-3, location='Sec. 8')
check('Calm regime: Holm p = 0.0012', round(fl(r0['p_holm']), 4), 0.0012, tol_abs=5e-5, location='Sec. 8')
check('Calm regime: 5/5 seeds agree', r0['per_seed_t'].strip().endswith('[5/5]'), True, location='Sec. 8')
check('Calm regime share of test observations ~70%', round(100 * int(r0['N']) / 4184), 70, location='Sec. 8')
r = row(dmr, family='F2_M2_vs_B1pooled', scope='overall')
check('M2 vs pooled linear: t = -3.57', round(fl(r['t_boot']), 2), -3.57, tol_abs=5e-3, location='Sec. 8')
check('M2 vs pooled linear: p = 0.0024', round(fl(r['p_boot']), 4), 0.0024, tol_abs=5e-5, location='Sec. 8')
f3 = [x for x in dmr if x['family'].startswith('F3') and x['scope'] == 'overall']
check('Interaction vs pooled linear: t = -0.06', round(fl(f3[0]['t_boot']), 2), -0.06, tol_abs=5e-3, location='Sec. 8')
check('Interaction vs pooled linear: p = 0.95', round(fl(f3[0]['p_boot']), 2), 0.95, tol_abs=5e-3, location='Sec. 8')
f4 = [x for x in dmr if x['family'].startswith('F4') and x['scope'] == 'overall']
check('M2 vs RW: t = +8.57', round(fl(f4[0]['t_boot']), 2), 8.57, tol_abs=5e-3, location='Sec. 8')
check('M2 vs RW: p = 1e-4', fl(f4[0]['p_boot']), 0.0001, tol_abs=1e-9, location='Sec. 8')
bm = mf1['baseline_mses']
for k, e, lab in [('rw', 0.1039, 'RW'), ('B1_pooled', 0.0746, 'pooled linear'), ('B2_interaction', 0.0746, 'interaction'),
                  ('B3_stratified', 0.0754, 'stratified linear'), ('M2_ens', 0.0757, 'M2 ensemble')]:
    check(f'Test MSE {lab}', round(float(bm[k]), 4), e, tol_abs=5e-5, location='Appendix (RV)')
check('Claim level C (most conservative)', mf1['claim_level'], 'C', location='Sec. 8')
check('Hyperparameter grid: 4 cells', len(mf1['grid']), 4, location='Appendix (RV)')
check('Selected cell: lr 3e-4', float(mf1['selected']['lr']), 3e-4, tol_abs=1e-12, location='Appendix (RV)')
check('Selected cell: hidden 16', int(mf1['selected']['hidden']), 16, location='Appendix (RV)')
check('5 final seeds', len(mf1['seeds']), 5, location='Sec. 8')
check('Final-run minutes ~2.2 each on GPU (checklist Q8)', round(float(np.mean(mf1['run_minutes'])), 1), 2.2, tol_abs=0.05, location='Checklist')

# ============================================================================
section('20-seed fresh-seed block (seeds 100-119): Table 1 caption, Sec. 6.2')
# ============================================================================
# The pickle stores torch version metadata; torch is not required to read the
# numbers, so torch classes are stubbed if the package is not installed.
class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        pass


class _U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.startswith('torch'):
            try:
                return super().find_class(mod, name)
            except Exception:
                return _Stub
        return super().find_class(mod, name)

    def persistent_load(self, pid):
        return _Stub()


with open(os.path.join(ART, 'kdd_s81_freshseed.pkl'), 'rb') as f:
    fs = _U(f).load()
check('Fresh block is the confirmation seed set (100-119)', fs['design']['seed_set'], 'confirmation (100-119)', location='Sec. 6.2')
P = fs['predictions']
tt = np.asarray(fs['targets'])
check('20 M2 seeds in the fresh block', len(P['M2_per_seed']), 20, location='Sec. 6.2')
m2 = np.array([np.mean((np.asarray(p) - tt) ** 2) for p in P['M2_per_seed']])
m15 = np.array([np.mean((np.asarray(p) - tt) ** 2) for p in P['NN_pooled_M1.5_per_seed']])
from scipy import stats as _st
t_unp = _st.ttest_ind(m15, m2, equal_var=False).statistic
check('Fresh set: unpaired t = +8.67 (M2 per-seed MSE below pooled NN)', round(float(t_unp), 2), 8.67, tol_abs=5e-3, location='Sec. 6.2')
check('Fresh set: unpaired p < 1e-5', _st.ttest_ind(m15, m2, equal_var=False).pvalue < 1e-5, True, location='Sec. 6.2')
check('Fresh set: same-seed comparisons favoring M2 = 20/20 (39/40 with the primary block)', int((m2 < m15).sum()), 20, location='Sec. 6.2')
sn_rw = [r for r in fs['fair_dm']['table'] if r['comparison'].startswith('C1') and r['scope'] == 'overall'][0]
check('Table 1 caption: stratified-NN ensemble vs RW, fresh set, clustered p = 0.54', round(float(sn_rw['p_cluster']), 2), 0.54, tol_abs=5e-3, location='Table 1 caption / Appendix')

# ============================================================================
section('20-seed primary block (seeds 0-19): Table 1, Table 2 row 5, Sec. 6.2')
# ============================================================================
import hashlib
_pp = os.path.join(ART, 'vdem_v1h_postanalysis_20seed.pkl')
with open(_pp, 'rb') as f:
    _sha = hashlib.sha256(f.read()).hexdigest()
check('Primary block SHA-256 matches the hash recorded by downstream runs (dff52af6...)', _sha[:16], 'dff52af6dcdf8a66', location='artifacts')
with open(_pp, 'rb') as f:
    pb = _U(f).load()
P20 = pb['predictions']
t20 = np.asarray(pb['targets'])
c20 = np.asarray(pb['country_id'])


def _mse(p):
    return float(np.mean((np.asarray(p) - t20) ** 2))


rw20 = _mse(P20['rw'])
print('\n--- Table 1 (deterministic rows) ---')
for key, e_mse, e_skill, lab in [('rw', 0.00340, 0.00, 'Random walk'),
                                  ('linear_stratified_M1.5', 0.00337, 0.84, 'Stratified linear M1.5'),
                                  ('linear_Y_tm1_only', 0.00346, -1.99, 'Linear in Y_{t-1} only'),
                                  ('linear_pooled_M1.5', 0.00347, -2.25, 'Pooled linear M1.5')]:
    m = _mse(P20[key])
    check(f'{lab}: ensemble MSE', round(m, 5), e_mse, tol_abs=5e-6, location='Table 1')
    check(f'{lab}: skill vs RW (%)', round(100 * (1 - m / rw20), 2), e_skill, tol_abs=5e-3, location='Table 1')
print('\n--- Table 1 (20-seed rows: per-run mean +/- sd, ensemble, skill) ---')
for key, ens_key, e_mean, e_sd, e_ens, e_skill, lab in [
        ('NN_pooled_M1_per_seed', 'NN_pooled_M1_ensemble', 0.00335, 0.00005, 0.00333, 1.80, 'M1 NN (Y-only)'),
        ('NN_pooled_M1.5_per_seed', 'NN_pooled_M1.5_ensemble', 0.00350, 0.00004, 0.00339, 0.08, 'Pooled NN M1.5'),
        ('NN_stratified_M1.5_per_seed', 'NN_stratified_M1.5_ensemble', 0.00344, 0.00005, 0.00324, 4.60, 'Stratified NN M1.5'),
        ('M2_per_seed', 'M2_mean', 0.00342, 0.00003, 0.00339, 0.26, 'M2 (PANAR-X)')]:
    per = np.array([_mse(q) for q in P20[key]])
    ens = _mse(P20[ens_key])
    check(f'{lab}: 20 seeds', len(per), 20, location='Table 1')
    check(f'{lab}: per-run mean MSE', round(float(per.mean()), 5), e_mean, tol_abs=5e-6, location='Table 1')
    check(f'{lab}: per-run sd', round(float(per.std(ddof=1)), 5), e_sd, tol_abs=5e-6, location='Table 1')
    check(f'{lab}: ensemble MSE', round(ens, 5), e_ens, tol_abs=5e-6, location='Table 1')
    check(f'{lab}: skill vs RW (%)', round(100 * (1 - ens / rw20), 2), e_skill, tol_abs=5e-3, location='Table 1')
print('\n--- Sec. 6.2 per-run vs ensemble, primary block ---')
m2p = np.array([_mse(q) for q in P20['M2_per_seed']])
m15p = np.array([_mse(q) for q in P20['NN_pooled_M1.5_per_seed']])
check('Primary set: unpaired t = +6.73', round(float(_st.ttest_ind(m15p, m2p, equal_var=False).statistic), 2), 6.73, tol_abs=5e-3, location='Sec. 6.2')
check('Primary set: unpaired p < 1e-5', _st.ttest_ind(m15p, m2p, equal_var=False).pvalue < 1e-5, True, location='Sec. 6.2')
check('Primary set: same-seed comparisons favoring M2 = 19/20', int((m2p < m15p).sum()), 19, location='Sec. 6.2')
check('Both blocks: 39 of 40 same-seed comparisons favor M2', int((m2p < m15p).sum()) + int((m2 < m15).sum()), 39, location='Sec. 6.2')
check('Sec. 6.2: pooled NN per-run 0.00350 over 20 seeds', round(float(m15p.mean()), 5), 0.00350, tol_abs=5e-6, location='Sec. 6.2')
check('Sec. 6.2: pooled NN 20-seed ensemble 0.00339', round(_mse(P20['NN_pooled_M1.5_ensemble']), 5), 0.00339, tol_abs=5e-6, location='Sec. 6.2')


def _clustered(pa, pb_):
    dd = (np.asarray(pb_) - t20) ** 2 - (np.asarray(pa) - t20) ** 2
    G = len(np.unique(c20))
    n = len(dd)
    m = dd.mean()
    ssum = 0.0
    for g in np.unique(c20):
        e = dd[c20 == g] - m
        ssum += e.sum() ** 2
    var = ssum / n ** 2 * G / (G - 1)
    tt = m / np.sqrt(var)
    return m, tt, 2 * _st.t.sf(abs(tt), G - 1), G


print('\n--- Table 2 row 5 (seed-fair 20-vs-20) and Table 1 caption ---')
m, tt, pv, G = _clustered(P20['M2_mean'], P20['NN_pooled_M1.5_ensemble'])
check('Row 5: mean d x1e4 = +0.062', round(m * 1e4, 3), 0.062, tol_abs=5e-4, location='Table 2')
check('Row 5: cluster t = +0.245', round(tt, 3), 0.245, tol_abs=5e-4, location='Table 2')
check('Row 5: cluster p = 0.807', round(pv, 3), 0.807, tol_abs=5e-4, location='Table 2, Sec. 6.2')
fh = [x for x in pb['fair_dm']['table'] if x['comparison'].startswith('M2 ens vs pooled NN M1.5 ens') and x['scope'] == 'overall'][0]
check('Row 5: stored block-bootstrap p = 0.829', round(float(fh['p_blockboot']), 3), 0.829, tol_abs=5e-4, location='Table 2')
check('Row 5: stored clustered t agrees with recomputation', round(float(fh['t_cluster']), 3), round(tt, 3), tol_abs=1e-3, location='Table 2')
m, tt, pv, G = _clustered(P20['NN_stratified_M1.5_ensemble'], P20['rw'])
check('Table 1 caption: stratified-NN ensemble vs RW, primary set, clustered p = 0.51', round(pv, 2), 0.51, tol_abs=5e-3, location='Table 1 caption / Appendix')
check('151 country clusters', G, 151, location='Table 2')

# ============================================================================
section('Frontier pickle: ten-cell V-Dem parity and corrected leakage bound (Sec. 9, Sec. 4, Sec. 6.3)')
# ============================================================================
with open(os.path.join(ART, 'vdem_v1h_frontier.pkl'), 'rb') as f:
    fr = pickle.load(f)
res = fr['results']
check('Ten enforcement cells including hard routing', len(res), 10, location='Sec. 9')
means = {k: float(np.mean([e['test_mse'] for e in v])) for k, v in res.items()}
sds = {k: float(np.std([e['test_mse'] for e in v], ddof=1)) for k, v in res.items()}
check('Per-run M2 MSE min over cells rounds to 0.00343', round(min(means.values()), 5), 0.00343, tol_abs=5e-6, location='Sec. 9')
check('Per-run M2 MSE max over cells rounds to 0.00345', round(max(means.values()), 5), 0.00345, tol_abs=5e-6, location='Sec. 9')
check('Seed sd min rounds to 0.00003', round(min(sds.values()), 5), 0.00003, tol_abs=5e-6, location='Sec. 9')
check('Seed sd max rounds to 0.00005', round(max(sds.values()), 5), 0.00005, tol_abs=5e-6, location='Sec. 9')
check('Hard-routing cell inside the parity range', 0.00343 <= round(means[('hard_routing', 0.0)], 5) <= 0.00345, True, location='Sec. 9')
soft = [e for k, v in res.items() if k[0] != 'hard_routing' for e in v]
check('45 soft-gated audited runs', len(soft), 45, location='Sec. 4')
corr = []
shift = []
for e in soft:
    c = max(e['L_max'] + lp * e['M'] for lp in e['L_phi_eq_per_regime'])
    corr.append(c)
    shift.append(c - e['L_eq_max'])
check('Corrected (eps = 1) L_eq,max at most 0.362 over 45 runs', round(max(corr), 3), 0.362, tol_abs=5e-4, location='Sec. 4, Sec. 6.3')
check('Max per-run shift at most 0.006', max(shift) <= 0.006, True, location='Sec. 4')
check('All 45 soft-gated runs certified under the corrected bound', all(c < 1 for c in corr), True, location='Sec. 4, Sec. 6.3')
rho_c = [c ** (1 - e['alpha_emp']) * e['B_trans'] ** e['alpha_emp'] if e['B_trans'] > 1 else c for c, e in zip(corr, soft)]
check('Corrected rho at most 0.37', max(rho_c) <= 0.37, True, location='Sec. 6.3')

# ============================================================================
section('Figure 1 relabeling (panels C-E)')
# ============================================================================
with open(os.path.join(ART, 'vdem_v1h_postanalysis.pkl'), 'rb') as f:
    pa = pickle.load(f)
t = np.asarray(pa['targets'])
reg = np.asarray(pa['regime_assignment'])
e_pool = (np.asarray(pa['predictions']['NN_pooled_M1.5']) - t) ** 2
e_m2 = (np.asarray(pa['predictions']['M2_mean']) - t) ** 2
for rg, e_pct, e_n in [(0, 7.3, 252), (1, 0.5, 621), (2, 6.3, 637)]:
    m = reg == rg
    pct = 100 * (e_pool[m].mean() - e_m2[m].mean()) / e_pool[m].mean()
    check(f'Regime {rg}: N', int(m.sum()), e_n, location='Fig. 1')
    check(f'Regime {rg}: MSE {e_pct}% below pooled', round(pct, 1), e_pct, tol_abs=0.05, location='Fig. 1')

# ============================================================================
section('FINAL VERDICT (camera-ready additions)')
# ============================================================================
print()
print(f'Total checks:  {n_pass + n_fail}')
print(f'Passed:        {n_pass}')
print(f'Failed:        {n_fail}')
print()
if n_fail == 0:
    print('All camera-ready numbers reproduce from the archived rebuttal artifacts.')
    sys.exit(0)
else:
    print(f'{n_fail} numerical disagreement(s): see [FAIL] entries above.')
    sys.exit(1)
