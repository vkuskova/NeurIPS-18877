#!/usr/bin/env bash
#
# run_all.sh
#
# Top-level replication script for the PANAR-X reproducibility package.
# Runs the synthetic v1 baselines, synthetic Pareto-frontier sweep, V-Dem
# 50-run sweep, postanalysis, certified-bound audit, figure builder, and
# finally the verification script that asserts every paper number against
# the regenerated artifacts.
#
# Usage:
#   bash scripts/run_all.sh
#
# Expected runtime:
#   ~8 hours CPU
#   ~85 minutes Colab GPU (T4)
#
# This script must be invoked from the package root, not from inside scripts/.

set -e
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "================================================================"
echo "PANAR-X reproducibility pipeline"
echo "Started at: $(date)"
echo "Root: $ROOT"
echo "================================================================"
echo

echo "[1/6] Synthetic v1 baselines (M1, M1.5, M2 single-seed reference)"
echo "      Expected: ~5 min CPU, ~2 min GPU"
echo "      Output: artifacts/synthetic_v1_results.pkl"
python code/synthetic/synthetic_v1_baselines.py
echo

echo "[2/6] Synthetic Pareto-frontier sweep (45 runs)"
echo "      Expected: ~4.5 hours CPU, ~30 min GPU"
echo "      Output: artifacts/synthetic_v1f_frontier.pkl"
python code/synthetic/synthetic_v1f_sweep.py
echo

echo "[3/6] V-Dem 50-run sweep + baselines"
echo "      Expected: ~3.5 hours CPU, ~30 min GPU"
echo "      Output: artifacts/vdem_v1h_frontier.pkl"
python code/vdem/vdem_v1h_v3_cell.py
echo

echo "[4/6] V-Dem postanalysis (paired DM predictions)"
echo "      Expected: ~2 hours CPU, ~22 min GPU"
echo "      Output: artifacts/vdem_v1h_postanalysis.pkl"
python code/vdem/vdem_v1h_postanalysis.py
echo

echo "[5/6] Certified spectral-norm bound audit (Appendix D.2)"
echo "      Expected: ~10 min CPU, ~2 min GPU"
echo "      Output: artifacts/certified_audit_results.pkl"
python code/vdem/certified_bound_audit.py
echo

echo "[6/6] Build paper Figure 1"
echo "      Expected: ~15 sec"
echo "      Output: figures/figure_diagnostic.pdf"
python code/vdem/build_figure.py
echo

echo "================================================================"
echo "Verifying paper numbers against regenerated artifacts..."
echo "================================================================"
python verify_paper_numbers.py
echo

echo "================================================================"
echo "Pipeline complete."
echo "Finished at: $(date)"
echo "================================================================"
