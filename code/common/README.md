# Common code

Reference code that is **not** required to reproduce the paper's results
but is included for context and comparison.

## Files

- **`NAVAR.py`** — The original NAVAR architecture from Bussmann et al.
  (2021), included for reference. PANAR-X uses NAVAR-style per-edge
  additive structure but is implemented self-contained in
  `code/vdem/vdem_v1h_v3_cell.py` and `code/synthetic/synthetic_v1f_sweep.py`,
  not as a dependency on this file. The file is provided so a reader can
  compare PANAR-X's architecture decisions against NAVAR's.

- **`dataloader.py`** — NAVAR's panel-aware data loader (with the
  `split_timeseries` option). Same status as `NAVAR.py`: reference only.

## Original NAVAR repository

https://github.com/bartbussmann/NAVAR

Cite as:

> Bussmann, B., Nys, J., & Latré, S. (2021). Neural Additive Vector
> Autoregression Models for Causal Discovery in Time Series. *Discovery
> Science.* https://doi.org/10.1007/978-3-030-88942-5_35
