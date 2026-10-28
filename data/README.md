# Data: V-Dem v15 panel

## File

`merged_completevars_balanced_zero_missing.csv` (2.6 MB; 8,456 rows × 48 columns)

## Provenance

Derived from the **Varieties of Democracy (V-Dem) v15** dataset
(Coppedge et al. 2025, https://doi.org/10.23696/vdemds25). The original
V-Dem v15 country–year panel was filtered to the subset of country–year
observations with no missing values across all variables used in the paper,
and balanced to a common time window of 1970–2025 across the resulting
151 countries.

## Variables used in the paper

- `country_id`, `year`: panel index
- `v2xel_frefair`: Clean Elections Index (target Y)
- Four predetermined structural covariates (X channel):
  - `v2csantimv`: anti-system movements
  - `v2jureform`: judicial reforms
  - `v2psorgs`: party organization
  - `v2mecorrpt`: media corruption

The remaining columns are V-Dem variables retained for context and possible
future analyses but not used in this paper.

## Train/test split

For each of 151 countries, the last 10 years (2016–2025) are held out as
test data and earlier years (1970–2015) are training. After lag-window
construction (lag = 5), final sample sizes are:
- Training: 6,191 observations
- Test: 1,510 observations

The split is fully deterministic given the CSV.

## Licensing

V-Dem data is released under the **Creative Commons Attribution 4.0
International (CC BY 4.0)** license:
https://creativecommons.org/licenses/by/4.0/

## Citation

Required attribution when using the V-Dem data:

> Coppedge, M., Gerring, J., Knutsen, C. H., Lindberg, S. I., Teorell, J.,
> Altman, D., Bernhard, M., et al. (2025).
> *V-Dem [Country–Year/Country–Date] Dataset v15.*
> Varieties of Democracy (V-Dem) Project.
> https://doi.org/10.23696/vdemds25

## SHA-256

```
9c61549222c65491401c4552bc3311213dabaad396c2ee8aa32ccbbee722660f
```

To verify integrity:
```
sha256sum merged_completevars_balanced_zero_missing.csv
```
