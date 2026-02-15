# Raw Data Validation Report (IMDb)

## Split summaries

**Split:** train
- Rows: 25000
- Null text: 0
- Empty text: 0
- Label distribution: {0: 12500, 1: 12500}
- Chars (min/p50/p95/max): 52/979/3432/13704
- Words (min/p50/p95/max): 10/174/598/2470

**Split:** test
- Rows: 25000
- Null text: 0
- Empty text: 0
- Label distribution: {0: 12500, 1: 12500}
- Chars (min/p50/p95/max): 32/962/3333/12988
- Words (min/p50/p95/max): 4/172/582/2278

## Duplicates & leakage

- Exact duplicate texts in train: **96**
- Exact duplicate texts in test: **199**
- Exact text overlap train ↔ test (leakage risk): **123**
