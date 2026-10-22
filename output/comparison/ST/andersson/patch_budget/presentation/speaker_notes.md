# BC1 patch-budget presentation visuals

All figures include CCC, PCC(M), PCC(H), MSE and MAE. Explained variance is excluded.

## Suggested presentation message

Reducing context from 25 patches to K=5 does not necessarily reduce prediction quality. The best correlation results occur at K=5, while Similarity K=15 gives the lowest errors. Performance is not monotonic with increasing K, suggesting that some context patches are redundant or distracting.

Important wording: K=5 is sufficient among the tested budgets; it has not been proven to be the minimum possible budget.

## Best observed setting

- CCC: Similarity K=10, 0.096 (+7.5% vs Baseline-25).
- PCC(M): ST Hard K=5, 0.217 (+3.2% vs Baseline-25).
- PCC(H): Similarity K=5, 0.381 (+1.9% vs Baseline-25).
- MSE: Similarity K=15, 1.344 (+1.6% vs Baseline-25).
- MAE: Similarity K=15, 0.890 (+0.9% vs Baseline-25).

## Reading the normalized charts

The plotted percentage is improvement relative to Baseline-25. For CCC/PCC, higher raw values are better. For MSE/MAE, reductions are converted to positive improvements, so upward/green always means better.
