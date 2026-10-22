# BC1 Level 2 compact-neighbor experiment

## Correctness

| Method | Level 1 tokens | Level 2 tokens | Max absolute logit difference | Exact within 1e-5 |
|---|---:|---:|---:|---:|
| Random | 25 | 5 | 0 | True |
| Similarity | 25 | 5 | 0 | True |
| Diversity | 25 | 5 | 0 | True |

All compact caches were also validated spot-by-spot against the original 25-token files.

## Physical storage

The original BC1 CIGAR neighbor cache is 757 MiB. Each K=5 method-specific cache is 134 MiB.

## Efficiency comparison

Mask means the original K=5 implementation: 25 tokens are loaded and unselected tokens are masked. Level 1 loads 25 and compacts to 5 inside the model. Level 2 loads only 5 from disk.

Positive throughput percentages indicate faster Level 2; negative transfer/memory percentages indicate improvement.

| Method | Loader vs Mask | Transfer vs Mask | End-to-end vs Mask | End-to-end vs Level 1 | Peak memory vs Mask |
|---|---:|---:|---:|---:|---:|
| Random | +15.91% | -8.94% | +7.68% | +21.04% | -6.78% |
| Similarity | +20.13% | -7.31% | +14.96% | +23.87% | -6.78% |
| Diversity | +20.63% | -8.40% | +12.60% | +25.84% | -6.78% |

Benchmarks use BC1 fold 0, batch size 128, physical GPU 2 (RTX A6000), and five repeats.
Because Mask and Level 1 use the same input pipeline but showed timing variation, the Level 2-versus-Mask end-to-end comparison is the more conservative headline result.
Prediction metrics are inherited from the numerically equivalent Level 1 inputs; no redundant 8-fold retraining was performed.
