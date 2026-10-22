# Level 1 compact-neighbor experiment

Selected neighbor tokens are gathered before the NeighborEncoder and FusionEncoder. Original 5x5 indices are retained for relative-position bias.

## Numerical equivalence

Across the four K=5 methods and eight folds, the maximum absolute PCC(M) difference was 9.16e-06.

## Efficiency benchmark

A real BC1 fold-0 training batch (batch size 128, AMP, physical GPU 2) was measured with 10 warm-up and 100 timed iterations over five repeats.

At K=5, total training peak allocated memory decreased by 6.75%, while the median step-time change was -0.74%.

Interpretation: token compaction is correct and reduces training memory, but the 25-token neighbor sequence is too short and too small a fraction of total TRIPLEX compute to guarantee an end-to-end speedup.
