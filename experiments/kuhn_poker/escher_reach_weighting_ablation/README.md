# ESCHER Reach-Probability Weighting Ablation

This experiment tests whether weighting the average-policy supervised loss by
the acting player's reach probability improves the learned ESCHER average policy
in Kuhn poker.

The baseline arm is the aligned ESCHER implementation: average-policy samples are
weighted by CFR iteration only. The treatment arm multiplies that same iteration
weight by the acting player's reach probability, then mean-normalises the reach
multiplier inside each policy-training batch. This tests relative sample
emphasis without simply changing the effective learning-rate scale.

The stored reach probability excludes chance reach. Chance already controls
which sampled states enter memory, so including it again in the loss weight would
double-count chance.

## Run

```bash
python -m experiments.kuhn_poker.escher_reach_weighting_ablation.run
```

Quick smoke test:

```bash
python -m experiments.kuhn_poker.escher_reach_weighting_ablation.run \
  --seeds 1234 \
  --iterations 2 \
  --traversals 5 \
  --value-traversals 5 \
  --policy-network-train-steps 2 \
  --regret-network-train-steps 2 \
  --value-network-train-steps 2 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests
```

## Outputs

Each run creates a timestamped directory under `outputs/` containing:

- `seed_summary.csv`: one row per variant and seed.
- `variant_aggregate_summary.csv`: long-form variant/metric summaries.
- `paired_differences_vs_baseline.csv`: matched-seed reach-minus-baseline deltas.
- `paired_difference_summary.csv` and `paired_difference_summary.json`: aggregate paired deltas.
- `checkpoint_curves.csv`: long-form diagnostic curves for both variants.
- `aggregate_summary.json`: JSON form of variant and paired summaries.
- `experiment_metadata.json`: config, variants, seeds, versions, and implementation note.
- `reach_weighting_ablation_curves.npz`: compact NumPy export.
- PNG plots for exploitability, value error, diagnostic losses, and paired effects.

Negative paired deltas mean the reach-weighted treatment improved an error
metric relative to the iteration-only baseline.

