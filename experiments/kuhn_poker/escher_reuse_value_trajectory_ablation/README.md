# ESCHER Value-Trajectory Reuse Ablation

This experiment tests whether ESCHER can reduce traversal work by reusing
player-0 regret traversals as history-value training data.

The baseline arm matches the aligned ESCHER implementation: a dedicated
value-traversal pass populates the history-value memory before regret-network
training. The treatment arm skips that dedicated value-training traversal pass
and records value targets during player-0 regret traversals instead. A small
value-test traversal set is still collected for diagnostics and is not used for
training.

## Run

```bash
python -m experiments.kuhn_poker.escher_reuse_value_trajectory_ablation.run
```

Quick smoke test:

```bash
python -m experiments.kuhn_poker.escher_reuse_value_trajectory_ablation.run \
  --seeds 1234 \
  --iterations 2 \
  --traversals 5 \
  --value-traversals 5 \
  --value-test-traversals 2 \
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
- `paired_differences_vs_baseline.csv`: matched-seed reuse-minus-baseline deltas.
- `paired_difference_summary.csv` and `paired_difference_summary.json`: aggregate paired deltas.
- `checkpoint_curves.csv`: long-form diagnostic curves for both variants.
- `aggregate_summary.json`: JSON form of variant and paired summaries.
- `experiment_metadata.json`: config, variants, seeds, versions, and implementation note.
- `reuse_value_trajectory_ablation_curves.npz`: compact NumPy export.
- PNG plots for exploitability, value error, diagnostic losses, runtime, node use, and paired effects.

Negative paired deltas for exploitability and value-error metrics favour the
trajectory-reuse treatment. Negative traversal-budget and runtime deltas indicate
lower resource use than the dedicated-value-traversal baseline.
