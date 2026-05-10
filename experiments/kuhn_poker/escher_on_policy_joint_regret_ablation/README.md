# ESCHER On-Policy Joint-Regret Ablation

This experiment compares the baseline ESCHER regret-data collection scheme
against an on-policy joint-regret update variant in Kuhn poker.

The baseline arm uses separate player-specific regret traversal batches in each
ESCHER iteration. The treatment arm samples one batch of trajectories from the
current joint regret-matching policy and, at each visited decision node, writes a
regret target for whichever player is acting. The value-training pass and final
average-policy fitting remain shared with the baseline.

## Run

```bash
python -m experiments.kuhn_poker.escher_on_policy_joint_regret_ablation.run
```

Quick smoke test:

```bash
python -m experiments.kuhn_poker.escher_on_policy_joint_regret_ablation.run \
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
- `paired_differences_vs_baseline.csv`: matched-seed on-policy-minus-baseline deltas.
- `paired_difference_summary.csv` and `paired_difference_summary.json`: aggregate paired deltas.
- `checkpoint_curves.csv`: long-form diagnostic curves for both variants.
- `aggregate_summary.json`: JSON form of variant and paired summaries.
- `experiment_metadata.json`: config, variants, seeds, versions, and implementation note.
- `on_policy_joint_regret_ablation_curves.npz`: compact NumPy export.
- PNG plots for exploitability, policy-value error, diagnostic losses, nominal regret traversals, and paired effects.

Negative paired deltas for exploitability and policy-value error favour the
on-policy joint-regret update treatment. Negative deltas for nodes, wall-clock
time, and nominal regret traversals indicate lower measured or nominal traversal
cost than the separate-player baseline.
