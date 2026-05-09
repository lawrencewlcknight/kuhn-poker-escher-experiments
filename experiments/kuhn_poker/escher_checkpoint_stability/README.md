# ESCHER Checkpoint-Stability Experiment

This experiment asks whether later ESCHER average-policy checkpoints are consistently stronger than earlier checkpoints.

The experiment runs a checkpointed ESCHER arm using the baseline configuration and saves playable average-policy snapshots at fixed training iterations. It then evaluates every checkpoint against every other checkpoint using exact OpenSpiel policy values, averaged across both seats. A continuous-baseline arm can also be run to check whether checkpoint/resume changes the final policy relative to a single uninterrupted ESCHER run.

## Default Schedule

```text
10, 30, 50, 75, 100, 115, 130
```

The final checkpoint matches the 130-update ESCHER baseline horizon.

## Run

From the repository root:

```bash
python -m experiments.kuhn_poker.escher_checkpoint_stability.run
```

Quick smoke test:

```bash
python -m experiments.kuhn_poker.escher_checkpoint_stability.run \
  --seeds 1234 \
  --checkpoint-schedule 1,2 \
  --traversals 50 \
  --value-traversals 50 \
  --policy-network-train-steps 20 \
  --regret-network-train-steps 20 \
  --value-network-train-steps 20 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests
```

## Main Outputs

- `experiment_metadata.json`
- `checkpoint_training_curves.csv`
- `checkpoint_stage_summary.csv`
- `continuous_baseline_summary.csv`
- `snapshot_inventory.csv`
- `loaded_policy_inventory.csv`
- `checkpoint_exploitability_metrics.csv`
- `head_to_head_exact_pairwise.csv`
- `head_to_head_exact_mean_matrix.csv`
- `head_to_head_seed_win_fraction_matrix.csv`
- `head_to_head_monotonicity_summary_by_seed.csv`
- `head_to_head_strength_with_metrics.csv`
- `head_to_head_aggregate_strength_summary.csv`
- `best_checkpoint_summary.csv`
- `final_checkpoint_vs_continuous_baseline.csv`
- `head_to_head_exact_mean_matrix.png`
- `head_to_head_later_vs_earlier_matrix.png`
- `head_to_head_seed_win_fraction_matrix.png`
- `head_to_head_strength_vs_earlier_aggregate.png`
- `head_to_head_vs_previous_checkpoint_aggregate.png`
- `checkpoint_exploitability_aggregate.png`
- `exploitability_vs_head_to_head_strength.png`

Policy snapshots are saved under `policy_snapshots/`. Full solver checkpoints are saved under `checkpoints/` when enabled.
