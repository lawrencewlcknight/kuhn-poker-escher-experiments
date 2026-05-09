# ESCHER Intermediate Average-Policy Training Ablation

This experiment tests whether ESCHER's intermediate exploitability measurement changes final results.

In this implementation, exploitability can only be computed for a playable policy. Intermediate exploitability checkpoints therefore train the average-policy network from the current average-policy memory. That diagnostic step is useful, but it is not completely passive.

## Variants

- `intermediate_every_5_baseline` — the aligned ESCHER baseline. It trains/evaluates the average-policy network at intermediate checkpoints and trains once more at final policy extraction.
- `final_only_1000_steps` — disables intermediate exploitability checks and trains the average-policy network once at the end for the usual single-event budget.
- `final_only_matched_steps` — disables intermediate exploitability checks and trains once at the end with the same total policy-gradient step budget used by the baseline arm.

The default ESCHER regret/history-value configuration and seed set are inherited from `escher_multiseed_baseline`.

## Run

From the repository root:

```bash
python -m experiments.kuhn_poker.escher_intermediate_policy_training_ablation.run
```

Quick smoke test:

```bash
python -m experiments.kuhn_poker.escher_intermediate_policy_training_ablation.run \
  --seeds 1234 \
  --iterations 10 \
  --traversals 50 \
  --value-traversals 50 \
  --policy-network-train-steps 20 \
  --regret-network-train-steps 20 \
  --value-network-train-steps 20 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests
```

To run a subset of arms:

```bash
python -m experiments.kuhn_poker.escher_intermediate_policy_training_ablation.run \
  --variant-ids intermediate_every_5_baseline,final_only_1000_steps
```

## Main Outputs

- `seed_summary.csv`
- `checkpoint_curves.csv`
- `variant_aggregate_summary.csv`
- `aggregate_summary.json`
- `paired_differences_vs_baseline.csv`
- `paired_difference_summary.csv`
- `paired_difference_summary.json`
- `experiment_metadata.json`
- `final_exploitability_by_variant.png`
- `final_policy_value_error_by_variant.png`
- `runtime_by_variant.png`
- `policy_gradient_budget_by_variant.png`
- `paired_delta_final_exploitability_vs_baseline.png`
- `baseline_intermediate_exploitability_curve.png`
- `baseline_intermediate_policy_value_error_curve.png`
