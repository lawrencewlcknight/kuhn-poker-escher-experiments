# ESCHER Constrained Hyperparameter Search

This experiment performs a bounded search around the thesis ESCHER baseline configuration for Kuhn poker.

The motivating issue is that the baseline ESCHER run does not reliably converge toward low exploitability, even though Kuhn poker is small enough for exact exploitability evaluation. The search tests whether this behavior can be explained by an avoidable optimisation or approximation setting rather than by the ESCHER implementation alone.

## Design

The default protocol has two stages:

- **Screening:** baseline plus targeted and random candidate configurations are trained for a reduced budget over matched seeds.
- **Confirmation:** the strongest screening candidates are compared against the baseline using the full ESCHER baseline training budget and matched seeds.

The constrained search varies learning rate, traversal budgets, value/regret/policy fitting steps, network sizes, regret/value reinitialisation, batch sizes, replay capacity, and ESCHER traversal/value-target controls. The default search space is intentionally capped around the lightweight baseline: it includes smaller and moderately larger random candidates, while targeted arms isolate heavier interventions such as extra traversals, extra fitting, and wider networks.

To keep the full search practical on cloud VMs, the default screening stage uses 40 iterations, exploitability checks every 20 iterations, and two random candidates. Confirmation keeps the baseline 80-iteration budget but also evaluates every 20 iterations. Each config/seed run executes in a fresh Python worker process by default, so TensorFlow solver/network/replay state is released between runs.

## Run

From the repository root:

```bash
python -m experiments.kuhn_poker.escher_constrained_hyperparameter_search.run
```

For local debugging only, subprocess isolation can be disabled:

```bash
python -m experiments.kuhn_poker.escher_constrained_hyperparameter_search.run \
  --disable-subprocess-isolation
```

Quick smoke test:

```bash
python -m experiments.kuhn_poker.escher_constrained_hyperparameter_search.run \
  --screening-seeds 1234 \
  --confirmation-seeds 1234 \
  --screening-iterations 2 \
  --confirmation-iterations 2 \
  --screening-evaluation-interval 1 \
  --confirmation-evaluation-interval 1 \
  --n-random-candidates 1 \
  --confirmation-top-k 1 \
  --traversals 50 \
  --value-traversals 50 \
  --policy-network-train-steps 20 \
  --regret-network-train-steps 20 \
  --value-network-train-steps 20 \
  --output-root outputs/smoke_tests
```

For a screening-only run:

```bash
python -m experiments.kuhn_poker.escher_constrained_hyperparameter_search.run \
  --skip-confirmation
```

## Main Outputs

- `experiment_metadata.json`
- `aggregate_summary.json`
- `screening_seed_summary.csv`
- `screening_aggregate_by_variant.csv`
- `confirmation_seed_summary.csv`
- `confirmation_aggregate_by_variant.csv`
- `confirmation_paired_differences_vs_baseline.csv`
- `confirmation_paired_difference_summary.csv`
- `checkpoint_curves.csv`
- `partial_screening_seed_summary.jsonl`
- `partial_confirmation_seed_summary.jsonl`
- `partial_checkpoint_curves.jsonl`
- `worker_results/`
- `worker_logs/`
- `screening_exploitability_by_iteration.png`
- `screening_exploitability_by_nodes.png`
- `confirmation_exploitability_by_iteration.png`
- `confirmation_exploitability_by_nodes.png`
- `screening_final_exploitability_by_variant.png`
- `confirmation_final_exploitability_by_variant.png`
- `confirmation_final_window_exploitability_by_variant.png`
