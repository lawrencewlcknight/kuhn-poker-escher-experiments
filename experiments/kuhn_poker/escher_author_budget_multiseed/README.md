# ESCHER Author-Budget Multi-Seed Validation

This experiment validates the best-performing configuration from Experiment 12
over a larger 80-iteration budget and five seeds. It keeps the Experiment 1
multi-seed runner structure, but replaces the lightweight baseline solver
settings with the `author_budget_no_is_uniform` configuration identified by the
diagnostic hypothesis sweep.

The default solver specification is:

- 80 ESCHER iterations
- five seeds: `1234,2025,31415,27182,16180`
- 500 regret traversals and 500 history-value traversals per iteration
- 256-by-128 policy, regret, and history-value networks
- average-policy batch size 10,000
- 1,000 average-policy training steps per policy-training event
- 200 regret/value training steps per update
- no importance-sampling correction in regret targets
- uniform legal-action fallback when all positive regrets are zero

## Run

From the repository root:

```bash
python -m experiments.kuhn_poker.escher_author_budget_multiseed.run
```

Quick smoke test:

```bash
python -m experiments.kuhn_poker.escher_author_budget_multiseed.run \
  --seeds 1234 \
  --iterations 2 \
  --traversals 5 \
  --value-traversals 5 \
  --policy-network-train-steps 2 \
  --regret-network-train-steps 2 \
  --value-network-train-steps 2 \
  --evaluation-interval 1 \
  --policy-network-layers 32,32 \
  --regret-network-layers 32,32 \
  --value-network-layers 32,32 \
  --output-root outputs/smoke_tests
```

## Main outputs

- `seed_summary.csv`
- `aggregate_summary.json`
- `checkpoint_curves.csv`
- `experiment_metadata.json`
- `exploitability_by_iteration_multiseed.png`
- `exploitability_by_nodes_multiseed.png`
- `policy_value_error_multiseed.png`
- diagnostic loss plots
