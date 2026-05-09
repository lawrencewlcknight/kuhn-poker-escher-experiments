# ESCHER Multi-Seed Baseline

This experiment runs the aligned ESCHER baseline on OpenSpiel `kuhn_poker`.

The experiment is designed to be the ESCHER counterpart to the Deep CFR multi-seed validation experiment. It uses the same seed set and exports the same broad categories of metrics and thesis-style plots, while retaining ESCHER-specific diagnostics such as regret-network and history-value-network losses.

## Run

From the repository root:

```bash
python -m experiments.kuhn_poker.escher_multiseed_baseline.run
```

Quick smoke test:

```bash
python -m experiments.kuhn_poker.escher_multiseed_baseline.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --value-traversals 50 \
  --policy-network-train-steps 20 \
  --regret-network-train-steps 20 \
  --value-network-train-steps 20 \
  --evaluation-interval 5 \
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
