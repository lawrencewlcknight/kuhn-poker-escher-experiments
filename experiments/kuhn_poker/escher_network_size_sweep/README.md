# ESCHER Network-Size Sweep

This experiment tests how hidden-layer width and depth affect the revised
ESCHER configuration identified by Experiments 12 and 13. It keeps the
Experiment 13 training protocol fixed and varies only the architecture shared
by the policy, regret, and history-value networks.

The default run uses one seed, `1234`, and evaluates:

- `tiny_32_32`: `(32, 32)`
- `lightweight_64_64`: `(64, 64)`
- `narrow_128_64`: `(128, 64)`
- `balanced_128_128`: `(128, 128)`
- `exp13_reference_256_128`: `(256, 128)`
- `wide_256_256`: `(256, 256)`
- `very_wide_512_256`: `(512, 256)`
- `shallow_256`: `(256,)`
- `deep_128_128_64`: `(128, 128, 64)`
- `deep_256_256_128`: `(256, 256, 128)`

## Run

From the repository root:

```bash
python -m experiments.kuhn_poker.escher_network_size_sweep.run
```

Run a subset:

```bash
python -m experiments.kuhn_poker.escher_network_size_sweep.run \
  --variant-ids lightweight_64_64,exp13_reference_256_128,wide_256_256
```

Quick smoke test:

```bash
python -m experiments.kuhn_poker.escher_network_size_sweep.run \
  --variant-ids tiny_32_32,shallow_256 \
  --iterations 2 \
  --traversals 5 \
  --value-traversals 5 \
  --policy-network-train-steps 2 \
  --regret-network-train-steps 2 \
  --value-network-train-steps 2 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests
```

## Main outputs

- `variant_summary.csv`
- `checkpoint_curves.csv`
- `summary.json`
- `experiment_metadata.json`
- `final_exploitability_by_variant.png`
- `intermediate_exploitability_by_iteration.png`
- `average_policy_value_by_iteration.png`
