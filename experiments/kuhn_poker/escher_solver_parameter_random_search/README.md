# ESCHER Solver-Parameter Random Search

This experiment performs a bounded random search over configurable ESCHER solver
parameters in Kuhn poker. It is broader than the constrained targeted
hyperparameter search: candidates are sampled across traversal budgets,
history-value fitting, regret fitting, average-policy extraction, network
capacity, exploration, importance sampling, and selected value-estimation
controls.

The experiment uses two stages. Screening runs the ESCHER baseline plus sampled
solver configurations for a reduced budget under matched seeds. Confirmation
then compares the strongest screening candidates against the ESCHER baseline
using the full baseline training budget and matched seeds.

## Run

```bash
python -m experiments.kuhn_poker.escher_solver_parameter_random_search.run
```

Quick smoke test:

```bash
python -m experiments.kuhn_poker.escher_solver_parameter_random_search.run \
  --screening-seeds 1234 \
  --confirmation-seeds 1234 \
  --screening-iterations 2 \
  --confirmation-iterations 2 \
  --screening-evaluation-interval 1 \
  --confirmation-evaluation-interval 1 \
  --n-random-candidates 1 \
  --confirmation-top-k 1 \
  --traversals 5 \
  --value-traversals 5 \
  --policy-network-train-steps 2 \
  --regret-network-train-steps 2 \
  --value-network-train-steps 2 \
  --policy-network-layers 32,32 \
  --regret-network-layers 32,32 \
  --value-network-layers 32,32 \
  --all-actions true \
  --use-balanced-probs false \
  --val-bootstrap false \
  --output-root outputs/smoke_tests
```

## Outputs

Each run creates a timestamped directory under `outputs/` containing:

- `screening_seed_summary.csv`: one row per screening variant and seed.
- `screening_aggregate_by_variant.csv`: screening means, standard errors, and sampled solver parameters.
- `confirmation_seed_summary.csv`: one row per confirmed variant and seed.
- `confirmation_aggregate_by_variant.csv`: confirmation means, standard errors, and sampled solver parameters.
- `confirmation_paired_differences_vs_baseline.csv`: matched-seed deltas relative to the ESCHER baseline.
- `confirmation_paired_difference_summary.csv`: aggregate paired deltas.
- `checkpoint_curves.csv`: combined screening and confirmation checkpoint curves.
- `screening_checkpoint_curves.csv` and `confirmation_checkpoint_curves.csv`: stage-specific curve exports.
- `aggregate_summary.json`: JSON form of screening, confirmation, and paired summaries.
- `experiment_metadata.json`: baseline config, search space outputs, selected candidates, seeds, and versions.
- `solver_parameter_random_search_curves.npz`: compact NumPy export.
- PNG plots for screening/confirmation exploitability, final metrics, paired confirmation deltas, and confirmation diagnostic losses.

Negative paired deltas in confirmation favour the sampled solver configuration
relative to the ESCHER baseline.
