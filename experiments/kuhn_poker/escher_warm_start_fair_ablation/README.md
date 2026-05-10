# ESCHER Warm-Start Fair Ablation

This experiment tests whether interrupting ESCHER training, saving the full solver
state, loading it into a fresh solver, and then continuing changes final policy
quality in Kuhn poker.

The design is paired by seed:

- `baseline_continuous`: one uninterrupted ESCHER run.
- `warm_start`: train to `warm_start_boundary`, save a full solver checkpoint,
  reload into a fresh solver, and continue to the same final iteration axis.

The default configuration follows the aligned ESCHER baseline: 130 total
iterations, a warm-start boundary at iteration 30, 500 regret traversals, 500
history-value traversals, and the same ten thesis seeds.

## Run

```bash
python -m experiments.kuhn_poker.escher_warm_start_fair_ablation.run
```

Quick smoke test:

```bash
python -m experiments.kuhn_poker.escher_warm_start_fair_ablation.run \
  --seeds 1234 \
  --iterations 2 \
  --warm-start-boundary 1 \
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

- `seed_summary.csv`: one row per seed and arm.
- `aggregate_summary.csv` and `aggregate_summary.json`: arm-level summaries.
- `paired_summary.csv`: warm-start minus continuous-baseline paired deltas.
- `paired_aggregate_summary.csv`: aggregate paired-difference statistics.
- `checkpoint_curves.csv`: long-form diagnostic curves for thesis plots.
- `paired_checkpoint_differences.csv`: paired warm-start minus baseline curves.
- `experiment_metadata.json`: config, seeds, versions, and interpretation notes.
- `warm_start_fair_ablation_curves.npz`: compact NumPy export.
- `checkpoints/`: full solver checkpoint files for the warm-start boundary.
- PNG plots for exploitability, policy-value error, and paired differences.

Positive paired deltas mean the warm-start arm is worse for error metrics such as
exploitability and policy-value error.

