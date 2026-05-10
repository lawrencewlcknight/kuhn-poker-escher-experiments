# ESCHER Learning-Rate Schedule Ablation

This experiment tests whether a decaying learning rate stabilises ESCHER training
in Kuhn poker. The default comparison matches the notebook: the constant
learning-rate ESCHER baseline is paired by seed against a cosine decay schedule
that decays from `1e-3` to `1e-4`.

The counter-hypothesis is that ESCHER's weak convergence is not mainly caused by
late-stage update magnitude. If value-target quality, traversal data, or regret
estimation bias dominates, a decaying learning rate may not reduce
exploitability and may even reduce adaptation too early.

## Run

```bash
python -m experiments.kuhn_poker.escher_lr_schedule_ablation.run
```

Quick smoke test:

```bash
python -m experiments.kuhn_poker.escher_lr_schedule_ablation.run \
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

Optional schedules can be included with:

```bash
python -m experiments.kuhn_poker.escher_lr_schedule_ablation.run \
  --include-optional-schedules true
```

or selected explicitly:

```bash
python -m experiments.kuhn_poker.escher_lr_schedule_ablation.run \
  --include-optional-schedules true \
  --schedules constant_baseline_escher,cosine_decay_to_10pct,linear_decay_to_10pct
```

## Outputs

Each run creates a timestamped directory under `outputs/` containing:

- `seed_summary.csv`: one row per schedule and seed.
- `schedule_aggregate_summary.csv`: long-form schedule/metric summaries.
- `paired_differences_vs_baseline.csv`: matched-seed deltas versus the constant baseline.
- `paired_difference_summary.csv`: aggregate paired-difference statistics.
- `checkpoint_curves.csv`: long-form checkpoint curves with active learning rate.
- `aggregate_summary.json`: JSON form of schedule and paired summaries.
- `experiment_metadata.json`: config, schedule definitions, seeds, and library versions.
- `lr_schedule_curves.npz`: compact NumPy export for thesis plotting.
- PNG plots for learning-rate curves, exploitability, value error, losses, and paired deltas.

Positive paired deltas mean the scheduled arm is worse for error metrics such as
exploitability or policy-value error.

