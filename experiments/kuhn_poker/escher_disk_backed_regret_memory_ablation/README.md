# ESCHER Disk-Backed Regret-Memory Ablation

This experiment tests a memory-efficiency implementation change for ESCHER:
storing regret replay in disk-backed TFRecord shards and streaming those shards
during regret-network training.

The baseline arm stores regret replay in ordinary in-memory reservoir buffers.
The treatment arm writes regret samples to per-player, per-iteration TFRecord
shards. Average-policy replay is disk-backed in both arms so that the intended
treatment is only the regret-memory storage backend.

## Run

```bash
python -m experiments.kuhn_poker.escher_disk_backed_regret_memory_ablation.run
```

Quick smoke test:

```bash
python -m experiments.kuhn_poker.escher_disk_backed_regret_memory_ablation.run \
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
- `paired_differences_vs_baseline.csv`: matched-seed disk-minus-baseline deltas.
- `paired_difference_summary.csv` and `paired_difference_summary.json`: aggregate paired deltas.
- `checkpoint_curves.csv`: long-form diagnostic curves for both variants.
- `aggregate_summary.json`: JSON form of variant and paired summaries.
- `experiment_metadata.json`: config, variants, seeds, versions, and implementation note.
- `regret_memory_ablation_curves.npz`: compact NumPy export.
- `replay/`: disk-backed replay artifacts used by the experiment.
- PNG plots for exploitability, value error, diagnostic losses, peak RSS, regret disk footprint, and paired effects.

Negative paired deltas for exploitability and policy-value error favour the
disk-backed treatment. Negative peak-RSS deltas indicate lower process memory
use than the in-memory regret replay baseline.
