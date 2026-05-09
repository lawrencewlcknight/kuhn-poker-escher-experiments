# Output conventions

Each experiment should create a timestamped directory under `outputs/` and write the following common files where applicable:

- `experiment_metadata.json` — configuration, seeds, and run timestamp.
- `seed_summary.csv` — one row per seed.
- `aggregate_summary.json` — mean, standard deviation, standard error, and count for scalar summary metrics.
- `checkpoint_curves.csv` — one row per evaluation checkpoint per seed.
- thesis-style PNG plots using consistent figure names.

The ESCHER baseline currently produces:

- `exploitability_by_iteration_multiseed.png`
- `exploitability_by_nodes_multiseed.png`
- `policy_value_error_multiseed.png`
- `policy_loss_diagnostic.png`
- `regret_loss_diagnostic.png`
- `value_loss_diagnostic.png`

Future ablations should preserve these outputs where the metrics are meaningful.

For multi-arm ablations, use the same common filenames with a `variant_id` column:

- `seed_summary.csv` — one row per variant/seed.
- `checkpoint_curves.csv` — one row per checkpoint per variant/seed, plus final policy-evaluation rows when useful.
- `variant_aggregate_summary.csv` — long-form variant/metric summary table.
- `paired_differences_vs_baseline.csv` — one row per matched seed and non-reference variant.
- `paired_difference_summary.csv` and `paired_difference_summary.json` — aggregate paired deltas.

Plot filenames should describe the comparison plainly, for example:

- `final_exploitability_by_variant.png`
- `final_policy_value_error_by_variant.png`
- `runtime_by_variant.png`
- `paired_delta_final_exploitability_vs_baseline.png`

For checkpoint-stability experiments, keep the training and exact head-to-head analysis in the same timestamped run directory:

- `checkpoint_training_curves.csv` — per-stage intermediate ESCHER diagnostics.
- `checkpoint_stage_summary.csv` — one row per seed/checkpoint in the checkpointed arm.
- `continuous_baseline_summary.csv` — one row per seed for the uninterrupted baseline arm.
- `snapshot_inventory.csv` and `loaded_policy_inventory.csv` — saved playable policy artifacts.
- `checkpoint_exploitability_metrics.csv` — exact exploitability and policy-value metrics by checkpoint.
- `head_to_head_exact_pairwise.csv` — exact seat-averaged EV for every ordered checkpoint pair.
- `head_to_head_exact_mean_matrix.csv` — mean EV matrix across seeds.
- `head_to_head_seed_win_fraction_matrix.csv` — fraction of seeds where the row checkpoint clearly beats the column checkpoint.
- `head_to_head_monotonicity_summary_by_seed.csv` — per-seed monotonicity rates and violations.
- `head_to_head_strength_with_metrics.csv` and `head_to_head_aggregate_strength_summary.csv` — checkpoint strength summaries for plots.
- `best_checkpoint_summary.csv` — best checkpoint by head-to-head strength and exploitability.
- `final_checkpoint_vs_continuous_baseline.csv` — final checkpointed policy versus uninterrupted baseline comparison.

For constrained hyperparameter-search experiments, keep screening and confirmation artifacts in one timestamped run directory:

- `screening_seed_summary.csv` — one row per screening variant/seed.
- `screening_aggregate_by_variant.csv` — screening means, standard errors, and hyperparameter columns.
- `confirmation_seed_summary.csv` — one row per confirmed variant/seed.
- `confirmation_aggregate_by_variant.csv` — confirmation means, standard errors, and hyperparameter columns.
- `confirmation_paired_differences_vs_baseline.csv` — matched-seed deltas relative to the ESCHER baseline.
- `confirmation_paired_difference_summary.csv` — aggregate paired deltas.
- `checkpoint_curves.csv` — one row per evaluation checkpoint across both stages.
- `aggregate_summary.json` — JSON form of screening, confirmation, and paired summaries.
