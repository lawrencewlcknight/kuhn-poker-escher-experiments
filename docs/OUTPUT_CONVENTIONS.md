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
