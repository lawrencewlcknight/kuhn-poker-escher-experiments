# Kuhn Poker ESCHER Experiments

This repository contains reproducible experiments for evaluating ESCHER-style neural counterfactual regret minimisation on Kuhn poker using DeepMind's OpenSpiel library.

The immediate aim is to establish a thesis-quality ESCHER baseline that is aligned with the sister Deep CFR repository. Kuhn poker is used as the diagnostic environment because it is a small two-player zero-sum imperfect-information game with a known game value and exact exploitability evaluation. The results from this repository are intended to sit alongside the Deep CFR Kuhn poker experiments in an MPhil thesis on neural CFR methods for poker.

The repository is organised so that each experiment can be run independently while sharing reusable ESCHER code. The shared `escher_poker` package contains the ESCHER solver, neural-network definitions, reservoir replay buffer, plotting helpers, seeding utilities, and experiment export utilities. Each experiment lives in its own package under `experiments/kuhn_poker/<experiment_name>/`.

## Repository structure

```text
.
├── escher_poker/                                      # Shared reusable code
│   ├── solver.py                                      # ESCHER solver
│   ├── networks.py                                    # Policy, regret, and history-value networks
│   ├── replay.py                                      # Reservoir replay buffer
│   ├── experiment_utils.py                            # Run-dir, metric, and export helpers
│   ├── plotting.py                                    # Thesis-style plots
│   ├── ablation_plotting.py                           # Multi-arm ablation plots
│   ├── policy_snapshots.py                            # Saved policy snapshot helpers
│   ├── checkpoint_analysis.py                         # Exact checkpoint head-to-head analysis
│   ├── checkpoint_plotting.py                         # Checkpoint-stability plots
│   ├── hyperparameter_search.py                       # Search-stage runners and summaries
│   ├── constants.py                                   # Kuhn value, thresholds, shuffle sizes
│   └── seeding.py                                     # TensorFlow/NumPy/Python seeding helpers
├── experiments/
│   └── kuhn_poker/
│       ├── escher_multiseed_baseline/                 # Experiment 1
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_intermediate_policy_training_ablation/ # Experiment 2
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_checkpoint_stability/               # Experiment 3
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_constrained_hyperparameter_search/  # Experiment 4
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_warm_start_fair_ablation/           # Experiment 5
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_lr_schedule_ablation/               # Experiment 6
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_reach_weighting_ablation/           # Experiment 7
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_reuse_value_trajectory_ablation/    # Experiment 8
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_disk_backed_regret_memory_ablation/ # Experiment 9
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       ├── escher_on_policy_joint_regret_ablation/   # Experiment 10
│       │   ├── config.py
│       │   ├── run.py
│       │   └── README.md
│       └── escher_solver_parameter_random_search/    # Experiment 11
│           ├── config.py
│           ├── run.py
│           └── README.md
├── docs/
│   └── OUTPUT_CONVENTIONS.md
├── notebooks/                                        # Original notebook archive
├── outputs/                                          # Experiment outputs (gitignored)
├── tests/                                            # Placeholder test package
├── venv/                                             # Placeholder only; environment not committed
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── TESTING.md
```

## Experiments

### 1. Kuhn poker ESCHER multi-seed baseline

[`experiments/kuhn_poker/escher_multiseed_baseline/`](experiments/kuhn_poker/escher_multiseed_baseline/README.md)

Runs the aligned ESCHER baseline on OpenSpiel `kuhn_poker` across the same ten random seeds used in the Deep CFR baseline experiments. The primary metric is exploitability, reported as NashConv divided by two. Secondary metrics include policy-value error from the known Kuhn game value, nodes touched, wall-clock time, and final/best/final-window exploitability. Diagnostic metrics include average-policy loss, regret-network losses, history-value-network train/test loss, and replay-buffer sizes.

**Question:** under a fixed training protocol, does the ESCHER implementation learn a low-exploitability average policy in Kuhn poker, and how variable is the result across random seeds?

### 2. ESCHER intermediate average-policy training ablation

[`experiments/kuhn_poker/escher_intermediate_policy_training_ablation/`](experiments/kuhn_poker/escher_intermediate_policy_training_ablation/README.md)

Compares the baseline ESCHER diagnostic protocol against final-only average-policy extraction. In the baseline, each intermediate exploitability checkpoint trains a playable average-policy network from average-policy memory. The ablation asks whether those repeated supervised policy-network training events affect final exploitability, or whether they are mainly an evaluation cost.

**Question:** does disabling intermediate policy-network training change final ESCHER performance, once the regret/history-value training configuration and seeds are held fixed?

The experiment has three arms: the baseline intermediate-checkpoint regime, final-only policy training with the usual single-event budget, and final-only policy training with the baseline's total policy-gradient budget matched at final extraction.

### 3. ESCHER checkpoint-stability head-to-head experiment

[`experiments/kuhn_poker/escher_checkpoint_stability/`](experiments/kuhn_poker/escher_checkpoint_stability/README.md)

Saves playable average-policy checkpoints during ESCHER training and evaluates whether later checkpoints consistently beat earlier checkpoints in exact head-to-head play. The experiment also supports a continuous-baseline arm so the checkpoint/resume mechanism can be checked against a single uninterrupted ESCHER run.

**Question:** as ESCHER training progresses, do later average-policy checkpoints become stronger than earlier checkpoints, or is checkpoint quality non-monotonic?

### 4. ESCHER constrained hyperparameter search

[`experiments/kuhn_poker/escher_constrained_hyperparameter_search/`](experiments/kuhn_poker/escher_constrained_hyperparameter_search/README.md)

Runs a bounded search around the ESCHER baseline configuration to test whether poor convergence in Kuhn poker can be explained by avoidable optimisation or approximation settings. The experiment uses a screening stage over baseline, targeted, and random candidates, then confirms the strongest candidates against the baseline under matched seeds.

**Question:** can a constrained change to ESCHER hyperparameters produce reliably lower exploitability than the thesis baseline?

### 5. ESCHER warm-start fair ablation

[`experiments/kuhn_poker/escher_warm_start_fair_ablation/`](experiments/kuhn_poker/escher_warm_start_fair_ablation/README.md)

Runs paired continuous and checkpoint/resume ESCHER arms to test whether interrupting training, saving the full solver state, loading it into a fresh solver, and continuing changes final policy quality. The warm-start boundary defaults to iteration 30, matching the staged exploratory workflow.

**Question:** does checkpoint/resume behaviour introduce an unintended confound relative to an uninterrupted ESCHER baseline with the same headline training budget?

### 6. ESCHER learning-rate schedule ablation

[`experiments/kuhn_poker/escher_lr_schedule_ablation/`](experiments/kuhn_poker/escher_lr_schedule_ablation/README.md)

Compares the constant-learning-rate ESCHER baseline against a decaying learning-rate schedule under matched seeds and matched training budgets. The default scheduled arm uses cosine decay from the baseline learning rate to 10% of that value.

**Question:** can learning-rate decay stabilise ESCHER's value/regret optimisation enough to reduce exploitability relative to the constant-learning-rate baseline?

### 7. ESCHER average-policy reach-weighting ablation

[`experiments/kuhn_poker/escher_reach_weighting_ablation/`](experiments/kuhn_poker/escher_reach_weighting_ablation/README.md)

Compares the baseline average-policy regression loss, weighted by CFR iteration only, against a treatment that also weights samples by the acting player's reach probability. Reach multipliers are mean-normalised within each policy-training batch and exclude chance reach.

**Question:** does reach-probability weighting improve the learned average policy produced from ESCHER's average-policy memory?

### 8. ESCHER value-trajectory reuse ablation

[`experiments/kuhn_poker/escher_reuse_value_trajectory_ablation/`](experiments/kuhn_poker/escher_reuse_value_trajectory_ablation/README.md)

Compares the baseline ESCHER value-data collection scheme, which uses a dedicated history-value traversal pass, against a treatment that reuses player-0 regret traversals to populate the history-value memory. The treatment keeps value-test traversals for diagnostics but removes the dedicated value-training traversal pass.

**Question:** can ESCHER reduce traversal cost by reusing regret trajectories for value training without degrading the learned average policy?

### 9. ESCHER disk-backed regret-memory ablation

[`experiments/kuhn_poker/escher_disk_backed_regret_memory_ablation/`](experiments/kuhn_poker/escher_disk_backed_regret_memory_ablation/README.md)

Compares the standard in-memory regret replay buffers against a disk-backed TFRecord regret replay backend streamed during regret-network training. Average-policy replay is disk-backed in both arms so the treatment isolates regret-memory storage.

**Question:** can ESCHER reduce regret replay RAM pressure with disk-backed TFRecord shards while preserving strategic performance?

### 10. ESCHER on-policy joint-regret ablation

[`experiments/kuhn_poker/escher_on_policy_joint_regret_ablation/`](experiments/kuhn_poker/escher_on_policy_joint_regret_ablation/README.md)

Compares the baseline separate player-specific regret traversal batches against an on-policy joint-regret update variant. The treatment samples one trajectory batch from the current joint regret-matching policy and writes regret targets for the acting player at each visited decision node.

**Question:** can ESCHER reduce regret-data generation work by collecting on-policy joint regret samples without degrading the learned average policy?

### 11. ESCHER solver-parameter random search

[`experiments/kuhn_poker/escher_solver_parameter_random_search/`](experiments/kuhn_poker/escher_solver_parameter_random_search/README.md)

Runs a bounded two-stage random search over configurable ESCHER solver parameters. Screening evaluates the baseline plus sampled solver configurations under a reduced budget; confirmation compares the strongest sampled configurations against the ESCHER baseline with matched seeds and the full baseline budget.

**Question:** is ESCHER's Kuhn poker non-convergence partly caused by a poor balance between traversal budget, value fitting, regret fitting, policy extraction, exploration, and network capacity?

## Setup

Create and activate a Python 3.9 virtual environment. The repository contains
a placeholder `venv/` directory, but the actual environment is not committed.

```bash
python3.9 -m venv venv
source venv/bin/activate       # macOS/Linux
# .\venv\Scripts\Activate.ps1   # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

OpenSpiel installation can vary by platform. If `pip install -r requirements.txt` fails on `open_spiel`, install OpenSpiel following the official instructions for your platform.

## Running the experiments

From the repository root:

```bash
# Experiment 1 — full aligned ESCHER baseline
python -m experiments.kuhn_poker.escher_multiseed_baseline.run

# Experiment 2 — intermediate policy-training ablation
python -m experiments.kuhn_poker.escher_intermediate_policy_training_ablation.run

# Experiment 3 — checkpoint-stability head-to-head analysis
python -m experiments.kuhn_poker.escher_checkpoint_stability.run

# Experiment 4 — constrained hyperparameter search
python -m experiments.kuhn_poker.escher_constrained_hyperparameter_search.run

# Experiment 5 — warm-start/checkpoint-resume fair ablation
python -m experiments.kuhn_poker.escher_warm_start_fair_ablation.run

# Experiment 6 — learning-rate schedule ablation
python -m experiments.kuhn_poker.escher_lr_schedule_ablation.run

# Experiment 7 — average-policy reach-weighting ablation
python -m experiments.kuhn_poker.escher_reach_weighting_ablation.run

# Experiment 8 — value-trajectory reuse ablation
python -m experiments.kuhn_poker.escher_reuse_value_trajectory_ablation.run

# Experiment 9 — disk-backed regret-memory ablation
python -m experiments.kuhn_poker.escher_disk_backed_regret_memory_ablation.run

# Experiment 10 — on-policy joint-regret ablation
python -m experiments.kuhn_poker.escher_on_policy_joint_regret_ablation.run

# Experiment 11 — solver-parameter random search
python -m experiments.kuhn_poker.escher_solver_parameter_random_search.run
```

For a quick smoke test:

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

python -m experiments.kuhn_poker.escher_intermediate_policy_training_ablation.run \
  --seeds 1234 \
  --iterations 10 \
  --traversals 50 \
  --value-traversals 50 \
  --policy-network-train-steps 20 \
  --regret-network-train-steps 20 \
  --value-network-train-steps 20 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests

python -m experiments.kuhn_poker.escher_checkpoint_stability.run \
  --seeds 1234 \
  --checkpoint-schedule 1,2 \
  --traversals 50 \
  --value-traversals 50 \
  --policy-network-train-steps 20 \
  --regret-network-train-steps 20 \
  --value-network-train-steps 20 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests

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

python -m experiments.kuhn_poker.escher_reach_weighting_ablation.run \
  --seeds 1234 \
  --iterations 2 \
  --traversals 5 \
  --value-traversals 5 \
  --policy-network-train-steps 2 \
  --regret-network-train-steps 2 \
  --value-network-train-steps 2 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests

python -m experiments.kuhn_poker.escher_reuse_value_trajectory_ablation.run \
  --seeds 1234 \
  --iterations 2 \
  --traversals 5 \
  --value-traversals 5 \
  --value-test-traversals 2 \
  --policy-network-train-steps 2 \
  --regret-network-train-steps 2 \
  --value-network-train-steps 2 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests

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

python -m experiments.kuhn_poker.escher_on_policy_joint_regret_ablation.run \
  --seeds 1234 \
  --iterations 2 \
  --traversals 5 \
  --value-traversals 5 \
  --policy-network-train-steps 2 \
  --regret-network-train-steps 2 \
  --value-network-train-steps 2 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests

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

Outputs are written to a timestamped subdirectory under `outputs/` by default. The key files are:

```text
seed_summary.csv
aggregate_summary.json
checkpoint_curves.csv
experiment_metadata.json
exploitability_by_iteration_multiseed.png
exploitability_by_nodes_multiseed.png
policy_value_error_multiseed.png
policy_loss_diagnostic.png
regret_loss_diagnostic.png
value_loss_diagnostic.png
```

Ablation experiments also export variant-level and paired-comparison files such as:

```text
variant_aggregate_summary.csv
paired_differences_vs_baseline.csv
paired_difference_summary.csv
paired_difference_summary.json
final_exploitability_by_variant.png
runtime_by_variant.png
```

Checkpoint-stability experiments also export policy snapshots, exact pairwise head-to-head matrices, monotonicity summaries, and checkpoint-strength plots such as:

```text
checkpoint_stage_summary.csv
checkpoint_exploitability_metrics.csv
head_to_head_exact_pairwise.csv
head_to_head_exact_mean_matrix.csv
head_to_head_monotonicity_summary_by_seed.csv
head_to_head_strength_vs_earlier_aggregate.png
head_to_head_later_vs_earlier_matrix.png
```

Hyperparameter-search experiments export screening and confirmation summaries, paired confirmation deltas, and stage-level plots such as:

```text
screening_seed_summary.csv
screening_aggregate_by_variant.csv
confirmation_seed_summary.csv
confirmation_aggregate_by_variant.csv
confirmation_paired_differences_vs_baseline.csv
confirmation_paired_difference_summary.csv
screening_exploitability_by_iteration.png
confirmation_final_exploitability_by_variant.png
```

Warm-start ablations export paired continuous/resumed summaries and checkpoint-resume artifacts such as:

```text
seed_summary.csv
paired_summary.csv
paired_aggregate_summary.csv
paired_checkpoint_differences.csv
warm_start_exploitability_by_iteration.png
warm_start_paired_delta_exploitability_warm_minus_baseline.png
checkpoints/
```

Learning-rate schedule ablations export schedule-level summaries, paired deltas, active learning-rate curves, and diagnostics such as:

```text
seed_summary.csv
schedule_aggregate_summary.csv
paired_differences_vs_baseline.csv
paired_difference_summary.csv
checkpoint_curves.csv
lr_schedule_learning_rates.png
lr_schedule_exploitability_by_iteration.png
lr_schedule_paired_delta_final_exploitability.png
```

Reach-weighting ablations export matched variant summaries, reach diagnostics, paired deltas, and plots such as:

```text
seed_summary.csv
variant_aggregate_summary.csv
paired_differences_vs_baseline.csv
paired_difference_summary.csv
checkpoint_curves.csv
exploitability_by_iteration_reach_ablation.png
paired_final_exploitability_delta_reach_minus_baseline.png
```

Value-trajectory reuse ablations export matched variant summaries, traversal-budget diagnostics, paired deltas, and plots such as:

```text
seed_summary.csv
variant_aggregate_summary.csv
paired_differences_vs_baseline.csv
paired_difference_summary.csv
checkpoint_curves.csv
exploitability_by_iteration_reuse_ablation.png
dedicated_value_traversals_reuse_ablation.png
paired_final_exploitability_delta_reuse_minus_baseline.png
```

Disk-backed regret-memory ablations export matched variant summaries, memory/storage diagnostics, paired deltas, and plots such as:

```text
seed_summary.csv
variant_aggregate_summary.csv
paired_differences_vs_baseline.csv
paired_difference_summary.csv
checkpoint_curves.csv
exploitability_by_iteration_regret_memory_ablation.png
peak_rss_by_variant.png
regret_storage_mb_by_variant.png
paired_final_exploitability_delta_disk_minus_baseline.png
replay/
```

On-policy joint-regret ablations export matched variant summaries, traversal-budget diagnostics, paired deltas, and plots such as:

```text
seed_summary.csv
variant_aggregate_summary.csv
paired_differences_vs_baseline.csv
paired_difference_summary.csv
checkpoint_curves.csv
exploitability_by_iteration_on_policy_ablation.png
nominal_regret_traversals_by_variant.png
paired_final_exploitability_delta_on_policy_minus_baseline.png
```

Solver-parameter random searches export screening and confirmation summaries, paired confirmation deltas, stage-specific curves, sampled-parameter metadata, and plots such as:

```text
screening_seed_summary.csv
screening_aggregate_by_variant.csv
confirmation_seed_summary.csv
confirmation_aggregate_by_variant.csv
confirmation_paired_differences_vs_baseline.csv
screening_checkpoint_curves.csv
confirmation_checkpoint_curves.csv
solver_parameter_random_search_curves.npz
screening_exploitability_by_iteration.png
confirmation_paired_delta_final_exploitability.png
```

## Notes for adding future experiments

When adding a new ESCHER experiment, follow the same pattern as the baseline:

1. create a new folder under `experiments/kuhn_poker/`;
2. include a `config.py`, `run.py`, and `README.md`;
3. hold the baseline protocol fixed except for the intended treatment variable;
4. use matched seeds where possible;
5. export the same core metrics and plots so the thesis results have consistent look and feel.
6. put reusable seed runners, metrics, and plotting helpers in `escher_poker` when they will likely apply to more than one experiment.
