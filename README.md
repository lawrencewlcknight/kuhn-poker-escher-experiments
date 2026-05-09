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
│       └── escher_constrained_hyperparameter_search/  # Experiment 4
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

## Setup

Create and activate a virtual environment. The repository contains a placeholder `venv/` directory, but the actual environment is not committed.

```bash
python -m venv venv
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

## Notes for adding future experiments

When adding a new ESCHER experiment, follow the same pattern as the baseline:

1. create a new folder under `experiments/kuhn_poker/`;
2. include a `config.py`, `run.py`, and `README.md`;
3. hold the baseline protocol fixed except for the intended treatment variable;
4. use matched seeds where possible;
5. export the same core metrics and plots so the thesis results have consistent look and feel.
6. put reusable seed runners, metrics, and plotting helpers in `escher_poker` when they will likely apply to more than one experiment.
