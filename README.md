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
│   ├── constants.py                                   # Kuhn value, thresholds, shuffle sizes
│   └── seeding.py                                     # TensorFlow/NumPy/Python seeding helpers
├── experiments/
│   └── kuhn_poker/
│       └── escher_multiseed_baseline/                 # Experiment 1
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

Future ESCHER ablations should be added as separate experiment folders under `experiments/kuhn_poker/`, while reusing the shared `escher_poker` package.

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

## Notes for adding future experiments

When adding a new ESCHER experiment, follow the same pattern as the baseline:

1. create a new folder under `experiments/kuhn_poker/`;
2. include a `config.py`, `run.py`, and `README.md`;
3. hold the baseline protocol fixed except for the intended treatment variable;
4. use matched seeds where possible;
5. export the same core metrics and plots so the thesis results have consistent look and feel.
