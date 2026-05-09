# Testing and Smoke Tests

This file describes the basic checks for the ESCHER Kuhn poker repository.

## 1. Environment setup

```bash
python -m venv venv
source venv/bin/activate            # macOS/Linux
# .\venv\Scripts\Activate.ps1       # Windows PowerShell
pip install --upgrade pip
pip install -r requirements-dev.txt
```

OpenSpiel can require platform-specific installation steps. If `pip install -r requirements.txt` fails on `open_spiel`, follow the official OpenSpiel install guide and rerun.

## 2. Syntax check

```bash
python -m compileall escher_poker experiments tests
```

This catches syntax errors. It does not guarantee that TensorFlow/OpenSpiel are installed correctly.

## 3. Unit-test placeholder

A test package is included so proper unit tests can be added as the ESCHER experiment suite grows.

```bash
pytest
```

At present, the most important operational test is the smoke run below.

## 4. Quick experiment smoke test

Run a tiny two-seed ESCHER baseline:

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

The run should produce a timestamped output directory containing CSV summaries, JSON metadata, and PNG plots. The smoke-test results should not be interpreted as evidence about ESCHER performance.

Run a tiny one-seed intermediate policy-training ablation:

```bash
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
```

This should produce the common output files plus variant and paired-difference summaries.

Run a tiny one-seed checkpoint-stability experiment:

```bash
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
```

This should produce checkpoint summaries, policy snapshots, exact head-to-head matrices, and PNG plots.

Run a tiny constrained hyperparameter-search experiment:

```bash
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

This should produce screening, confirmation, paired-difference, curve, and plot outputs.

## 5. Full experiment run

```bash
python -m experiments.kuhn_poker.escher_multiseed_baseline.run
python -m experiments.kuhn_poker.escher_intermediate_policy_training_ablation.run
python -m experiments.kuhn_poker.escher_checkpoint_stability.run
python -m experiments.kuhn_poker.escher_constrained_hyperparameter_search.run
```

The full run uses the default 10 seeds and the aligned ESCHER baseline configuration. This may take a long time.
