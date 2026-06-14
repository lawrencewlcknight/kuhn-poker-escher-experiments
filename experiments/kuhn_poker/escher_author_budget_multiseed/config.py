"""Configuration for the author-budget Kuhn poker ESCHER validation run."""

from __future__ import annotations

from copy import deepcopy

from experiments.kuhn_poker.escher_multiseed_baseline.config import (
    DEFAULT_CONFIG as BASELINE_DEFAULT_CONFIG,
    DEVELOPMENT_SEEDS_5,
)

DEFAULT_CONFIG = deepcopy(BASELINE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "kuhn_poker_escher_author_budget_multiseed",
    "num_iterations": 80,
    "num_traversals": 500,
    "num_val_fn_traversals": 500,
    "policy_network_layers": (256, 128),
    "regret_network_layers": (256, 128),
    "value_network_layers": (256, 128),
    "batch_size_regret": 256,
    "batch_size_value": 256,
    "batch_size_average_policy": 10_000,
    "policy_network_train_steps": 1000,
    "regret_network_train_steps": 200,
    "value_network_train_steps": 200,
    "importance_sampling": False,
    "zero_regret_fallback": "uniform",
    "all_actions": True,
    "expl": 1.0,
    "val_expl": 0.01,
})

DEFAULT_SEEDS = list(DEVELOPMENT_SEEDS_5)
