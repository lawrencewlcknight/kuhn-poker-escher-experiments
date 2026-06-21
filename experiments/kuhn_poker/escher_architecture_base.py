"""Base config helpers for ESCHER architecture experiments."""

from __future__ import annotations

from copy import deepcopy

from experiments.kuhn_poker.escher_author_budget_multiseed.config import (
    DEFAULT_CONFIG as AUTHOR_BUDGET_DEFAULT_CONFIG,
)

DEFAULT_SEED = 1234


def make_default_config(experiment_name: str):
    config = deepcopy(AUTHOR_BUDGET_DEFAULT_CONFIG)
    config.update({
        "experiment_name": experiment_name,
        "policy_network_activation": "leakyrelu",
        "regret_network_activation": "leakyrelu",
        "value_network_activation": "leakyrelu",
        "policy_network_layer_norm": True,
        "regret_network_layer_norm": True,
        "value_network_layer_norm": True,
        "policy_network_residual_mode": "same_width",
        "regret_network_residual_mode": "same_width",
        "value_network_residual_mode": "same_width",
        "policy_network_head_depth": 0,
        "regret_network_head_depth": 0,
        "policy_network_head_units": None,
        "regret_network_head_units": None,
    })
    return config
