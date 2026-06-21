"""Configuration for ESCHER separate-network architecture sweeps."""

from __future__ import annotations

from experiments.kuhn_poker.escher_architecture_base import (
    DEFAULT_SEED,
    make_default_config,
)

DEFAULT_CONFIG = make_default_config(
    "kuhn_poker_escher_separate_network_architecture_sweep"
)


def _variant(variant_id, variant_label, variant_description, **overrides):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        **overrides,
    }


VARIANTS = [
    _variant(
        "exp13_reference_all_256_128",
        "Experiment 13 reference",
        "Reference setting from Experiment 13: all three networks use 256x128.",
        policy_network_layers=(256, 128),
        regret_network_layers=(256, 128),
        value_network_layers=(256, 128),
    ),
    _variant(
        "regret_wide_policy_reference",
        "Wide regret only",
        "Tests whether regret approximation is the main capacity bottleneck.",
        policy_network_layers=(256, 128),
        regret_network_layers=(512, 256),
        value_network_layers=(256, 128),
    ),
    _variant(
        "policy_wide_regret_reference",
        "Wide policy only",
        "Tests whether the average-policy network is limiting final exploitability.",
        policy_network_layers=(512, 256),
        regret_network_layers=(256, 128),
        value_network_layers=(256, 128),
    ),
    _variant(
        "value_wide_reference_others",
        "Wide value only",
        "Tests whether history-value estimation is the main source of regret noise.",
        policy_network_layers=(256, 128),
        regret_network_layers=(256, 128),
        value_network_layers=(512, 256),
    ),
    _variant(
        "regret_value_wide_policy_reference",
        "Wide regret and value",
        "Tests whether target quality and regret capacity need to improve together.",
        policy_network_layers=(256, 128),
        regret_network_layers=(512, 256),
        value_network_layers=(512, 256),
    ),
]

