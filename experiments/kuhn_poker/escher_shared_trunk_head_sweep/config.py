"""Configuration for ESCHER shared-trunk/action-head sweeps."""

from __future__ import annotations

from experiments.kuhn_poker.escher_architecture_base import (
    DEFAULT_SEED,
    make_default_config,
)

DEFAULT_CONFIG = make_default_config("kuhn_poker_escher_shared_trunk_head_sweep")


def _variant(variant_id, variant_label, variant_description, **overrides):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        **overrides,
    }


VARIANTS = [
    _variant(
        "linear_heads_reference",
        "Linear heads reference",
        "Experiment 13 reference: shared trunk with a single linear action-output layer.",
        policy_network_head_depth=0,
        regret_network_head_depth=0,
    ),
    _variant(
        "policy_action_heads_64",
        "Policy action heads 64",
        "Adds one 64-unit separate per-action head to the average-policy network only.",
        policy_network_head_depth=1,
        policy_network_head_units=64,
        regret_network_head_depth=0,
    ),
    _variant(
        "regret_action_heads_64",
        "Regret action heads 64",
        "Adds one 64-unit separate per-action head to the regret networks only.",
        policy_network_head_depth=0,
        regret_network_head_depth=1,
        regret_network_head_units=64,
    ),
    _variant(
        "policy_regret_action_heads_64",
        "Policy and regret heads 64",
        "Adds one 64-unit separate per-action head to both policy and regret networks.",
        policy_network_head_depth=1,
        policy_network_head_units=64,
        regret_network_head_depth=1,
        regret_network_head_units=64,
    ),
    _variant(
        "policy_regret_action_heads_128",
        "Policy and regret heads 128",
        "Tests wider separate action heads after the shared trunk.",
        policy_network_head_depth=1,
        policy_network_head_units=128,
        regret_network_head_depth=1,
        regret_network_head_units=128,
    ),
    _variant(
        "policy_regret_deep_heads_64",
        "Policy and regret deep heads",
        "Tests two-layer separate action heads, which may overfit in small Kuhn poker.",
        policy_network_head_depth=2,
        policy_network_head_units=64,
        regret_network_head_depth=2,
        regret_network_head_units=64,
    ),
]

