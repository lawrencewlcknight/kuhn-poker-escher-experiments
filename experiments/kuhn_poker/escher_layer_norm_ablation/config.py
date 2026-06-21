"""Configuration for ESCHER layer-normalisation ablations."""

from __future__ import annotations

from experiments.kuhn_poker.escher_architecture_base import (
    DEFAULT_SEED,
    make_default_config,
)

DEFAULT_CONFIG = make_default_config("kuhn_poker_escher_layer_norm_ablation")


def _variant(variant_id, variant_label, variant_description, **overrides):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        **overrides,
    }


VARIANTS = [
    _variant(
        "layer_norm_all_on",
        "Layer norm all on",
        "Experiment 13 reference: layer normalisation in all three networks.",
        policy_network_layer_norm=True,
        regret_network_layer_norm=True,
        value_network_layer_norm=True,
    ),
    _variant(
        "layer_norm_all_off",
        "Layer norm all off",
        "Tests whether layer normalisation is stabilising or suppressing useful signal.",
        policy_network_layer_norm=False,
        regret_network_layer_norm=False,
        value_network_layer_norm=False,
    ),
    _variant(
        "layer_norm_policy_off",
        "Policy layer norm off",
        "Isolates the effect of layer normalisation in the average-policy network.",
        policy_network_layer_norm=False,
        regret_network_layer_norm=True,
        value_network_layer_norm=True,
    ),
    _variant(
        "layer_norm_regret_off",
        "Regret layer norm off",
        "Isolates the effect of layer normalisation in the regret networks.",
        policy_network_layer_norm=True,
        regret_network_layer_norm=False,
        value_network_layer_norm=True,
    ),
    _variant(
        "layer_norm_value_off",
        "Value layer norm off",
        "Isolates the effect of layer normalisation in the history-value network.",
        policy_network_layer_norm=True,
        regret_network_layer_norm=True,
        value_network_layer_norm=False,
    ),
]

