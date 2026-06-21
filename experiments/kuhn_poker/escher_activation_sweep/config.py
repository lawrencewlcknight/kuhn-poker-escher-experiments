"""Configuration for ESCHER activation-function sweeps."""

from __future__ import annotations

from experiments.kuhn_poker.escher_architecture_base import (
    DEFAULT_SEED,
    make_default_config,
)

DEFAULT_CONFIG = make_default_config("kuhn_poker_escher_activation_sweep")


def _activation_variant(activation: str, description: str):
    return {
        "variant_id": activation,
        "variant_label": activation,
        "variant_description": description,
        "policy_network_activation": activation,
        "regret_network_activation": activation,
        "value_network_activation": activation,
    }


VARIANTS = [
    _activation_variant("leakyrelu", "Experiment 13 reference activation."),
    _activation_variant("relu", "Plain ReLU activation."),
    _activation_variant("elu", "ELU activation, testing smoother negative responses."),
    _activation_variant("gelu", "GELU activation, testing smoother nonlinearities."),
    _activation_variant("swish", "Swish activation, testing smooth gated responses."),
    _activation_variant("tanh", "Tanh activation, testing bounded hidden representations."),
]

