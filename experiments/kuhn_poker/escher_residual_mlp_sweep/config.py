"""Configuration for ESCHER residual-MLP sweeps."""

from __future__ import annotations

from experiments.kuhn_poker.escher_architecture_base import (
    DEFAULT_SEED,
    make_default_config,
)

DEFAULT_CONFIG = make_default_config("kuhn_poker_escher_residual_mlp_sweep")


def _variant(variant_id, variant_label, variant_description, layers, residual_mode):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "policy_network_layers": layers,
        "regret_network_layers": layers,
        "value_network_layers": layers,
        "policy_network_residual_mode": residual_mode,
        "regret_network_residual_mode": residual_mode,
        "value_network_residual_mode": residual_mode,
    }


VARIANTS = [
    _variant(
        "reference_256_128_same_width",
        "Reference 256x128",
        "Experiment 13 reference; no skip is active because adjacent widths differ.",
        (256, 128),
        "same_width",
    ),
    _variant(
        "deep_plain_256_256_128",
        "Deep plain 256x256x128",
        "Deeper MLP with residual connections disabled.",
        (256, 256, 128),
        "none",
    ),
    _variant(
        "deep_same_width_256_256_128",
        "Deep same-width residual",
        "Activates a same-width skip on the repeated 256 layer.",
        (256, 256, 128),
        "same_width",
    ),
    _variant(
        "deep_projection_256_256_128",
        "Deep projection residual",
        "Uses projection residual blocks after the input layer.",
        (256, 256, 128),
        "projection",
    ),
    _variant(
        "constant_plain_256_256_256",
        "Constant-width plain",
        "Constant-width deep MLP with residuals disabled.",
        (256, 256, 256),
        "none",
    ),
    _variant(
        "constant_residual_256_256_256",
        "Constant-width residual",
        "Constant-width deep MLP with same-width residual blocks.",
        (256, 256, 256),
        "same_width",
    ),
]

