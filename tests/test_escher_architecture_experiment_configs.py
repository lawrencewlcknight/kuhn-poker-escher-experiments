"""Configuration checks for ESCHER architecture experiments."""

from __future__ import annotations

import importlib

from experiments.kuhn_poker.escher_variant_config_utils import make_variant_config


CONFIG_MODULES = [
    "experiments.kuhn_poker.escher_separate_network_architecture_sweep.config",
    "experiments.kuhn_poker.escher_regret_network_width_sweep.config",
    "experiments.kuhn_poker.escher_policy_network_width_sweep.config",
    "experiments.kuhn_poker.escher_layer_norm_ablation.config",
    "experiments.kuhn_poker.escher_activation_sweep.config",
    "experiments.kuhn_poker.escher_residual_mlp_sweep.config",
    "experiments.kuhn_poker.escher_bottleneck_architecture_sweep.config",
    "experiments.kuhn_poker.escher_shared_trunk_head_sweep.config",
]


def test_architecture_experiment_variant_ids_are_unique():
    for module_name in CONFIG_MODULES:
        module = importlib.import_module(module_name)
        ids = [variant["variant_id"] for variant in module.VARIANTS]

        assert ids
        assert len(ids) == len(set(ids))


def test_architecture_experiment_variants_are_configurable():
    for module_name in CONFIG_MODULES:
        module = importlib.import_module(module_name)
        variant_config = make_variant_config(
            module.DEFAULT_CONFIG,
            module.VARIANTS[0],
        )

        assert variant_config["variant_id"] == module.VARIANTS[0]["variant_id"]
        assert variant_config["total_policy_training_events_expected"] >= 1
        assert variant_config["policy_gradient_steps_expected"] >= 1
