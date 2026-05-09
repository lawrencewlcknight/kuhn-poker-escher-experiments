"""Default configuration for the aligned Kuhn poker ESCHER baseline."""

from __future__ import annotations

from escher_poker.constants import EXPLOITABILITY_THRESHOLD

DEFAULT_CONFIG = {
    "experiment_name": "kuhn_poker_escher_multiseed_baseline",
    "game_name": "kuhn_poker",
    # The exploratory notebook used 30 iterations plus a 100-iteration resumed run.
    # This baseline uses a single fixed 130-iteration run per seed to preserve the
    # approximate budget while removing checkpoint/resume as a confound.
    "num_iterations": 130,
    "num_traversals": 500,
    "num_val_fn_traversals": 500,
    "check_exploitability_every": 5,
    "policy_network_layers": (256, 128),
    "regret_network_layers": (256, 128),
    "value_network_layers": (256, 128),
    "learning_rate": 1e-3,
    "batch_size_regret": 256,
    "batch_size_value": 256,
    "batch_size_average_policy": 10_000,
    "memory_capacity": int(1e5),
    "policy_network_train_steps": 1_000,
    "regret_network_train_steps": 200,
    "value_network_train_steps": 200,
    "compute_exploitability": True,
    "reinitialize_regret_networks": True,
    "reinitialize_value_network": True,
    "save_policy_weights": False,
    "save_final_checkpoints": False,
    "train_device": "cpu",
    "infer_device": "cpu",
    "verbose": False,
    "exploitability_threshold": EXPLOITABILITY_THRESHOLD,
}

DEFAULT_SEEDS = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]
DEVELOPMENT_SEEDS_3 = [1234, 2025, 31415]
DEVELOPMENT_SEEDS_5 = [1234, 2025, 31415, 27182, 16180]
