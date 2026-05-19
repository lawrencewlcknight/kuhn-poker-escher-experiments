"""Default configuration for the aligned Kuhn poker ESCHER baseline."""

from __future__ import annotations

from escher_poker.constants import EXPLOITABILITY_THRESHOLD, KUHN_AVERAGE_POLICY_VALUE_TARGET

DEFAULT_CONFIG = {
    "experiment_name": "kuhn_poker_escher_multiseed_baseline",
    "game_name": "kuhn_poker",
    # Kuhn poker is small, so the default thesis baseline should be runnable on
    # modest cloud machines. Earlier notebook-aligned defaults used wider
    # networks and many supervised update steps; those were expensive enough to
    # crash long-running jobs. These defaults keep the experiment structure but
    # make the solver deliberately lightweight.
    "num_iterations": 80,
    "num_traversals": 150,
    "num_val_fn_traversals": 150,
    "check_exploitability_every": 10,
    "policy_network_layers": (64, 64),
    "regret_network_layers": (64, 64),
    "value_network_layers": (64, 64),
    "learning_rate": 1e-3,
    "batch_size_regret": 128,
    "batch_size_value": 128,
    "batch_size_average_policy": 2_048,
    "memory_capacity": int(5e4),
    "policy_network_train_steps": 200,
    "regret_network_train_steps": 50,
    "value_network_train_steps": 50,
    "compute_exploitability": True,
    "reinitialize_regret_networks": True,
    "reinitialize_value_network": True,
    "save_policy_weights": False,
    "save_final_checkpoints": False,
    "train_device": "cpu",
    "infer_device": "cpu",
    "verbose": False,
    "exploitability_threshold": EXPLOITABILITY_THRESHOLD,
    "average_policy_value_target": KUHN_AVERAGE_POLICY_VALUE_TARGET,
}

DEFAULT_SEEDS = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]
DEVELOPMENT_SEEDS_3 = [1234, 2025, 31415]
DEVELOPMENT_SEEDS_5 = [1234, 2025, 31415, 27182, 16180]
