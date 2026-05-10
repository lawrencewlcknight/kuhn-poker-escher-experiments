"""Configuration for the ESCHER solver-parameter random search."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from experiments.kuhn_poker.escher_multiseed_baseline.config import (
    DEFAULT_CONFIG as BASELINE_DEFAULT_CONFIG,
)
from escher_poker.hyperparameter_search import sample_candidate_configs

BASELINE_VARIANT_ID = "baseline_escher_exp1"

DEFAULT_CONFIG = deepcopy(BASELINE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "kuhn_poker_escher_solver_parameter_random_search",
    "variant_id": BASELINE_VARIANT_ID,
    "expl": 1.0,
    "val_expl": 0.01,
    "importance_sampling": True,
    "importance_sampling_threshold": 100.0,
    "clear_value_buffer": True,
    "val_bootstrap": False,
    "use_balanced_probs": False,
    "val_op_prob": 0.0,
    "all_actions": True,
})

SCREENING_ITERATIONS = 60
SCREENING_EVALUATION_INTERVAL = 10
SCREENING_SEEDS = [1234, 2025]
N_RANDOM_CANDIDATES = 8

CONFIRMATION_ITERATIONS = 130
CONFIRMATION_EVALUATION_INTERVAL = 5
CONFIRMATION_SEEDS = [1234, 2025, 31415]
CONFIRMATION_TOP_K = 3

OPTIONAL_SCREENING_SEEDS_3 = [1234, 2025, 31415]
OPTIONAL_CONFIRMATION_SEEDS_5 = [1234, 2025, 31415, 27182, 16180]
OPTIONAL_CONFIRMATION_SEEDS_10 = [
    1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001,
]

RANDOM_SEARCH_SEED = 314159

SEARCH_SPACE = {
    "learning_rate": [3e-4, 5e-4, 1e-3, 2e-3],
    "num_traversals": [250, 500, 750, 1000],
    "num_val_fn_traversals": [250, 500, 750, 1000],
    "regret_network_train_steps": [100, 200, 400, 800],
    "value_network_train_steps": [100, 200, 400, 800],
    "policy_network_train_steps": [500, 1000, 1500, 2000],
    "policy_network_layers": [(128, 128), (256, 128), (256, 256), (512, 256)],
    "regret_network_layers": [(128, 128), (256, 128), (256, 256), (512, 256)],
    "value_network_layers": [(128, 128), (256, 128), (256, 256), (512, 256)],
    "batch_size_regret": [128, 256, 512],
    "batch_size_value": [128, 256, 512],
    "batch_size_average_policy": [5000, 10000],
    "memory_capacity": [int(5e4), int(1e5), int(5e5)],
    "reinitialize_regret_networks": [True, False],
    "reinitialize_value_network": [True, False],
    "expl": [0.5, 1.0],
    "val_expl": [0.01, 0.05, 0.1],
    "importance_sampling": [True, False],
    "importance_sampling_threshold": [10.0, 50.0, 100.0],
    "clear_value_buffer": [True, False],
    "val_bootstrap": [False, True],
    "use_balanced_probs": [False, True],
    "val_op_prob": [0.0, 0.1, 0.25],
    "all_actions": [True, False],
}


def build_screening_configs(
    base_config: Dict[str, Any],
    n_random_candidates: int = N_RANDOM_CANDIDATES,
    random_search_seed: int = RANDOM_SEARCH_SEED,
) -> List[Dict[str, Any]]:
    """Build baseline plus randomly sampled solver-parameter configs."""
    random_candidates = sample_candidate_configs(
        base_config,
        SEARCH_SPACE,
        n_random_candidates,
        random_search_seed,
        variant_id_prefix="solver_random",
    )
    return [dict(base_config)] + random_candidates


def parse_seeds(seed_string: str | None, default: List[int]) -> List[int]:
    if not seed_string:
        return list(default)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]
