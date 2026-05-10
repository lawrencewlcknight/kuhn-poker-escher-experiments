"""Configuration for the ESCHER learning-rate schedule ablation."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

from experiments.kuhn_poker.escher_multiseed_baseline.config import (
    DEFAULT_CONFIG as BASELINE_DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    DEVELOPMENT_SEEDS_3,
    DEVELOPMENT_SEEDS_5,
)

DEFAULT_CONFIG = deepcopy(BASELINE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "kuhn_poker_escher_lr_schedule_ablation",
    "learning_rate_end": 1e-4,
    "learning_rate_schedule": "constant",
    "learning_rate_decay_rate": 1.0,
    "learning_rate_warmup_iterations": 0,
})

BASELINE_SCHEDULE_ID = "constant_baseline_escher"

SCHEDULE_CONFIGS = [
    {
        "schedule": BASELINE_SCHEDULE_ID,
        "schedule_label": "Constant baseline",
        "learning_rate_schedule": "constant",
        "learning_rate_end": DEFAULT_CONFIG["learning_rate"],
        "learning_rate_decay_rate": 1.0,
        "learning_rate_warmup_iterations": 0,
    },
    {
        "schedule": "cosine_decay_to_10pct",
        "schedule_label": "Cosine decay to 10%",
        "learning_rate_schedule": "cosine_decay",
        "learning_rate_end": DEFAULT_CONFIG["learning_rate_end"],
        "learning_rate_decay_rate": 0.1,
        "learning_rate_warmup_iterations": 0,
    },
]

OPTIONAL_EXTRA_SCHEDULES = [
    {
        "schedule": "linear_decay_to_10pct",
        "schedule_label": "Linear decay to 10%",
        "learning_rate_schedule": "linear_decay",
        "learning_rate_end": DEFAULT_CONFIG["learning_rate_end"],
        "learning_rate_decay_rate": 0.1,
        "learning_rate_warmup_iterations": 0,
    },
    {
        "schedule": "step_decay_halfway_to_10pct",
        "schedule_label": "Step decay halfway",
        "learning_rate_schedule": "step_decay",
        "learning_rate_end": DEFAULT_CONFIG["learning_rate_end"],
        "learning_rate_decay_rate": 0.1,
        "learning_rate_warmup_iterations": 0,
    },
]

DEFAULT_LR_SCHEDULE_SEEDS = DEVELOPMENT_SEEDS_3
EXTENDED_SEEDS_5 = DEVELOPMENT_SEEDS_5
EXTENDED_SEEDS_10 = DEFAULT_SEEDS


def parse_seeds(seed_string: str | None, default: List[int] | None = None) -> List[int]:
    if not seed_string:
        return list(DEFAULT_LR_SCHEDULE_SEEDS if default is None else default)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


def schedule_lookup(include_optional: bool = False) -> Dict[str, Dict]:
    configs = list(SCHEDULE_CONFIGS)
    if include_optional:
        configs.extend(OPTIONAL_EXTRA_SCHEDULES)
    return {config["schedule"]: dict(config) for config in configs}


def parse_schedule_ids(value: str | None, include_optional: bool = False) -> List[str]:
    lookup = schedule_lookup(include_optional=include_optional)
    if not value:
        return [config["schedule"] for config in SCHEDULE_CONFIGS]
    schedule_ids = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [schedule_id for schedule_id in schedule_ids if schedule_id not in lookup]
    if unknown:
        raise ValueError(f"Unknown schedule id(s): {unknown}")
    return schedule_ids

