"""CLI entry point for ESCHER policy-network width sweeps."""

from __future__ import annotations

from experiments.kuhn_poker.escher_single_seed_variant_runner import main as run_main

from . import config


def main(argv=None) -> int:
    return run_main(
        config,
        argv,
        description="Run a single-seed ESCHER policy-network width sweep.",
        output_root="outputs/policy_network_width_sweeps",
        progress_label="Policy-width variants",
        final_plot_title="ESCHER policy-network width sweep: final exploitability",
        unknown_label="policy-network width",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

