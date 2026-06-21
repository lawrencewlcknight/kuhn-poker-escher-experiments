"""CLI entry point for ESCHER separate-network architecture sweeps."""

from __future__ import annotations

from experiments.kuhn_poker.escher_single_seed_variant_runner import main as run_main

from . import config


def main(argv=None) -> int:
    return run_main(
        config,
        argv,
        description="Run a single-seed ESCHER separate-network architecture sweep.",
        output_root="outputs/separate_network_architecture_sweeps",
        progress_label="Separate-network variants",
        final_plot_title="ESCHER separate-network architecture sweep: final exploitability",
        unknown_label="separate-network architecture",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

