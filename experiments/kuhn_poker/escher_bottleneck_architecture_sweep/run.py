"""CLI entry point for ESCHER bottleneck architecture sweeps."""

from __future__ import annotations

from experiments.kuhn_poker.escher_single_seed_variant_runner import main as run_main

from . import config


def main(argv=None) -> int:
    return run_main(
        config,
        argv,
        description="Run a single-seed ESCHER bottleneck architecture sweep.",
        output_root="outputs/bottleneck_architecture_sweeps",
        progress_label="Bottleneck variants",
        final_plot_title="ESCHER bottleneck architecture sweep: final exploitability",
        unknown_label="bottleneck architecture",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

