"""CLI entry point for ESCHER activation-function sweeps."""

from __future__ import annotations

from experiments.kuhn_poker.escher_single_seed_variant_runner import main as run_main

from . import config


def main(argv=None) -> int:
    return run_main(
        config,
        argv,
        description="Run a single-seed ESCHER activation-function sweep.",
        output_root="outputs/activation_sweeps",
        progress_label="Activation variants",
        final_plot_title="ESCHER activation-function sweep: final exploitability",
        unknown_label="activation",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

