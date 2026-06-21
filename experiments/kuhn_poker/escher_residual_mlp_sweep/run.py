"""CLI entry point for ESCHER residual-MLP sweeps."""

from __future__ import annotations

from experiments.kuhn_poker.escher_single_seed_variant_runner import main as run_main

from . import config


def main(argv=None) -> int:
    return run_main(
        config,
        argv,
        description="Run a single-seed ESCHER residual-MLP sweep.",
        output_root="outputs/residual_mlp_sweeps",
        progress_label="Residual-MLP variants",
        final_plot_title="ESCHER residual-MLP sweep: final exploitability",
        unknown_label="residual-MLP",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

