"""CLI entry point for ESCHER shared-trunk/action-head sweeps."""

from __future__ import annotations

from experiments.kuhn_poker.escher_single_seed_variant_runner import main as run_main

from . import config


def main(argv=None) -> int:
    return run_main(
        config,
        argv,
        description="Run a single-seed ESCHER shared-trunk/action-head sweep.",
        output_root="outputs/shared_trunk_head_sweeps",
        progress_label="Action-head variants",
        final_plot_title="ESCHER shared-trunk/action-head sweep: final exploitability",
        unknown_label="shared-trunk/action-head",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

