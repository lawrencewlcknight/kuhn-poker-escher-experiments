"""CLI entry point for ESCHER layer-normalisation ablations."""

from __future__ import annotations

from experiments.kuhn_poker.escher_single_seed_variant_runner import main as run_main

from . import config


def main(argv=None) -> int:
    return run_main(
        config,
        argv,
        description="Run a single-seed ESCHER layer-normalisation ablation.",
        output_root="outputs/layer_norm_ablations",
        progress_label="Layer-norm variants",
        final_plot_title="ESCHER layer-normalisation ablation: final exploitability",
        unknown_label="layer-normalisation",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

