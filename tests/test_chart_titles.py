from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import escher_poker  # noqa: F401
from escher_poker.chart_titles import (
    CHART_TITLE_PREFIX,
    format_chart_title,
    game_variant_label,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TITLE_SOURCE_DIRS = (
    REPO_ROOT / "escher_poker",
    REPO_ROOT / "experiments" / "kuhn_poker",
)


def test_format_chart_title_adds_algorithm_and_poker_prefix():
    assert (
        format_chart_title("ESCHER checkpoint exploitability over training")
        == f"{CHART_TITLE_PREFIX}checkpoint exploitability over training"
    )
    assert (
        format_chart_title("Kuhn Poker ESCHER: Exploitability Across Seeds")
        == f"{CHART_TITLE_PREFIX}Exploitability Across Seeds"
    )
    assert (
        format_chart_title("Exploitability by iteration", game_name="leduc_poker")
        == "ESCHER - Leduc Poker - Exploitability by iteration"
    )
    assert game_variant_label("kuhn_poker") == "Kuhn"


def test_format_chart_title_is_idempotent():
    title = f"{CHART_TITLE_PREFIX}Exploitability Across Seeds"
    assert format_chart_title(title) == title


def test_matplotlib_axes_titles_are_prefixed():
    fig, ax = plt.subplots()
    try:
        ax.set_title("ESCHER learning-rate schedules")
        assert ax.get_title() == f"{CHART_TITLE_PREFIX}learning-rate schedules"
    finally:
        plt.close(fig)


def test_matplotlib_title_calls_use_standard_formatter():
    offenders = []
    for source_dir in TITLE_SOURCE_DIRS:
        for path in source_dir.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "set_title"
                ):
                    continue
                if any(
                    isinstance(child, ast.Call)
                    and (
                        (
                            isinstance(child.func, ast.Name)
                            and child.func.id == "format_plot_title"
                        )
                        or (
                            isinstance(child.func, ast.Attribute)
                            and child.func.attr == "format_plot_title"
                        )
                    )
                    for child in ast.walk(node)
                ):
                    continue
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert not offenders, "Raw set_title calls must use format_plot_title: " + ", ".join(offenders)
