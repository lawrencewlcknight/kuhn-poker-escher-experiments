"""Chart-title helpers for Kuhn poker ESCHER outputs."""

from __future__ import annotations

from functools import wraps
import re
from typing import Any


ALGORITHM_VARIANT_LABEL = "ESCHER"
POKER_VARIANT_LABEL = "Kuhn"
CHART_TITLE_PREFIX = f"{ALGORITHM_VARIANT_LABEL} - {POKER_VARIANT_LABEL} - "


def game_variant_label(game_name: str = "kuhn_poker") -> str:
    """Return the thesis-facing poker-game label used in chart titles."""
    normalized = str(game_name).strip().lower()
    if normalized in {"kuhn", "kuhn_poker", "kuhn poker"}:
        return POKER_VARIANT_LABEL
    return normalized.replace("_", " ").title()


def _chart_title_prefix(
    *,
    algorithm_variant: str = ALGORITHM_VARIANT_LABEL,
    poker_variant: str | None = None,
    game_name: str = "kuhn_poker",
) -> str:
    algorithm = str(algorithm_variant).strip() or ALGORITHM_VARIANT_LABEL
    poker = str(poker_variant or game_variant_label(game_name)).strip()
    return f"{algorithm} - {poker} - "


def format_chart_title(
    title: Any,
    *,
    algorithm_variant: str = ALGORITHM_VARIANT_LABEL,
    poker_variant: str | None = None,
    game_name: str = "kuhn_poker",
) -> Any:
    """Return a chart title with the repository-standard prefix."""
    if not isinstance(title, str):
        return title

    prefix = _chart_title_prefix(
        algorithm_variant=algorithm_variant,
        poker_variant=poker_variant,
        game_name=game_name,
    )
    text = title.strip()
    if text.startswith(prefix):
        return text

    text = re.sub(r"\bKuhn\s+Poker\b", "", text, flags=re.IGNORECASE)
    text = re.sub(rf"\b{re.escape(str(algorithm_variant))}\b", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*:[ \t]*", ": ", text)
    text = re.sub(r"[ \t]+-[ \t]+", " - ", text)
    text = re.sub(r"^[\s:;-]+", "", text)
    text = text.strip(" \t:-")

    if not text:
        return prefix.rstrip(" -")
    return f"{prefix}{text}"


def format_plot_title(
    title: Any,
    *,
    algorithm_variant: str = ALGORITHM_VARIANT_LABEL,
    poker_variant: str | None = None,
    game_name: str = "kuhn_poker",
) -> Any:
    """Compatibility wrapper for plot helpers that format titles explicitly."""
    return format_chart_title(
        title,
        algorithm_variant=algorithm_variant,
        poker_variant=poker_variant,
        game_name=game_name,
    )


def install_chart_title_prefix() -> None:
    """Patch Matplotlib Axes titles so generated charts use the standard prefix."""
    try:
        from matplotlib.axes import Axes
    except Exception:
        return

    current_set_title = Axes.set_title
    if getattr(current_set_title, "_escher_chart_title_prefix_installed", False):
        return

    @wraps(current_set_title)
    def prefixed_set_title(self: Any, label: Any, *args: Any, **kwargs: Any) -> Any:
        return current_set_title(self, format_chart_title(label), *args, **kwargs)

    prefixed_set_title._escher_chart_title_prefix_installed = True  # type: ignore[attr-defined]
    Axes.set_title = prefixed_set_title
