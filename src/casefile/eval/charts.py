"""Draw the chart of tokens used per evaluation case.

    python -m casefile.eval.charts

Reads reports/live_run_report.json and writes docs/tokens_per_case_light.png
and docs/tokens_per_case_dark.png, one for each GitHub theme. Needs
matplotlib, which is installed with the ``dev`` extra.

The same numbers are in the report itself and in the README's results table,
so the chart is never the only place a value can be read.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # draw to files; no display is needed

import matplotlib.pyplot as plt  # noqa: E402 - must follow the backend choice

from casefile.config import PROJECT_ROOT, REPORTS_DIR  # noqa: E402

REPORT_PATH = REPORTS_DIR / "live_run_report.json"
DOCS_DIR = PROJECT_ROOT / "docs"


@dataclass(frozen=True)
class Theme:
    """The colors of one rendering. Text never takes a series color."""

    name: str
    surface: str
    ink: str
    secondary_ink: str
    muted: str
    grid: str
    baseline: str
    straight: str  # cases that ran straight through
    sent_back: str  # cases in which the reviewer returned work


# The first two slots of a categorical palette checked for colour-blind
# separation; each theme uses the step chosen for its own surface.
THEMES = (
    Theme(
        "light", "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
        "#2a78d6", "#eb6834",
    ),
    Theme(
        "dark", "#1a1a19", "#ffffff", "#c3c2b7", "#898781", "#2c2c2a", "#383835",
        "#3987e5", "#d95926",
    ),
)  # fmt: skip


def draw(report: dict[str, Any], theme: Theme, path: Path) -> None:
    """Draw the chart in one theme and save it."""
    cases = report["cases"]
    ceiling = report["token_ceiling_per_case"]
    tokens = [case["tokens"] for case in cases]
    sent_back = ["sent_back" in case["events"] for case in cases]
    largest = max(range(len(cases)), key=tokens.__getitem__)

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    figure, axes = plt.subplots(figsize=(11, 4.6), dpi=160)
    figure.patch.set_facecolor(theme.surface)
    axes.set_facecolor(theme.surface)

    # One colour per group, not per bar: the colour says how the case ran,
    # the height says what it cost.
    for flag, color, label in (
        (False, theme.straight, "ran straight through"),
        (True, theme.sent_back, "the reviewer sent work back"),
    ):
        positions = [i for i, value in enumerate(sent_back) if value is flag]
        if positions:
            axes.bar(
                positions,
                [tokens[i] for i in positions],
                width=0.64,
                color=color,
                linewidth=0,
                label=label,
            )

    axes.set_xticks(range(len(cases)), [case["case_id"] for case in cases])
    axes.tick_params(axis="x", colors=theme.muted, labelsize=7.5, length=0)
    axes.tick_params(axis="y", colors=theme.muted, labelsize=8.5, length=0)
    axes.yaxis.set_major_formatter(lambda value, _: f"{int(value):,}")
    axes.set_ylim(0, max(tokens) * 1.22)
    axes.set_xlim(-0.8, len(cases) - 0.2)
    axes.grid(axis="y", color=theme.grid, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(theme.baseline)

    # Only the extreme is labelled; every value is in the results table.
    axes.annotate(
        f"{tokens[largest]:,}",
        (largest, tokens[largest]),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=8.5,
        color=theme.ink,
    )
    figure.text(
        0.012, 0.955, "Tokens used per evaluation case",
        fontsize=13, fontweight="bold", color=theme.ink, va="top",
    )  # fmt: skip
    figure.text(
        0.012, 0.885,
        f"{report['model']}, all three agents. The safety ceiling of {ceiling:,} "
        f"tokens per case is {ceiling / max(tokens):.0f} times the largest case "
        "and far above this axis.",
        fontsize=9, color=theme.secondary_ink, va="top",
    )  # fmt: skip
    # Above the plot, where no gridline runs through it.
    legend = axes.legend(
        loc="lower left",
        bbox_to_anchor=(-0.005, 1.0),
        frameon=False,
        fontsize=8.5,
        ncols=2,
    )
    for text in legend.get_texts():
        text.set_color(theme.secondary_ink)

    figure.subplots_adjust(left=0.065, right=0.99, top=0.74, bottom=0.09)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=theme.surface)
    plt.close(figure)


def main() -> int:
    """Draw the chart in both themes. Returns an exit code."""
    if not REPORT_PATH.exists():
        print(
            f"{REPORT_PATH} does not exist. Run casefile.eval.run_live first.",
            file=sys.stderr,
        )
        return 1
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    for theme in THEMES:
        path = DOCS_DIR / f"tokens_per_case_{theme.name}.png"
        draw(report, theme, path)
        print(f"Wrote {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
