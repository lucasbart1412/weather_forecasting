"""Readable correlation chart helpers."""

import matplotlib.pyplot as plt


def plot_top_correlations(top15_corr, city, horizon, target_col_generated, keywords_to_skip):
    """Display and return a signed horizontal bar chart of the strongest correlations."""
    values = top15_corr.to_numpy()
    labels = top15_corr.index.astype(str)
    colors = ["#4C72B0" if value >= 0 else "#C44E52" for value in values]

    figure, axis = plt.subplots(figsize=(12, max(5, 0.42 * len(values))))
    positions = range(len(values))
    axis.barh(positions, values, color=colors, height=0.72, zorder=3)
    axis.set_yticks(list(positions))
    axis.set_yticklabels(labels)
    axis.axvline(0, color="#333333", linestyle="--", linewidth=1, zorder=2)
    axis.xaxis.grid(True, color="#D9DEE7", linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)

    max_abs_value = max((abs(float(value)) for value in values), default=0.0)
    margin = max(0.04, max_abs_value * 0.16)
    axis.set_xlim(-max_abs_value - margin, max_abs_value + margin)
    for position, value in enumerate(values):
        value = float(value)
        axis.annotate(
            f"{value:.3f}",
            xy=(value, position),
            xytext=(5 if value >= 0 else -5, 0),
            textcoords="offset points",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
            fontweight="bold",
        )

    axis.set_title(
        f"Top correlated features | {city} | Horizon H+{horizon}\n"
        f"Target: {target_col_generated} | Exclusions: {keywords_to_skip}",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    axis.set_xlabel("Pearson correlation coefficient")
    axis.set_ylabel("Feature")
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    figure.tight_layout()
    plt.show()
    return figure


__all__ = ["plot_top_correlations"]
