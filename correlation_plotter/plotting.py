"""Makes the Top 15 chart with historical settings unchanged."""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_top_correlations(top15_corr, city, horizon, target_col_generated, keywords_to_skip):
    """Makes the Top 15 chart with historical settings unchanged."""
    plt.figure(figsize=(12, 7))
    colors = ["#4C72B0" if val >= 0 else "#C44E52" for val in top15_corr.values]
    sns.barplot(
        x=top15_corr.values,
        y=top15_corr.index,
        palette=colors,
        hue=top15_corr.index,
        legend=False,
    )
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
    plt.title(
        f"Top 15 Correlated Features | City: {city} | Horizon: H+{horizon}\n"
        f"Analyzed target: {target_col_generated} (Exclusions: {keywords_to_skip})",
        fontsize=12,
        fontweight='bold',
        pad=12,
    )
    plt.xlabel("Correlation coefficient (Pearson)", fontsize=10)
    plt.ylabel("Features", fontsize=10)

    for i, val in enumerate(top15_corr.values):
        ha = "left" if val >= 0 else "right"
        offset = 0.005 if val >= 0 else -0.005
        plt.text(
            val + offset,
            i,
            f"{val:.3f}",
            va='center',
            fontsize=9,
            fontweight='bold',
            ha=ha,
        )
    plt.tight_layout()
    plt.show()


__all__ = ["plot_top_correlations"]
