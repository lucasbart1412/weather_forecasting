import matplotlib.pyplot as plt
import pandas as pd

from correlation_plotter.plotting import plot_top_correlations


def test_correlation_plotter_exposes_renderer():
    assert callable(plot_top_correlations)


def test_correlation_chart_labels_positive_and_negative_values(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    correlations = pd.Series([0.72, -0.41], index=["temperature", "humidity"])

    figure = plot_top_correlations(
        correlations,
        "Test City",
        24,
        "target_temp_h24",
        ["target_", "date"],
    )

    axis = figure.axes[0]
    assert len(axis.patches) == 2
    assert axis.get_xlabel() == "Pearson correlation coefficient"
    assert list(axis.get_yticklabels()[index].get_text() for index in range(2)) == [
        "temperature",
        "humidity",
    ]
    plt.close(figure)
