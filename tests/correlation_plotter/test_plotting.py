from correlation_plotter.plotting import plot_top_correlations


def test_correlation_plotter_exposes_renderer():
    assert callable(plot_top_correlations)
