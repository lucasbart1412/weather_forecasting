"""Historical facade of the correlation plotter."""

from correlation_plotter.analysis_impl import analyze_and_plot_correlations, main_loop

__all__ = ["analyze_and_plot_correlations", "main_loop"]


if __name__ == "__main__":
    main_loop()
