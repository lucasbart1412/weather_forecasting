import pandas as pd
import plotly.graph_objects as go

from presentation.dashboard_helpers import (
    chart_variable_options,
    filter_short_term_precipitation,
    plot_hourly,
)


def test_precipitation_is_only_available_in_short_term_chart_options():
    short_term_options = chart_variable_options(include_precipitation=True)
    extended_options = chart_variable_options(include_precipitation=False)

    assert "precip" in short_term_options.values()
    assert "precip" not in extended_options.values()


def test_short_term_precipitation_is_limited_to_18_hours():
    forecast = pd.DataFrame(
        {
            "horizon_h": [1, 18, 19, 24, 72, 168],
            "precip": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )

    filtered = filter_short_term_precipitation(forecast)

    assert filtered["horizon_h"].tolist() == [1, 18]


def test_precipitation_chart_is_rendered_when_forecast_is_dry():
    forecast = pd.DataFrame(
        {
            "time": pd.date_range("2026-10-14 12:00", periods=18, freq="h"),
            "precip": [0.0] * 18,
            "precip_prob": [0.0] * 18,
        }
    )

    figure = plot_hourly(forecast, "precip")

    assert isinstance(figure, go.Figure)
    assert [trace.name for trace in figure.data] == [
        "Precipitation (mm)",
        "Probability (%)",
    ]
    assert list(figure.data[0].y) == [0.0] * 18
