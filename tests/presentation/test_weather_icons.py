from presentation.weather_icons import get_weather_icon


def test_weather_icon_renderer_is_available():
    assert get_weather_icon(None, 0) == "☀️"
    assert get_weather_icon(0, 5) == "⛈️"
