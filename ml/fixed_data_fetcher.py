"""
inference.py
 MLOps inference layer for WeatherMaster.
Isolates API patches and vector inference by batch
without relying on a monolithic forecast script.
"""

from datetime import datetime
from io import StringIO
from random import uniform
from time import sleep

import pandas as pd
from requests import Session

from data_fetcher.data_fetcher import DataFetcher
from infrastructure.config import WeatherConfig, logger
from ml.predictor import WeatherPredictor


class FixedDataFetcher(DataFetcher):
    """
    Adapt for DataFetcher.

    Fixes the Open-Meteo API bug where forecast_days=0 excluded the current day.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep the session injectable so it remains mockable via 'inference.Session'.
        self.session = Session()

    def _fetch_single_point(self, name, lat, lon, geo, is_archive):
        base_url = (
            "https://archive-api.open-meteo.com/v1/archive"
            if is_archive
            else "https://api.open-meteo.com/v1/forecast"
        )

        def _execute_query(hourly_vars, model_name):
            params = {
                "latitude": str(lat),
                "longitude": str(lon),
                "hourly": ",".join(hourly_vars),
                "timezone": geo["tz"],
                "format": "csv",
            }
            if is_archive:
                params.update(
                    {
                        "models": model_name,
                        "start_date": "2010-01-01",
                        "end_date": (datetime.now() - pd.Timedelta(days=5)).strftime(
                            "%Y-%m-%d"
                        ),
                    }
                )
            else:
                params.update({"past_days": 7, "forecast_days": 2})

            for attempt in range(5):
                try:
                    headers = {
                        "User-Agent": self.ua.random
                        if self.ua is not None
                        else "Mozilla/5.0 (compatible; WeatherMaster/1.0)"
                    }
                    response = self.session.get(
                        base_url, params=params, headers=headers, timeout=15
                    )
                    if response.status_code == 200:
                        lines = response.text.strip().split("\n")
                        idx = next(
                            i
                            for i, line in enumerate(lines)
                            if line.lower().startswith("time")
                        )
                        df = pd.read_csv(StringIO("\n".join(lines[idx:])))
                        df.columns = [
                            c.split(" ")[0].strip().lower() for c in df.columns
                        ]
                        df = df.rename(columns={"time": "date"})
                        rename_dict = {
                            k.lower(): v for k, v in WeatherConfig.MAPPING_VARS.items()
                        }
                        df = df.rename(columns=rename_dict)
                        df = df.loc[:, ~df.columns.duplicated()]
                        return df
                    elif response.status_code == 400:
                        logger.error(
                            f"API error 400 for {name} ({model_name}) : {response.text[:200]}"
                        )
                        break
                    else:
                        logger.warning(
                            f"Unexpected response {response.status_code} for {name} ({model_name}). Attempt {attempt + 1}/5."
                        )
                        sleep(2**attempt + uniform(0, 5))
                except Exception as e:
                    logger.error(
                        f"Exception during the request for {name} ({model_name}) : {str(e)[:100]}. Attempt {attempt + 1}/5."
                    )
                    sleep(2**attempt + uniform(0, 5))
            return None

        if is_archive and (name == "center" or "loc" in name):
            return _execute_query(WeatherConfig.API_VARS, "era5")
        else:
            model_default = "era5" if is_archive else "best_match"
            return _execute_query(WeatherConfig.API_VARS, model_default)


class BatchWeatherPredictor(WeatherPredictor):
    """
    WeatherPredictor extension for batch matrix inference.
    Reduces inference time from 168 sequential calls to 3 vector calls.
    """

