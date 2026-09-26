#!/usr/bin/env python3
"""
_runtime.py 
Module for managing the runtime environment of the weather application, including data fetching, feature engineering, and model training.
"""

# ML Libraries

from warnings import filterwarnings
import argparse
import ctypes
import gc
import os
import random
import sys
from pathlib import Path
from time import time, sleep
from typing import Dict, Optional, Tuple

from catboost import CatBoostRegressor
from joblib import dump as job_dump, load as job_load
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from numpy import (
    arange,
    ascontiguousarray,
    asarray,
    array,
    concatenate,
    cos,
    exp,
    float32,
    int32,
    isin,
    linspace,
    maximum,
    mean,
    minimum,
    nan,
    pi,
    sin,
    sort,
    sqrt,
    where,
    inf
)
from pandas import (
    DataFrame,
    Series,
    Timedelta,
    Timestamp,
    concat,
    errors as pd_errors,
    read_parquet,
    to_datetime,
    to_numeric,
)
from sklearn.linear_model import RidgeCV, TweedieRegressor
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from optuna import create_study, pruners
from optuna.exceptions import TrialPruned
from optuna_integration import LightGBMPruningCallback
from xgboost import XGBRegressor

# Local Imports

from data_fetcher.data_fetcher import DataFetcher
from feature_engineering.feature_engineering import FeatureEngineer
from infrastructure.config import WeatherConfig, logger
from infrastructure.geo_api import LocationManager


# ============================================================================
# DELETING WARNINGS
# ============================================================================
filterwarnings("ignore", category=pd_errors.PerformanceWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", message=".*X does not have valid feature names.*")