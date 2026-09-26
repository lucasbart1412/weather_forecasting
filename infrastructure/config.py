"""
================================================================================
config.py - Centralized Configuration for WeatherMaster Ultimate
================================================================================
Automatic environment detection (Colab/Local/Kaggle) and hardware (CPU/GPU).
This module does not depend on any heavy ML library.
"""

import logging
import os
import shlex
import subprocess
import sys
import warnings
from logging import INFO, FileHandler, StreamHandler, basicConfig, getLogger
from pathlib import Path

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================


def detect_environment() -> str:
    """
    Automatically detects the runtime environment.

    Returns:
        str: 'colab', 'kaggle', or 'local'
    """
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle").exists():
        return "kaggle"
    elif "COLAB_GPU" in os.environ or Path("/content").exists():
        return "colab"
    else:
        return "local"


def detect_gpu() -> bool:
    """
    Detects the presence of a robustly available CUDA GPU.
    Never fails even if torch or nvidia-smi are not installed.

    Returns:
        bool: True if CUDA GPU available, False otherwise
    """
    # Method 1: via nvidia-smi (most reliable)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError, Exception):
        pass

    # Method 2: via PyTorch (if available and functional)
    try:
        import torch

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return True
    except (ImportError, OSError, RuntimeError, AttributeError, Exception):
        pass

    # Method 3: CUDA variable environment
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return True

    return False


def get_available_ram_gb() -> float:
    """
    Recovers the available RAM in GB.

    Returns:
        float: RAM available in gigabytes (0.0 if psutil unavailable)
    """
    if not HAS_PSUTIL:
        return 0.0
    try:
        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        return 0.0


# ============================================================================
# GLOBAL SETUP
# ============================================================================


class WeatherConfig:
    """
    Centralized configuration of the WeatherMaster application.

    Attributes:
        ENVIRONMENT: Type of environment ('colab'/'kaggle'/'local')
        HAS_GPU: Presence of a CUDA GPU
        AVAILABLE_RAM_GB: RAM available in GB
        BASE_DIR: Base directory for data files/templates
        Geo_CACHE_FILE: JSON cache file for geocoding
        MARINE_API_URL: Open-Meteo Marine API URL
        HORIZONS: List of forecast horizons in hoursTARGETS: TARGET weather variables (9 targets)LEVELS: Atmospheric pressure levels (hPa)TIDE_FEATURES: Features used for tidal prediction
        API_VARS: Variables to be retrieved via the Open-Meteo API
        ERA5_LAND_BLACKLIST: Missing variables from ERA5-Land
        MAPPING_VARS: API Mapping -> internal names
    """

    # Automatic detection when loading the module
    ENVIRONMENT = detect_environment()
    HAS_GPU = detect_gpu()
    AVAILABLE_RAM_GB = get_available_ram_gb()

    # Adaptive paths according to the environment
    if ENVIRONMENT == "kaggle":
        BASE_DIR = Path("/kaggle/working")
    elif ENVIRONMENT == "colab":
        BASE_DIR = Path("/content")
    else:
        BASE_DIR = Path(__file__).resolve().parent.parent / "binary"


    # Cache Files
    GEO_CACHE_FILE = BASE_DIR / "geo_cache.json"

    # API URLs
    MARINE_API_URL = "https://marine-api.open-meteo.com/v1/marine"

    # Forecast Horizons (hours)
    HORIZONS = [
        1,
        2,
        3,
        4,
        6,
        12,
        18,
        24,
        30,
        36,
        42,
        48,
        54,
        60,
        66,
        72,
        84,
        96,
        108,
        120,
        132,
        144,
        156,
        168,
    ]

    # Target weather variables
    TARGETS = [
        "temp",
        "hum",
        "press",
        "wind",
        "precip",
        "cloud",
        "dew",
        "gusts",
        "tide",
    ]

    # Atmospheric pressure levels (hPa)
    LEVELS = ["1000hPa", "850hPa", "700hPa", "500hPa", "250hPa"]

    # Features for tides
    TIDE_FEATURES = ["wind", "wind_dir", "gusts", "temp", "press", "surf_press"]

    # Variables to be retrieved via Open-Meteo API
    API_VARS = (
        [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "pressure_msl",
            "precipitation",
            "wind_speed_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "cloud_cover",
            "cape",
            "convective_inhibition",
            "geopotential_height_500hPa",
            "geopotential_height_1000hPa",
            "surface_pressure",
            "soil_temperature_0_to_7cm",
            "soil_moisture_0_to_7cm",
            "shortwave_radiation",
            "snowfall",
        ]
        + [f"temperature_{level}" for level in LEVELS]
        + [f"relative_humidity_{level}" for level in LEVELS]
        + [f"wind_speed_{level}" for level in ["250hPa", "850hPa"]]
    )

    # Variables not available in ERA5-Land
    ERA5_LAND_BLACKLIST = [
        "geopotential_height_500hPa",
        "geopotential_height_1000hPa",
        "wind_speed_250hPa",
        "wind_speed_850hPa",
        "temperature_1000hPa",
        "temperature_850hPa",
        "temperature_700hPa",
        "temperature_500hPa",
        "temperature_250hPa",
        "relative_humidity_1000hPa",
        "relative_humidity_850hPa",
        "relative_humidity_700hPa",
        "relative_humidity_500hPa",
        "relative_humidity_250hPa",
        "cape",
        "convective_inhibition",
    ]

    # API Mapping -> Internal Names
    MAPPING_VARS = {
        "temperature_2m": "temp",
        "temperature_2m_era5_land": "temp",
        "relative_humidity_2m_era5_land": "hum",
        "dew_point_2m_era5_land": "dew",
        "relative_humidity_2m": "hum",
        "dew_point_2m": "dew",
        "pressure_msl": "press",
        "pressure_msl_era5_land": "press",
        "precipitation": "precip",
        "precipitation_era5_land": "precip",
        "cloud_cover": "cloud",
        "cloud_cover_era5_land": "cloud",
        "wind_speed_10m": "wind",
        "wind_speed_10m_era5_land": "wind",
        "wind_direction_10m": "wind_dir",
        "wind_direction_10m_era5_land": "wind_dir",
        "wind_gusts_10m": "gusts",
        "wind_gusts_10m_era5_land": "gusts",
        "snowfall": "snow",
        "snowfall_era5_land": "snow",
        "cape": "cape",
        "cape_era5_land": "cape",
        "convective_inhibition": "cin",
        "convective_inhibition_era5_land": "cin",
        "surface_pressure": "surf_press",
        "surface_pressure_era5_land": "surf_press",
        "shortwave_radiation": "sw_rad",
        "shortwave_radiation_era5_land": "sw_rad",
        "wind_speed_2m": "w2m",
        "wind_speed_2m_era5_land": "w2m",
        "soil_temperature_0_to_7cm": "soil_temp",
        "soil_temperature_0_to_7cm_era5_land": "soil_temp",
        "soil_moisture_0_to_7cm": "soil_moist",
        "soil_moisture_0_to_7cm_era5_land": "soil_moist",
        "geopotential_height_500hPa": "geo_500",
        "geopotential_height_1000hPa": "geo_1000",
        "wind_speed_250hPa": "jet_wind",
        "wind_speed_850hPa": "wind_850",
        "temperature_1000hPa": "t1000",
        "temperature_850hPa": "t850",
        "temperature_700hPa": "t700",
        "temperature_500hPa": "t500",
        "temperature_250hPa": "t250",
        "relative_humidity_1000hPa": "h1000",
        "relative_humidity_850hPa": "h850",
        "relative_humidity_700hPa": "h700",
        "relative_humidity_500hPa": "h500",
        "relative_humidity_250hPa": "h250",
    }

    @staticmethod
    def wrap_lon(lon: float) -> float:
        """
        Normalizes a longitude in the interval (-180, 180], keeping
        exactly the limits already valid (e.g. 180.0 remains 180.0).

        Args:
            lon: Longitude in degrees

        Returns:
            Normalized longitude in (-180, 180]
        """
        original_sign_negative = lon < 0
        lon = lon % 360  # returns to [0, 360)
        if lon > 180:
            lon -= 360
        elif lon == 180 and original_sign_negative:
            # Special case: -180.0 (and its equivalents, e.g.: -540.0) must
            # remain negative rather than being reduced to 180.0
            lon = -180.0
        return lon


# ============================================================================
# LOGGER CONFIGURATION
# ============================================================================


def setup_logger() -> logging.Logger:
    """
    Configures and returns the application's global logger.

    Returns:
        Logger configured with handlers file and console
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    log_date_fmt = "%H:%M:%S"

    # Adaptive Log File Path
    if WeatherConfig.ENVIRONMENT == "kaggle":
        log_file_path = "/kaggle/working/weather_output.txt"
    elif WeatherConfig.ENVIRONMENT == "colab":
        log_file_path = "/content/weather_output.txt"
    else:
        log_file_path = str(WeatherConfig.BASE_DIR / "weather_output.txt")

    # Creation of handlers
    WeatherConfig.BASE_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = FileHandler(log_file_path, mode="w", encoding="utf-8")
    stream_handler = StreamHandler()

    # Global settings
    basicConfig(
        level=INFO,
        format=log_format,
        datefmt=log_date_fmt,
        handlers=[file_handler, stream_handler],
        force=True,
    )

    logger = getLogger("WeatherMaster")

    # Automatic capture of fatal errors
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.excepthook(exc_type, exc_value, exc_traceback)
            return
        logger.critical(
            'A fatal error has occurred:',
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = handle_exception

    # System Information
    logger.info(f"🌍 Environment detected: {WeatherConfig.ENVIRONMENT}")
    logger.info(f"🎮 GPU available: {WeatherConfig.HAS_GPU}")
    logger.info(f"💾 Available RAM: {WeatherConfig.AVAILABLE_RAM_GB:.1f} GB")

    return logger


# Initialization of the logger on first import
logger = setup_logger()

# Filters to remove common warnings
try:
    from pandas import errors as pd_errors

    warnings.filterwarnings("ignore", category=pd_errors.PerformanceWarning)
except ImportError:
    pass

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*torch.distributed.reduce_op.*"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*X does not have valid feature names.*"
)

# Memory configuration to avoid fragmentation problems
os.environ.setdefault("MALLOC_ARENA_MAX", "2")


def trigger_optional_vpn(command: str | None = None) -> None:
    """Run an optional VPN command only when explicitly configured.

    This keeps network recovery behavior opt-in rather than executing a VPN on
    every HTTP issue from the data fetch layer.
    """
    vpn_command = command or os.getenv("WEATHER_VPN_CONNECT_CMD")
    if not vpn_command:
        return

    try:
        subprocess.run(
            shlex.split(vpn_command),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive best effort
        logger.warning(f"Optional VPN command failed: {exc}")


__all__ = ["WeatherConfig", "logger", "trigger_optional_vpn", "setup_logger"]