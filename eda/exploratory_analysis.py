"""Generate an EDA report and plots for a weather archive Parquet file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "binary" / "full_grid_archive.parquet"
CORE_VARIABLES = [
    "temp",
    "hum",
    "dew",
    "press",
    "precip",
    "wind",
    "wind_dir",
    "gusts",
    "cloud",
    "surf_press",
    "soil_temp",
    "soil_moist",
    "sw_rad",
    "snow",
]
RELATIONSHIPS = [
    ("temp", "dew", "Air temperature and dew point"),
    ("wind", "gusts", "Mean wind and gusts"),
    ("temp", "soil_temp", "Air and soil temperature"),
    ("press", "surf_press", "Surface and sea-level pressure"),
    ("hum", "cloud", "Humidity and cloud cover"),
    ("precip", "hum", "Precipitation and humidity"),
]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and chronologically sort a parquet archive containing a date column."""
    if not path.is_file():
        raise FileNotFoundError(f"Archive not found: {path}")
    frame = pd.read_parquet(path)
    if "date" not in frame.columns:
        raise ValueError("The Parquet file must contain a 'date' column.")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.sort_values("date").reset_index(drop=True)


def _format_number(value: float) -> str:
    return "n.d." if pd.isna(value) else f"{value:.3f}"


def _log_recommendation(series: pd.Series, skewness: float) -> str:
    values = series.dropna()
    if values.empty or skewness <= 1:
        return "No priority log transform: skewness is moderate."
    if values.min() < 0:
        return "No direct log: negative values; consider Yeo-Johnson or a justified shift."

    zero_share = values.eq(0).mean()
    if zero_share >= 0.2:
        return (
            "Highly skewed with many zeros: separate occurrence from intensity; "
            "test log1p on strictly positive values."
        )
    return "Test log1p(x) to reduce skew; compare against the raw values."


def _make_distribution_plots(frame: pd.DataFrame, variables: list[str], output: Path) -> None:
    if not variables:
        return
    figure, axes = plt.subplots(len(variables), 2, figsize=(12, 3.0 * len(variables)))
    axes = np.atleast_2d(axes)
    for row, name in enumerate(variables):
        values = frame[name].dropna()
        axes[row, 0].hist(values, bins=50, color="#287271", alpha=0.85)
        axes[row, 0].set_title(f"{name}: raw scale")
        axes[row, 1].hist(np.log1p(values), bins=50, color="#e07a5f", alpha=0.85)
        axes[row, 1].set_title(f"{name} : log1p(x)")
        axes[row, 0].set_ylabel("Hourly observations")
    figure.tight_layout()
    figure.savefig(str(output / "distributions_log1p.png"), dpi=150)
    plt.close(figure)


def _make_correlation_plot(frame: pd.DataFrame, variables: list[str], output: Path) -> None:
    if len(variables) < 2:
        return
    corr = frame[variables].corr(method="spearman")
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(range(len(variables)), variables, rotation=60, ha="right")
    axis.set_yticks(range(len(variables)), variables)
    figure.colorbar(image, ax=axis, label="Spearman correlation")
    axis.set_title("Weather variable relationships (Spearman)")
    figure.tight_layout()
    figure.savefig(str(output / "correlations_spearman.png"), dpi=150)
    plt.close(figure)


def _make_relationship_plots(frame: pd.DataFrame, output: Path) -> None:
    pairs = [(x, y, title) for x, y, title in RELATIONSHIPS if x in frame and y in frame]
    if not pairs:
        return
    sample = frame.sample(min(len(frame), 6000), random_state=42)
    columns = 2
    rows = (len(pairs) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows))
    axes = np.asarray(axes).reshape(-1)
    for axis, (x, y, title) in zip(axes[: len(pairs)], pairs, strict=True):
        axis.scatter(sample[x], sample[y], s=7, alpha=0.2, color="#287271", edgecolors="none")
        axis.set_xlabel(x)
        axis.set_ylabel(y)
        axis.set_title(title)
    for axis in axes[len(pairs) :]:
        axis.remove()
    figure.tight_layout()
    figure.savefig(str(output / "relations_meteo.png"), dpi=150)
    plt.close(figure)


def build_report(frame: pd.DataFrame, source: Path, output: Path) -> str:
    numeric = [name for name in CORE_VARIABLES if name in frame]
    if not numeric:
        raise ValueError("No recognized core weather variables were found in the Parquet file.")

    numeric_frame = frame[numeric].apply(pd.to_numeric, errors="coerce")
    describe = numeric_frame.describe(percentiles=[0.5, 0.9, 0.99]).T
    skewness = numeric_frame.skew()
    missing = frame.isna().mean().mul(100).sort_values(ascending=False)
    correlation = numeric_frame.corr(method="spearman")

    log_candidates = [
        name
        for name in numeric
        if numeric_frame[name].dropna().ge(0).all()
        and skewness[name] > 1
        and numeric_frame[name].max() > 0
    ]
    strong_links = []
    for left_index, left in enumerate(numeric):
        for right in numeric[left_index + 1 :]:
            coefficient = correlation.loc[left, right]
            if pd.notna(coefficient) and abs(coefficient) >= 0.7:
                strong_links.append((left, right, coefficient))
    strong_links.sort(key=lambda item: abs(item[2]), reverse=True)

    start = frame["date"].min()
    end = frame["date"].max()
    interval_hours = frame["date"].diff().dt.total_seconds().div(3600)
    modal_interval = interval_hours.dropna().mode()
    interval_text = f"{modal_interval.iloc[0]:g} h" if not modal_interval.empty else "undetermined"

    lines = [
        "# Exploratory analysis of the weather archive",
        "",
        f"- File: `{source}`",
        f"- Dimensions: {len(frame):,} rows × {len(frame.columns)} columns".replace(",", " "),
        f"- Date range: {start} to {end}",
        f"- Modal time step: {interval_text}",
        "",
        "## Data quality",
        "",
        f"- Unparseable dates: {frame['date'].isna().sum()}.",
        f"- Duplicate dates: {frame['date'].duplicated().sum()}.",
        "- Columns with the most missing values: "
        + ", ".join(f"{name} ({value:.2f}%)" for name, value in missing.head(8).items())
        + ".",
        "- The archive is hourly when the modal step is 1 h; check for gaps and duplicates before creating time shifts.",
        "",
        "## Relationships between variables",
        "",
        "Spearman correlations for the core weather variables. These describe simultaneous associations, "
        "not causal effects; consecutive hourly observations are not independent.",
        "",
        "| Variables | Spearman ρ | Interpretation |",
        "| --- | ---: | --- |",
    ]
    if strong_links:
        for left, right, coefficient in strong_links[:12]:
            lines.append(f"| {left} – {right} | {coefficient:.3f} | Strong monotonic relationship; check for redundancy before using both. |")
    else:
        lines.append("| None | n/a | No relationship exceeds an absolute Spearman rho of 0.70. |")

    lines.extend(
        [
            "",
            "Weather context: air temperature and dew point often follow the same air mass; wind and gusts are "
            "redundant; pressure and sea-level pressure may nearly duplicate information. Humidity, cloud cover, "
            "and solar radiation are affected by the diurnal cycle. A weak correlation with rain does not imply "
            "no relationship: precipitation is intermittent and often nonlinear.",
            "",
            "For forecasting, also build lagged relationships and rolling aggregates; never calculate these "
            "features from observations later than the prediction time.",
            "",
            "## Logarithms and transformations",
            "",
            "`log1p(x) = ln(1 + x)` accepts zero. Skewness alone is not enough: check the zero rate, sign, and "
            "physical meaning, then compare distributions and model validation results.",
            "",
            "| Variable | Skewness | Zeros | Min–max | Recommendation |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for name in numeric:
        values = numeric_frame[name].dropna()
        zero_share = values.eq(0).mean() * 100 if len(values) else np.nan
        lines.append(
            f"| {name} | {_format_number(skewness[name])} | {zero_share:.1f} % | "
            f"{_format_number(describe.loc[name, 'min'])} to {_format_number(describe.loc[name, 'max'])} | "
            f"{_log_recommendation(numeric_frame[name], skewness[name])} |"
        )

    lines.extend(
        [
            "",
            "### Practical decisions",
            "",
            "- **Precipitation and snow**: many zeros; consider a two-stage target (occurrence, then conditional "
            "amount). Test `log1p` on positive amounts; the transformation alone does not solve the dry/wet imbalance.",
            "- **Precipitation target in this project**: quantity regression is trained in `log1p` space and "
            "restored with `expm1` for inference and evaluation. Rain-probability classification and predictor "
            "features are unchanged.",
            "- **Solar radiation**: separate night from day first because nighttime zeros are structural; compare "
            "`log1p` on daytime values.",
            "- **Wind and gusts**: start in physical units; transform only if residuals or validation show a benefit.",
            "- **Temperature, humidity, pressure, and wind direction**: no direct logarithm is recommended. For "
            "direction in degrees, use sine/cosine to respect the 360° → 0° wraparound.",
            "- Inverse-transform any target before calculating metrics in weather units.",
            "",
            "## Descriptive statistics",
            "",
            "| Variable | Mean | Median | P90 | P99 | Standard deviation |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in numeric:
        row = describe.loc[name]
        lines.append(
            f"| {name} | {_format_number(row['mean'])} | {_format_number(row['50%'])} | "
            f"{_format_number(row['90%'])} | {_format_number(row['99%'])} | {_format_number(row['std'])} |"
        )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `correlations_spearman.png`: matrix of monotonic associations.",
            "- `relations_meteo.png`: scatter plots from a reproducible sample.",
            "- `distributions_log1p.png`: raw and `log1p` distributions for highly skewed non-negative variables.",
            "",
        ]
    )

    _make_correlation_plot(numeric_frame, numeric, output)
    _make_relationship_plots(frame, output)
    _make_distribution_plots(numeric_frame, log_candidates, output)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET, help="Weather archive in Parquet format")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eda" / "results",
        help="Directory for the report and figures",
    )
    args = parser.parse_args()

    frame = load_dataset(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    report = build_report(frame, args.data, args.output)
    report_path = args.output / "eda_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"EDA report: {report_path}")
    print(f"EDA figures: {args.output}")


if __name__ == "__main__":
    main()