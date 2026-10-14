# Exploratory analysis of the weather archive

- File: `/home/lucas/Documents/Lucas/Projets_python/Projets/Projets_IA/weather/weather_app_backup/binary/full_grid_archive.parquet`
- Dimensions: 146 928 rows × 295 columns
- Date range: 2010-01-01 00:00:00 to 2026-10-05 23:00:00
- Modal time step: 1 h

## Data quality

- Unparseable dates: 0.
- Duplicate dates: 0.
- Columns with the most missing values: e_hemi_snow (0.01%), e_hemi_sw_rad (0.01%), temp (0.01%), hum (0.01%), dew (0.01%), press (0.01%), w_hemi_soil_temp (0.01%), w_hemi_surf_press (0.01%).
- The archive is hourly when the modal step is 1 h; check for gaps and duplicates before creating time shifts.

## Relationships between variables

Spearman correlations for the core weather variables. These describe simultaneous associations, not causal effects; consecutive hourly observations are not independent.

| Variables | Spearman ρ | Interpretation |
| --- | ---: | --- |
| press – surf_press | 1.000 | Strong monotonic relationship; check for redundancy before using both. |
| temp – soil_temp | 0.975 | Strong monotonic relationship; check for redundancy before using both. |
| wind – gusts | 0.935 | Strong monotonic relationship; check for redundancy before using both. |
| temp – dew | 0.875 | Strong monotonic relationship; check for redundancy before using both. |
| dew – soil_temp | 0.875 | Strong monotonic relationship; check for redundancy before using both. |

Weather context: air temperature and dew point often follow the same air mass; wind and gusts are redundant; pressure and sea-level pressure may nearly duplicate information. Humidity, cloud cover, and solar radiation are affected by the diurnal cycle. A weak correlation with rain does not imply no relationship: precipitation is intermittent and often nonlinear.

For forecasting, also build lagged relationships and rolling aggregates; never calculate these features from observations later than the prediction time.

## Logarithms and transformations

`log1p(x) = ln(1 + x)` accepts zero. Skewness alone is not enough: check the zero rate, sign, and physical meaning, then compare distributions and model validation results.

| Variable | Skewness | Zeros | Min–max | Recommendation |
| --- | ---: | ---: | ---: | --- |
| temp | 0.137 | 0.2 % | -13.500 to 38.900 | No priority log transform: skewness is moderate. |
| hum | -0.937 | 0.0 % | 16.000 to 100.000 | No priority log transform: skewness is moderate. |
| dew | -0.313 | 0.3 % | -17.000 to 23.100 | No priority log transform: skewness is moderate. |
| press | -0.467 | 0.0 % | 969.700 to 1049.200 | No priority log transform: skewness is moderate. |
| precip | 7.160 | 82.6 % | 0.000 to 10.300 | Highly skewed with many zeros: separate occurrence from intensity; test log1p on strictly positive values. |
| wind | 0.875 | 0.0 % | 0.000 to 54.100 | No priority log transform: skewness is moderate. |
| wind_dir | -0.363 | 0.0 % | 1.000 to 360.000 | No priority log transform: skewness is moderate. |
| gusts | 0.921 | 0.0 % | 1.100 to 109.400 | No priority log transform: skewness is moderate. |
| cloud | -0.684 | 7.3 % | 0.000 to 100.000 | No priority log transform: skewness is moderate. |
| surf_press | -0.494 | 0.0 % | 962.500 to 1041.200 | No priority log transform: skewness is moderate. |
| soil_temp | 0.155 | 0.2 % | -8.500 to 33.100 | No priority log transform: skewness is moderate. |
| soil_moist | -0.858 | 0.0 % | 0.146 to 0.439 | No priority log transform: skewness is moderate. |
| sw_rad | 1.634 | 46.7 % | 0.000 to 897.000 | Highly skewed with many zeros: separate occurrence from intensity; test log1p on strictly positive values. |
| snow | 28.219 | 99.2 % | 0.000 to 2.030 | Highly skewed with many zeros: separate occurrence from intensity; test log1p on strictly positive values. |

### Practical decisions

- **Precipitation and snow**: many zeros; consider a two-stage target (occurrence, then conditional amount). Test `log1p` on positive amounts; the transformation alone does not solve the dry/wet imbalance.
- **Precipitation target in this project**: quantity regression is trained in `log1p` space and restored with `expm1` for inference and evaluation. Rain-probability classification and predictor features are unchanged.
- **Solar radiation**: separate night from day first because nighttime zeros are structural; compare `log1p` on daytime values.
- **Wind and gusts**: start in physical units; transform only if residuals or validation show a benefit.
- **Temperature, humidity, pressure, and wind direction**: no direct logarithm is recommended. For direction in degrees, use sine/cosine to respect the 360° → 0° wraparound.
- Inverse-transform any target before calculating metrics in weather units.

## Descriptive statistics

| Variable | Mean | Median | P90 | P99 | Standard deviation |
| --- | ---: | ---: | ---: | ---: | ---: |
| temp | 11.309 | 11.100 | 20.300 | 27.800 | 6.908 |
| hum | 78.834 | 83.000 | 95.000 | 99.000 | 15.361 |
| dew | 7.374 | 7.600 | 14.500 | 18.200 | 5.604 |
| press | 1015.964 | 1016.600 | 1027.400 | 1036.800 | 9.670 |
| precip | 0.088 | 0.000 | 0.200 | 1.589 | 0.319 |
| wind | 14.037 | 12.900 | 23.700 | 34.300 | 7.024 |
| wind_dir | 188.865 | 208.000 | 303.000 | 355.000 | 91.560 |
| gusts | 27.329 | 25.200 | 45.700 | 67.300 | 13.647 |
| cloud | 66.776 | 84.000 | 100.000 | 100.000 | 36.581 |
| surf_press | 1008.428 | 1009.100 | 1019.700 | 1028.900 | 9.589 |
| soil_temp | 11.596 | 11.400 | 20.100 | 26.300 | 6.461 |
| soil_moist | 0.330 | 0.351 | 0.401 | 0.428 | 0.069 |
| sw_rad | 130.180 | 6.000 | 461.000 | 775.000 | 200.730 |
| snow | 0.002 | 0.000 | 0.000 | 0.000 | 0.031 |

## Figures

- `correlations_spearman.png`: matrix of monotonic associations.
- `relations_meteo.png`: scatter plots from a reproducible sample.
- `distributions_log1p.png`: raw and `log1p` distributions for highly skewed non-negative variables.
