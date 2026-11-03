# 09 · Time Series Forecasting (Benchmarks, ETS/Holt-Winters, SARIMA)

A ready-to-use tool that forecasts a **univariate time series**. It prepares the date index, diagnoses trend, seasonality and stationarity, compares ten forecasting methods on a hold-out test period, refits the winner on the full series and forecasts ahead with confidence intervals.

> Part of the [Data-Science](https://github.com/PRBerlanda/Data-Science/tree/main) repository. Each branch holds one model.

---

## What this model does

A time series is a sequence of values ordered in time (monthly sales, daily demand, quarterly revenue…). Forecasting methods learn the series' level, trend and seasonality from the past and project them forward. Because the best in-sample fit is not always the best forecaster, this tool **chooses the model by its performance on data it has not seen**: the last periods of the series, kept aside as a test set.

Methods compared:

| Family | Models |
|---|---|
| Benchmarks | naive (last value), mean, drift, seasonal naive |
| Exponential smoothing | SES, Holt, additive Holt-Winters, multiplicative Holt-Winters |
| Automatic searches | best-AIC ETS, automatic ARIMA/SARIMA |

## When to use it

- Forecasting sales, demand, revenue, consumption, prices, call volume, passengers…
- One series at a time. Use the `FILTER` option to forecast a sub-series (e.g. one region).

**Use another model when** you want to test a recurrent neural network ([12 · LSTM](https://github.com/PRBerlanda/Data-Science/tree/12-LSTM-Time-Series)) or explain Y with other variables at the same point in time ([04](https://github.com/PRBerlanda/Data-Science/tree/04-Linear-Regression)).

## How it works

1. **Build the series**: detects the date column (or uses year + month columns, or the "1979.01" format), aggregates duplicates, optionally resamples (e.g. daily → monthly) and interpolates missing periods.
2. **Diagnostics**:
   - STL decomposition with **trend and seasonal strength**;
   - classical decomposition chart (multiplicative when the series is positive);
   - **ADF and KPSS** stationarity tests and the suggested differencing (d, D);
   - **ACF and PACF** of the differenced series.
3. **Train/test split** that respects time order. The test size defaults to one seasonal cycle (or 20% of the series).
4. **Fit every model on the training part** and forecast the test period. Each is scored by MAPE, RMSE, MAE, Theil's U and confidence interval coverage.
5. **Pick the best model** by the chosen criterion, **refit it on the full series** and forecast the requested horizon with confidence intervals.
6. **Residual diagnostics** of the final model (Ljung-Box and Shapiro-Wilk).

## Inputs

**Data:** a table with a value column and, ideally, a date column. Daily, weekly, monthly, quarterly and yearly data are supported.

**Configuration** (the `CONFIGURATION` cell):

| Parameter | What it controls |
|---|---|
| `FILE`, `SHEET`, `SEPARATOR` | Input file, Excel sheet and CSV separator. |
| `VALUE_COLUMN` | The series to forecast. |
| `DATE_COLUMN` | Date column name, `"auto"` (detect) or `None` (no date; `START_DATE` + `FREQUENCY` are used). |
| `YEAR_MONTH_COLUMNS` | Alternative when the date comes in two columns, e.g. `("year", "month")`. |
| `FILTER` | Select a sub-series, e.g. `{"region": "North"}`. |
| `FREQUENCY`, `AGGREGATION`, `RESAMPLE_TO` | Frequency (`"auto"`, `"MS"`, `"D"`…), how to aggregate duplicates, optional resampling. |
| `SEASONALITY` | Seasonal period (12, 4, 7…), `"auto"` from the frequency, or `None`. |
| `TEST_SIZE`, `FORECAST_HORIZON` | Number of test periods and of periods to forecast (`"auto"` = one cycle). |
| `MODELS` | List of models to compare. |
| `CRITERION` | `"MAPE"`, `"RMSE"` or `"MAE"`. |
| `CONFIDENCE_LEVEL`, `MAX_P_Q` | Interval level and maximum ARIMA order searched. |

## Outputs

Saved in `results/`:

| File | Content |
|---|---|
| `report.md` | Series summary, diagnostics, model ranking, final forecast, residual tests and interpretation. |
| `model_comparison_test.csv` | Specification and test metrics of every model, sorted by the criterion. |
| `test_forecasts.csv` | Actual test values and each model's forecast. |
| `final_forecast.csv` | Forecast with lower and upper bounds for the future periods. |
| `clean_series.csv` | The series after date handling, aggregation and interpolation. |
| `01_series.png` … `06_residuals.png` | Series, decomposition, ACF/PACF, test comparison, final forecast and residuals. |

## How to read the results

- **Trend / seasonal strength** (0 to 1): above about 0.64 is strong. Strong seasonality favors seasonal models and seasonal differencing.
- **ADF** (H0: not stationary) and **KPSS** (H0: stationary) together: stationary when ADF rejects and KPSS does not. Non-stationary series need differencing (handled automatically in ARIMA).
- **MAPE**: average absolute error in % of the actual value, easy to communicate. **RMSE** penalizes large errors. **MAE** is the average absolute error in the series' unit.
- **Theil's U**: below 1, the model beats the naive forecast (repeating the last value). Above 1, it is worse than naive.
- **CI coverage**: share of test values that fell inside the interval. It should be close to the confidence level.
- **If a simple benchmark wins**, the series has little exploitable pattern. Do not force a complex model.
- **Ljung-Box p ≥ 0.05**: residuals behave like white noise, so the model captured the pattern. A low p-value means some structure is left.
- The **confidence interval widens with the horizon**: distant forecasts are less precise.
- Intervals of multiplicative ETS models are simulated, so their coverage can vary slightly between runs.

## How to run

```bash
pip install -r requirements.txt
```

1. Put your data file next to `time_series_forecasting.py`.
2. Edit the `CONFIGURATION` cell (at least `FILE` and `VALUE_COLUMN`).
3. Run the script in Spyder (F5) or with `python time_series_forecasting.py`.
4. Open `results/report.md`.

## Example dataset

`airpassengers.xlsx` is the classic monthly airline passengers series: a month column (`mes`) and the number of passengers (`passengers`). It has a clear upward trend and yearly seasonality.

## Credits

Built from the "Time Series" course script by Prof. Fabiano Guasti Lima (MBA in Data Science and Analytics, USP/Esalq).
