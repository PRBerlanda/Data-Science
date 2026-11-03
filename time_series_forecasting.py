# -*- coding: utf-8 -*-
"""
TIME SERIES: DIAGNOSTICS, MODEL COMPARISON AND FORECASTING
MBA in Data Science and Analytics (USP/Esalq). Working version generated with
Claude from the "Time Series" course script (Prof. Fabiano Guasti Lima).

GOAL
- Forecast a univariate series (sales, demand, revenue, consumption, price...)
  choosing the best model by OUT-OF-SAMPLE performance.

WHAT THE SCRIPT DOES ON ITS OWN
1. Builds the date index (date column, "year + month" or "1979.01"),
   aggregates duplicates and fills in missing periods.
2. Diagnostics: decomposition (trend/seasonality), stationarity
   (ADF and KPSS), ACF/PACF.
3. Splits train/test respecting time order and compares: benchmarks (naive,
   mean, drift, seasonal naive), exponential smoothing (SES, Holt, additive
   and multiplicative Holt-Winters), best-AIC ETS and automatic ARIMA/SARIMA,
   by MAPE, RMSE, MAE and Theil's U.
4. Refits the best model on the full series, forecasts the requested horizon
   with confidence intervals and checks the residuals (Ljung-Box).

Installation (once, in the Anaconda Prompt):
    pip install pandas numpy scipy statsmodels matplotlib openpyxl
"""

#%% CONFIGURATION (the only place to edit)

# --- Data ----------------------------------------------------------------
FILE = "airpassengers.xlsx"
SHEET = None
SEPARATOR = None
VALUE_COLUMN = "passengers"     # series to forecast
DATE_COLUMN = "auto"            # name, "auto" (detect) or None (no date)
YEAR_MONTH_COLUMNS = None       # alternative: ("year", "month") when the date
                                # comes in two columns
FILTER = None                   # e.g. {"region": "CO"} for a sub-series
START_DATE = "2000-01-01"       # only used if there is no date column
FREQUENCY = "auto"              # "auto" or "MS" (monthly), "QS", "W", "D", "B", "YS"...
AGGREGATION = "sum"             # how to aggregate repeated values on the same date
                                # ("sum" or "mean")
RESAMPLE_TO = None              # e.g. "MS" to turn daily into monthly

# --- Modeling -------------------------------------------------------------
SEASONALITY = "auto"            # seasonal period (12 monthly, 4 quarterly, 7
                                # daily...): "auto" from the frequency; None = none
TEST_SIZE = "auto"              # number of test periods ("auto" = 1 seasonal
                                # cycle or 20% of the series)
FORECAST_HORIZON = "auto"       # number of periods to forecast ("auto" = TEST_SIZE)
MODELS = ["naive", "mean", "drift", "seasonal_naive", "ses", "holt",
          "hw_additive", "hw_multiplicative", "ets_auto", "arima_auto"]
CRITERION = "MAPE"              # "MAPE", "RMSE" or "MAE"
CONFIDENCE_LEVEL = 0.95
MAX_P_Q = 2                     # maximum p and q order in the ARIMA search

# --- Output ---------------------------------------------------------------
OUTPUT_FOLDER = "results"
SHOW_PLOTS = True
SIGNIFICANCE_LEVEL = 0.05


#%% Packages

from pathlib import Path
from itertools import product
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")


#%% Utility functions (shared by all tools; no need to edit)

SCRIPT_DIR = (Path(__file__).resolve().parent
              if "__file__" in globals() else Path.cwd())
REPORT = []


def resolve_path(relative):
    """Turn a path relative to the script folder into an absolute path."""
    p = Path(relative)
    return p if p.is_absolute() else (SCRIPT_DIR / p).resolve()


def read_data(file, sheet=None, separator=None):
    """
    Read CSV, Excel, Parquet, Pickle or JSON based on the file extension.

    For CSV, the separator is detected and the most common encodings are tried
    (utf-8, utf-8 with BOM and latin-1), avoiding errors with accented text.
    """
    path = resolve_path(file)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = path.suffix.lower()
    df = None
    if ext in (".csv", ".txt"):
        for enc in ("utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(path, sep=separator, engine="python", encoding=enc)
                break
            except UnicodeDecodeError:
                continue
    elif ext in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in (".pkl", ".pickle"):
        df = pd.read_pickle(path)
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    if isinstance(df, pd.Series):
        df = df.to_frame()
    # Remove spaces and invisible marks (BOM) from column names
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    return df


def prepare_output(folder):
    """Create (if needed) and return the results folder."""
    output = resolve_path(folder)
    output.mkdir(parents=True, exist_ok=True)
    return output


def log(text=""):
    """Print to the console and keep the line for report.md."""
    print(text)
    REPORT.append(str(text))


def save_figure(fig, name):
    """Save the chart as PNG in the output folder and show it (if configured)."""
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name}.png", dpi=120, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def save_report(title):
    """Write all logged lines to report.md."""
    text = f"# {title}\n\n" + "\n".join(REPORT) + "\n"
    (OUTPUT / "report.md").write_text(text, encoding="utf-8")
    print(f"\nReport saved to: {OUTPUT / 'report.md'}")


def md_table(df, decimals=3):
    """Convert a DataFrame into a Markdown table (without depending on tabulate)."""
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return ""
            if x != 0 and abs(x) < 10 ** -decimals:
                return f"{x:.2e}"
            return f"{round(float(x), decimals):g}" if abs(x) < 1e6 else f"{x:,.0f}"
        return x
    d = df.copy()
    d = d.reset_index()
    if d.columns[0] == "index":
        d = d.rename(columns={"index": ""})
    lines = ["| " + " | ".join(map(str, d.columns)) + " |",
             "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |"
              for r in d.astype(object).map(fmt).values]
    return "\n".join(lines)


OUTPUT = prepare_output(OUTPUT_FOLDER)


#%% 1. Build the time series

data = read_data(FILE, SHEET, SEPARATOR)
data.columns = [str(c).strip() for c in data.columns]
if FILTER:
    for col, value in FILTER.items():
        data = data[data[col] == value]
if VALUE_COLUMN not in data.columns:
    raise KeyError(f"Series column not found: {VALUE_COLUMN}")

# Keywords used to detect the date column automatically (English and
# Portuguese column names are recognized)
DATE_KEYWORDS = ("date", "month", "year", "period", "time",
                 "data", "mes", "mês", "ano", "periodo")


def convert_date(df):
    """Convert the date column(s) to datetime, accepting several formats."""
    if YEAR_MONTH_COLUMNS:
        year, month = YEAR_MONTH_COLUMNS
        return pd.to_datetime(dict(year=df[year].astype(int),
                                   month=df[month].astype(int), day=1))
    col = DATE_COLUMN
    if col == "auto":
        candidates = [c for c in df.columns if c != VALUE_COLUMN and (
            pd.api.types.is_datetime64_any_dtype(df[c])
            or any(k in c.lower() for k in DATE_KEYWORDS))]
        col = candidates[0] if candidates else None
    if col is None:
        return None
    series = df[col]
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_float_dtype(series) and series.between(1000, 3000).all():
        # "1979.01" format = year.month (careful: 1979.1 = October)
        text = series.map(lambda v: f"{v:.2f}")
        return pd.to_datetime(text.str[:4] + "-" + text.str[5:7] + "-01")
    if pd.api.types.is_integer_dtype(series) and series.between(1000, 3000).all():
        return pd.to_datetime(series.astype(str) + "-01-01")
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


dates = convert_date(data)
values = pd.to_numeric(data[VALUE_COLUMN], errors="coerce")
if dates is None:
    base_freq = "MS" if FREQUENCY == "auto" else FREQUENCY
    series = pd.Series(values.values,
                       index=pd.date_range(START_DATE, periods=len(values),
                                           freq=base_freq))
else:
    series = (pd.Series(values.values, index=pd.DatetimeIndex(dates.values))
              .dropna().groupby(level=0).agg(AGGREGATION).sort_index())
series = series[series.index.notna()]

if RESAMPLE_TO:
    series = series.resample(RESAMPLE_TO).agg(AGGREGATION)

freq = pd.infer_freq(series.index[:50]) if FREQUENCY == "auto" else FREQUENCY
if freq is None:
    # E.g. trading days (business days with holidays). Works as a sequence.
    freq_label = "irregular (treated as a sequence of periods)"
    model_series = pd.Series(series.values, index=pd.RangeIndex(len(series)))
else:
    complete = series.asfreq(freq)
    n_gaps = int(complete.isna().sum())
    series = complete.interpolate() if n_gaps else complete
    freq_label = freq + (f" ({n_gaps} missing periods interpolated)"
                         if n_gaps else "")
    model_series = series

log("## 1. Series")
log(f"- File: `{FILE}` | variable: `{VALUE_COLUMN}`"
    + (f" | filter: {FILTER}" if FILTER else ""))
log(f"- Period: {series.index.min():%Y-%m-%d} to {series.index.max():%Y-%m-%d}"
    f" | {len(series)} observations | frequency: {freq_label}")
log(md_table(series.describe().to_frame(VALUE_COLUMN)))

if SEASONALITY == "auto":
    season_map = {"M": 12, "Q": 4, "W": 52, "D": 7, "B": 5, "H": 24}
    key = (freq or "").upper().replace("S", "")[:1] if freq else ""
    m = season_map.get(key)
    if freq and freq.upper().startswith(("A", "Y")):
        m = None
else:
    m = SEASONALITY
if m and len(series) < 2 * m + 2:
    log(f"- Series too short for seasonality {m}: seasonal models turned off.")
    m = None
log(f"- Seasonal period considered: {m or 'none'}")
positive = bool((model_series > 0).all())

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(series.index, series.values)
ax.set(title=f"Series: {VALUE_COLUMN}", xlabel="Date", ylabel=VALUE_COLUMN)
save_figure(fig, "01_series")


#%% 2. Diagnostics: decomposition, stationarity and autocorrelation
# ADF: H0 = series is NOT stationary (has a unit root). p < 0.05 → stationary.
# KPSS: H0 = series is stationary. p < 0.05 → NOT stationary.
# Using both avoids fragile conclusions.

log("\n## 2. Diagnostics")
seasonal_strength = 0.0
if m:
    stl = STL(model_series, period=m, robust=True).fit()
    var_r = np.var(stl.resid)
    seasonal_strength = max(0.0, 1 - var_r / np.var(stl.seasonal + stl.resid))
    trend_strength = max(0.0, 1 - var_r / np.var(stl.trend + stl.resid))
    log(f"- Trend strength (STL): {trend_strength:.2f} | seasonal "
        f"strength: {seasonal_strength:.2f} (0 to 1; > 0.64 = strong)")
    decomp_type = "multiplicative" if positive else "additive"
    decomp = seasonal_decompose(model_series, model=decomp_type, period=m)
    fig = decomp.plot()
    fig.set_size_inches(11, 7)
    fig.suptitle(f"{'Multiplicative' if positive else 'Additive'} decomposition",
                 y=1.01)
    save_figure(fig, "02_decomposition")


def required_differences(x, maximum=2):
    """Number of differences until ADF rejects a unit root and KPSS does not reject."""
    for d in range(maximum + 1):
        y_d = np.diff(x, n=d) if d else np.asarray(x)
        p_adf = adfuller(y_d, autolag="AIC")[1]
        try:
            p_kpss = kpss(y_d, regression="c", nlags="auto")[1]
        except Exception:
            p_kpss = 1.0
        if p_adf < SIGNIFICANCE_LEVEL and p_kpss >= SIGNIFICANCE_LEVEL:
            return d, p_adf, p_kpss
    return maximum, p_adf, p_kpss

p_adf0 = adfuller(model_series, autolag="AIC")[1]
try:
    p_kpss0 = kpss(model_series, regression="c", nlags="auto")[1]
except Exception:
    p_kpss0 = np.nan
D = 1 if (m and seasonal_strength > 0.64) else 0
base_d = np.diff(model_series.values, n=m) if D else model_series.values
d, p_adf_d, p_kpss_d = required_differences(base_d)
log(f"- Original series: ADF p = {p_adf0:.4f} | KPSS p = {p_kpss0:.4f} → "
    f"{'stationary' if p_adf0 < SIGNIFICANCE_LEVEL and p_kpss0 >= SIGNIFICANCE_LEVEL else 'NOT stationary'}")
log(f"- Suggested differencing: d = {d} (regular), D = {D} (seasonal)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
diff_series = pd.Series(base_d).diff(d).dropna() if d else pd.Series(base_d)
lags = min(40, len(diff_series) // 2 - 1)
plot_acf(diff_series, lags=lags, ax=axes[0])
plot_pacf(diff_series, lags=lags, ax=axes[1], method="ywm")
axes[0].set_title(f"ACF (series with d={d}, D={D})")
axes[1].set_title(f"PACF (series with d={d}, D={D})")
save_figure(fig, "03_acf_pacf")


#%% 3. Train and test (respecting time order)

n_test = (int(TEST_SIZE) if TEST_SIZE != "auto"
          else (m if m and m <= 0.3 * len(model_series) else max(1, int(0.2 * len(model_series)))))
h_forecast = int(FORECAST_HORIZON) if FORECAST_HORIZON != "auto" else n_test
train, test = model_series.iloc[:-n_test], model_series.iloc[-n_test:]
log("\n## 3. Train and test")
log(f"- Train: {len(train)} periods | test: {n_test} periods | "
    f"final forecast: {h_forecast} periods ahead")

alpha = 1 - CONFIDENCE_LEVEL
z = stats.norm.ppf(1 - alpha / 2)


def forecast_model(name, y, h):
    """
    Fit model `name` on series y and forecast h steps.
    Returns (forecast, lower bound, upper bound, fitted object).
    """
    y = pd.Series(np.asarray(y, dtype=float))
    steps = np.arange(1, h + 1)
    if name == "naive":
        pred = np.repeat(y.iloc[-1], h)
        sd = np.std(np.diff(y), ddof=1) * np.sqrt(steps)
        return pred, pred - z * sd, pred + z * sd, None
    if name == "mean":
        pred = np.repeat(y.mean(), h)
        sd = y.std(ddof=1) * np.sqrt(1 + 1 / len(y))
        return pred, pred - z * sd, pred + z * sd, None
    if name == "drift":
        slope = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
        pred = y.iloc[-1] + slope * steps
        sd = np.std(np.diff(y), ddof=1) * np.sqrt(steps * (1 + steps / len(y)))
        return pred, pred - z * sd, pred + z * sd, None
    if name == "seasonal_naive":
        last = y.iloc[-m:].values
        pred = np.array([last[(i % m)] for i in range(h)])
        k = (steps - 1) // m + 1
        sd = np.std(y.values[m:] - y.values[:-m], ddof=1) * np.sqrt(k)
        return pred, pred - z * sd, pred + z * sd, None
    if name in ("ses", "holt", "hw_additive", "hw_multiplicative"):
        cfg = {"ses": dict(error="add", trend=None, seasonal=None),
               "holt": dict(error="add", trend="add", seasonal=None),
               "hw_additive": dict(error="add", trend="add", seasonal="add"),
               "hw_multiplicative": dict(error="mul", trend="add",
                                         seasonal="mul")}[name]
        if cfg["seasonal"]:
            cfg["seasonal_periods"] = m
        fit = ETSModel(y, **cfg).fit(disp=False)
        return _ets_intervals(fit, h)
    if name == "ets_auto":
        best = None
        errors = ["add", "mul"] if is_positive(y) else ["add"]
        seasonals = [None] + (["add", "mul"] if m and is_positive(y) else
                              (["add"] if m else []))
        for err, trend, damped, seas in product(errors, [None, "add"], [False, True],
                                                seasonals):
            if damped and trend is None:
                continue
            if seas == "mul" and err == "add":
                continue
            try:
                fit = ETSModel(y, error=err, trend=trend, damped_trend=damped,
                               seasonal=seas, seasonal_periods=m if seas else None
                               ).fit(disp=False)
                if np.isfinite(fit.aic) and (best is None or fit.aic < best.aic):
                    best = fit
            except Exception:
                continue
        return _ets_intervals(best, h)
    if name == "arima_auto":
        best = None
        seasonal = [(0, 0), (1, 0), (0, 1), (1, 1)] if m else [(0, 0)]
        for p, q, (P, Q) in product(range(MAX_P_Q + 1), range(MAX_P_Q + 1), seasonal):
            seasonal_order = (P, D, Q, m) if m else (0, 0, 0, 0)
            trend = "c" if (d + D) == 0 else ("t" if (d + D) == 1 else "n")
            try:
                fit = SARIMAX(y, order=(p, d, q), seasonal_order=seasonal_order,
                              trend=trend).fit(disp=False)
                if np.isfinite(fit.aic) and (best is None or fit.aic < best.aic):
                    best = fit
            except Exception:
                continue
        pred = best.get_forecast(h)
        ci = pred.conf_int(alpha=alpha)
        return (np.asarray(pred.predicted_mean), ci.iloc[:, 0].values,
                ci.iloc[:, 1].values, best)
    raise ValueError(f"Unknown model: {name}")


def is_positive(y):
    return bool((np.asarray(y) > 0).all())


def _ets_intervals(fit, h):
    """Forecast and intervals of a fitted ETSModel."""
    prediction = fit.get_prediction(start=fit.nobs, end=fit.nobs + h - 1)
    table = prediction.summary_frame(alpha=alpha)
    return (table["mean"].values, table["pi_lower"].values,
            table["pi_upper"].values, fit)


def describe(name, fit):
    """Short text with the chosen specification (ETS/ARIMA)."""
    if fit is None:
        return ""
    if name == "arima_auto":
        o = fit.model.order
        s = fit.model.seasonal_order
        return f"SARIMA{o}x{s}" if m else f"ARIMA{o}"
    if hasattr(fit.model, "error"):
        mod = fit.model
        return (f"ETS(error={mod.error}, trend={mod.trend}"
                f"{' damped' if mod.damped_trend else ''}, "
                f"seasonal={mod.seasonal})")
    return ""


#%% 4. Model comparison on the test period

valid_models = [mo for mo in MODELS
                if not (mo in ("seasonal_naive", "hw_additive", "hw_multiplicative") and not m)
                and not (mo == "hw_multiplicative" and not positive)]
actual = test.values.astype(float)
naive_error = forecast_model("naive", train, n_test)[0]
results, test_forecasts = [], {}
for name in valid_models:
    try:
        pred, low, high, fit = forecast_model(name, train, n_test)
    except Exception as error:
        log(f"- {name}: not fitted ({type(error).__name__})")
        continue
    test_forecasts[name] = pred
    error = actual - pred
    mape = np.mean(np.abs(error / actual)) * 100 if (actual != 0).all() else np.nan
    theil = (np.sqrt(np.sum(error[1:] ** 2)) /
             np.sqrt(np.sum((actual[1:] - actual[:-1]) ** 2))) if n_test > 1 else np.nan
    results.append({"model": name, "specification": describe(name, fit),
                    "MAPE_%": mape, "RMSE": np.sqrt(np.mean(error ** 2)),
                    "MAE": np.mean(np.abs(error)), "Theil_U": theil,
                    "CI_coverage_%": np.mean((actual >= low) & (actual <= high)) * 100})

ranking = pd.DataFrame(results).set_index("model")
column = {"MAPE": "MAPE_%", "RMSE": "RMSE", "MAE": "MAE"}[CRITERION]
if column == "MAPE_%" and ranking[column].isna().all():
    column = "RMSE"
ranking = ranking.sort_values(column)
best = ranking.index[0]

log("\n## 4. Test performance (out of sample)")
log(md_table(ranking, 3))
log(f"- Best model by {column}: **{best}** "
    f"{ranking.loc[best, 'specification']}")
log("- Theil's U < 1: the model beats the naive one; > 1: worse than "
    "simply repeating the last value.")

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(train.index[-3 * n_test:], train.iloc[-3 * n_test:], color="black",
        label="Train")
ax.plot(test.index, actual, color="black", ls="--", label="Test (actual)")
for name in ranking.index[:4]:
    ax.plot(test.index, test_forecasts[name], label=name)
ax.set(title="Forecasts on the test period (4 best models)",
       xlabel="Period", ylabel=VALUE_COLUMN)
ax.legend(fontsize=8)
save_figure(fig, "04_model_test")


#%% 5. Final model: refit on the full series and forecast

pred, low, high, final_fit = forecast_model(best, model_series, h_forecast)
if isinstance(model_series.index, pd.DatetimeIndex):
    future_dates = pd.date_range(model_series.index[-1], periods=h_forecast + 1,
                                 freq=freq)[1:]
else:
    future_dates = pd.RangeIndex(len(model_series), len(model_series) + h_forecast)
final_forecast = pd.DataFrame({"forecast": pred, "lower_bound": low,
                               "upper_bound": high}, index=future_dates)
final_forecast.index.name = "period"

log(f"\n## 5. Final forecast ({best}, {h_forecast} periods, "
    f"{CONFIDENCE_LEVEL:.0%} CI)")
forecast_tab = final_forecast.copy()
if isinstance(forecast_tab.index, pd.DatetimeIndex):
    forecast_tab.index = forecast_tab.index.strftime("%Y-%m-%d")
log(md_table(forecast_tab, 2))

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(model_series.index, model_series.values, color="black", label="History")
ax.plot(future_dates, pred, color="tab:blue", label=f"Forecast ({best})")
ax.fill_between(future_dates, low, high, color="tab:blue", alpha=0.2,
                label=f"{CONFIDENCE_LEVEL:.0%} CI")
ax.set(title="Final forecast", xlabel="Period", ylabel=VALUE_COLUMN)
ax.legend()
save_figure(fig, "05_final_forecast")


#%% 6. Final model residual diagnostics
# Ljung-Box: H0 = residuals without autocorrelation (white noise).
# p < 0.05 → there is still a pattern the model did not capture.

if final_fit is not None:
    residuals = pd.Series(np.asarray(final_fit.resid)).iloc[max(d + D * (m or 0), 1):]
    lb_lags = min(2 * (m or 5), len(residuals) // 5)
    lb = acorr_ljungbox(residuals, lags=[lb_lags], return_df=True)
    p_lb = float(lb["lb_pvalue"].iloc[0])
    p_sw = stats.shapiro(residuals[:5000])[1]
    log("\n## 6. Final model residuals")
    log(f"- Ljung-Box ({lb_lags} lags): p = {p_lb:.4f} → "
        f"{'white noise' if p_lb >= SIGNIFICANCE_LEVEL else 'remaining autocorrelation'}")
    log(f"- Shapiro-Wilk: p = {p_sw:.4f} → residuals "
        f"{'normal' if p_sw >= SIGNIFICANCE_LEVEL else 'not normal (approximate intervals)'}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(residuals.values)
    axes[0].axhline(0, color="red", ls="--")
    axes[0].set(title="Final model residuals", xlabel="Period",
                ylabel="Residual")
    plot_acf(residuals, lags=min(30, len(residuals) // 2 - 1), ax=axes[1])
    axes[1].set_title("Residual ACF")
    save_figure(fig, "06_residuals")
else:
    log("\n## 6. Residuals: benchmark model (no fitted residuals).")


#%% 7. Save results

ranking.to_csv(OUTPUT / "model_comparison_test.csv", encoding="utf-8-sig")
final_forecast.to_csv(OUTPUT / "final_forecast.csv", encoding="utf-8-sig")
pd.DataFrame({"actual": actual, **test_forecasts},
             index=test.index).to_csv(OUTPUT / "test_forecasts.csv",
                                      encoding="utf-8-sig")
series.to_frame(VALUE_COLUMN).to_csv(OUTPUT / "clean_series.csv",
                                     encoding="utf-8-sig")

log("\n## 7. How to interpret")
log("- The choice is made on the TEST set: the best fit on the training set "
    "does not guarantee the best forecast.")
log("- If a simple benchmark (naive, drift) wins, the series has little "
    "exploitable pattern; do not force a complex model.")
log("- The confidence interval widens with the horizon: distant forecasts "
    "are less precise.")

save_report("Report: Time Series")
