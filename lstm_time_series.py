# -*- coding: utf-8 -*-
"""
RECURRENT NEURAL NETWORKS (LSTM) FOR TIME SERIES: WORKING TOOL
MBA in Data Science and Analytics (USP/Esalq). Working version generated with
Claude from the Deep Learning course, Class 3 (Aula_3_1_2: LSTM with
weatherAUS; Aula_3_1_3: LSTM with PETR4).

GOAL
- Predict the next value of a series (and, in the univariate case, several
  steps ahead) using the last WINDOW observations as input to an LSTM.
  Accepts additional explanatory variables (lagged exogenous series).

WHAT THE SCRIPT DOES ON ITS OWN
- Sorts by date, creates sliding windows, splits train/validation/test in
  TIME ORDER, normalizes with MinMax fitted ONLY on the training data, trains
  with early stopping, compares with the naive model (last value) and
  forecasts the future recursively.

Installation (once, in the Anaconda Prompt):
    pip install pandas numpy scikit-learn matplotlib openpyxl tensorflow joblib
"""

#%% CONFIGURATION (the only place to edit)

FILE = "PETR4.SA.csv"
SHEET = None
SEPARATOR = None
DATE_COLUMN = "Date"            # date column (None = already in order)
TARGET_COLUMN = "Close"         # series to forecast
EXOGENOUS_VARIABLES = []        # e.g. ["Volume"] (used lagged, as input)
FILTER = None                   # e.g. {"Location": "Sydney"}

WINDOW = 20                     # number of past periods used in each prediction
TEST_PROPORTION = 0.2           # end of the series reserved for testing
VALIDATION_PROPORTION = 0.1     # end of the training part used for early stopping
FUTURE_HORIZON = 10             # steps ahead (univariate only; 0 = do not forecast)

LSTM_UNITS = [32]               # neurons per LSTM layer (e.g. [64, 32] = 2 layers)
DROPOUT = 0.2
MAX_EPOCHS = 300
PATIENCE = 25
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
SEED = 42

OUTPUT_FOLDER = "results"
SHOW_PLOTS = True


#%% Packages

from pathlib import Path
import os
import random
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = str(SEED)
import tensorflow as tf

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


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


#%% 1. Load and sort the series

data = read_data(FILE, SHEET, SEPARATOR)
if FILTER:
    for column, value in FILTER.items():
        data = data[data[column] == value]
columns = [TARGET_COLUMN] + list(EXOGENOUS_VARIABLES)
missing_cols = [c for c in columns if c not in data.columns]
if missing_cols:
    raise KeyError(f"Columns not found: {missing_cols}")
if DATE_COLUMN:
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    data = data.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN)
series = data[columns].apply(pd.to_numeric, errors="coerce")
n_missing = int(series.isna().any(axis=1).sum())
series = series.interpolate(limit_direction="both").reset_index(drop=True)
dates = (data[DATE_COLUMN].reset_index(drop=True) if DATE_COLUMN
         else pd.Series(range(len(series))))

log("## 1. Series")
log(f"- File: `{FILE}` | target: `{TARGET_COLUMN}` | exogenous: "
    f"{', '.join(EXOGENOUS_VARIABLES) or 'none'}"
    + (f" | filter: {FILTER}" if FILTER else ""))
log(f"- Observations: {len(series)} | missing values interpolated: {n_missing}")
if len(series) < 5 * WINDOW:
    raise ValueError(f"Series too short for WINDOW = {WINDOW}: reduce the window.")


#%% 2. Time-based split and normalization (no leakage)
# MinMaxScaler learns the minimum and maximum ONLY on the training data.
# Normalizing with the whole series would let the model "see" the future scale.

n = len(series)
test_cut = int(n * (1 - TEST_PROPORTION))
valid_cut = int(test_cut * (1 - VALIDATION_PROPORTION))
scaler = MinMaxScaler().fit(series.iloc[:valid_cut])
normalized = scaler.transform(series)


def make_windows(matrix, start, end):
    """Create pairs (last WINDOW rows, next target value)."""
    X, y = [], []
    for t in range(max(start, WINDOW), end):
        X.append(matrix[t - WINDOW:t])
        y.append(matrix[t, 0])
    return np.array(X), np.array(y)


X_tr, y_tr = make_windows(normalized, 0, valid_cut)
X_va, y_va = make_windows(normalized, valid_cut, test_cut)
X_te, y_te = make_windows(normalized, test_cut, n)
log(f"- Windows: train {len(X_tr)} | validation {len(X_va)} | test {len(X_te)} "
    f"(window = {WINDOW} periods, {len(columns)} variable(s) per period)")


def denormalize(values):
    """Bring normalized target values back to the original scale."""
    helper = np.zeros((len(values), len(columns)))
    helper[:, 0] = values
    return scaler.inverse_transform(helper)[:, 0]


#%% 3. LSTM model

layers = [tf.keras.Input(shape=(WINDOW, len(columns)))]
for i, units in enumerate(LSTM_UNITS):
    layers.append(tf.keras.layers.LSTM(units,
                                       return_sequences=i < len(LSTM_UNITS) - 1))
    layers.append(tf.keras.layers.Dropout(DROPOUT))
layers.append(tf.keras.layers.Dense(1))
model = tf.keras.Sequential(layers)
model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss="mse")
stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                        restore_best_weights=True)
history = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=MAX_EPOCHS,
                    batch_size=BATCH_SIZE, callbacks=[stop], shuffle=False,
                    verbose=0).history
log("\n## 2. Training")
log(f"- Architecture: LSTM {LSTM_UNITS} + Dropout {DROPOUT} + Dense(1)")
log(f"- Epochs: {len(history['loss'])} (best epoch: "
    f"{int(np.argmin(history['val_loss'])) + 1})")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history["loss"], label="Train")
ax.plot(history["val_loss"], label="Validation")
ax.set(title="Learning curve (MSE on the normalized scale)", xlabel="Epoch",
       ylabel="Loss")
ax.legend()
save_figure(fig, "01_learning_curve")


#%% 4. Test evaluation x naive model
# Naive = predict that the next value equals the last observed one. In
# financial series it is very hard to beat; the LSTM is only worth it if it does.

pred_te = denormalize(model.predict(X_te, verbose=0).ravel())
actual_te = denormalize(y_te)
naive = denormalize(X_te[:, -1, 0])


def metrics(actual, pred):
    """RMSE, MAE, MAPE and R²."""
    return {"RMSE": np.sqrt(mean_squared_error(actual, pred)),
            "MAE": mean_absolute_error(actual, pred),
            "MAPE_%": (np.mean(np.abs((actual - pred) / actual)) * 100
                       if (actual != 0).all() else np.nan),
            "R2": r2_score(actual, pred)}


comparison = pd.DataFrame({"LSTM": metrics(actual_te, pred_te),
                           "Naive (last value)": metrics(actual_te, naive)}).T
log("\n## 3. Test (one-step-ahead forecast)")
log(md_table(comparison, 4))
lstm_wins = comparison.loc["LSTM", "RMSE"] < comparison.loc["Naive (last value)", "RMSE"]
log("- " + ("The LSTM beats the naive model." if lstm_wins else
            "The LSTM does NOT beat the naive model: the series has little "
            "exploitable pattern beyond the last value (common in stock prices)."))

dates_te = dates.iloc[max(test_cut, WINDOW):n]
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(dates_te.values, actual_te, color="black", label="Actual")
ax.plot(dates_te.values, pred_te, color="darkred", label="LSTM")
ax.plot(dates_te.values, naive, color="grey", ls=":", label="Naive")
ax.set(title="Test: actual x predicted", xlabel="Period", ylabel=TARGET_COLUMN)
ax.legend()
save_figure(fig, "02_test")


#%% 5. Future forecast (recursive, univariate case only)

future = None
if FUTURE_HORIZON and not EXOGENOUS_VARIABLES:
    window = normalized[-WINDOW:, :1].copy()
    forecasts = []
    for _ in range(FUTURE_HORIZON):
        next_value = model.predict(window[np.newaxis], verbose=0)[0, 0]
        forecasts.append(next_value)
        window = np.vstack([window[1:], [[next_value]]])
    future = pd.DataFrame({"step": range(1, FUTURE_HORIZON + 1),
                           "forecast": denormalize(np.array(forecasts))})
    log(f"\n## 4. Forecast for the next {FUTURE_HORIZON} periods")
    log(md_table(future.set_index("step"), 4))
    log("- Recursive forecast: each step uses the previous forecast as "
        "input; the error accumulates with the horizon.")
elif EXOGENOUS_VARIABLES:
    log("\n- Recursive future forecast not made: with exogenous variables, "
        "their future values would need to be known.")


#%% 6. Save

model.save(OUTPUT / "lstm_model.keras")
joblib.dump({"scaler": scaler, "columns": columns, "window": WINDOW},
            OUTPUT / "scaler_and_config.joblib")
comparison.to_csv(OUTPUT / "test_metrics.csv", encoding="utf-8-sig")
pd.DataFrame({"period": dates_te.values, "actual": actual_te, "lstm": pred_te,
              "naive": naive}).to_csv(OUTPUT / "test_forecasts.csv", index=False,
                                      encoding="utf-8-sig")
if future is not None:
    future.to_csv(OUTPUT / "future_forecast.csv", index=False, encoding="utf-8-sig")

log("\n## 5. How to interpret")
log("- Always compare with the naive model and with the classical models of "
    "model 09 (ETS/ARIMA), which are usually easier to explain.")
log("- Neural network results vary with the seed; for decisions, repeat "
    "with other seeds and check the stability.")

save_report("Report: LSTM for Time Series")
