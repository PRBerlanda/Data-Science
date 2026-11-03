# 12 · LSTM Recurrent Neural Networks for Time Series

A ready-to-use tool that forecasts the next value of a time series with an **LSTM** (Long Short-Term Memory) network, using the last N observations as input. It splits the data in time order, normalizes without leakage, trains with early stopping, compares the network with the naive forecast and, for univariate series, forecasts several steps ahead.

> Part of the [Data-Science](https://github.com/PRBerlanda/Data-Science/tree/main) repository. Each branch holds one model.

---

## What this model does

An LSTM is a recurrent neural network designed for sequences. It reads a **window** of past values one step at a time and keeps an internal memory, controlled by "gates", that decides what to remember and what to forget. The network learns to map each window of the last N periods to the next value. Additional series (exogenous variables, such as volume) can be added as inputs.

Since many series, especially financial prices, are hard to predict beyond their last value, the tool always compares the LSTM with the **naive model** (next value = last value).

## When to use it

- Long series with complex, non-linear patterns.
- Multivariate forecasting, where other series help predict the target.
- As a challenger to the classical models in [09 · Time Series Forecasting](https://github.com/PRBerlanda/Data-Science/tree/09-Time-Series-Forecasting).

**Prefer the classical models** for short series or when you need an explainable model and reliable intervals.

## How it works

1. **Load and sort** the series by date, optionally filtering a sub-series, and interpolate missing values.
2. **Time-based split** into train, validation (end of the training part, used for early stopping) and test (end of the series).
3. **MinMax normalization fitted only on the training part**, so the model never "sees" the scale of the future.
4. **Sliding windows**: each input is the last `WINDOW` periods and the output is the next value of the target.
5. **LSTM network**: one or more LSTM layers with dropout, followed by a dense output, trained with Adam and early stopping.
6. **Test evaluation** (one step ahead) against the naive model, using RMSE, MAE, MAPE and R² on the original scale.
7. **Recursive future forecast** (univariate only): each predicted value is fed back as input for the next step.
8. **Save** the network and the scaler configuration.

## Inputs

**Data:** a table with a date column (or rows already in time order) and the target series. Optional exogenous columns.

**Configuration** (the `CONFIGURATION` cell):

| Parameter | What it controls |
|---|---|
| `FILE`, `SHEET`, `SEPARATOR` | Input file, Excel sheet and CSV separator. |
| `DATE_COLUMN` | Date column (`None` if the rows are already ordered). |
| `TARGET_COLUMN` | The series to forecast. |
| `EXOGENOUS_VARIABLES` | Extra series used as lagged inputs, e.g. `["Volume"]`. |
| `FILTER` | Select a sub-series, e.g. `{"Location": "Sydney"}`. |
| `WINDOW` | Number of past periods used for each prediction. |
| `TEST_PROPORTION`, `VALIDATION_PROPORTION` | Shares of the series reserved for testing and for early stopping. |
| `FUTURE_HORIZON` | Steps to forecast ahead (univariate only; 0 = none). |
| `LSTM_UNITS`, `DROPOUT` | Neurons per LSTM layer (e.g. `[64, 32]` = two layers) and dropout rate. |
| `MAX_EPOCHS`, `PATIENCE`, `BATCH_SIZE`, `LEARNING_RATE`, `SEED` | Training settings. |

## Outputs

Saved in `results/`:

| File | Content |
|---|---|
| `report.md` | Series summary, window counts, training epochs, LSTM vs. naive comparison, future forecast and interpretation. |
| `test_metrics.csv` | RMSE, MAE, MAPE and R² of the LSTM and of the naive model on the test period. |
| `test_forecasts.csv` | Actual values, LSTM forecast and naive forecast for each test period. |
| `future_forecast.csv` | Recursive forecast for the next periods. |
| `lstm_model.keras`, `scaler_and_config.joblib` | Trained network, scaler, columns and window size. |
| `01_learning_curve.png`, `02_test.png` | Learning curve and test period chart. |

## How to read the results

- **LSTM vs. naive**: the LSTM is only worth using if it has a lower error than the naive model. If it does not, the series has little exploitable pattern beyond its last value, which is common in stock prices.
- **RMSE / MAE**: average error in the series' unit (RMSE penalizes large errors more). **MAPE**: average % error. **R²**: share of the variance explained. It tends to look high for trending series even when the model only follows the last value, so always compare with the naive model.
- **Learning curve**: training and validation losses should both fall and level off. A rising validation loss signals overfitting. The best epoch is the one restored by early stopping.
- **Recursive forecast**: each step uses the previous forecast as input, so the error **accumulates** with the horizon.
- Neural network results vary with the random seed. For decisions, repeat with other seeds and check the stability.

## How to run

```bash
pip install -r requirements.txt
pip install tensorflow
```

1. Put your data file next to `lstm_time_series.py`.
2. Edit the `CONFIGURATION` cell (at least `FILE`, `DATE_COLUMN` and `TARGET_COLUMN`).
3. Run the script in Spyder (F5) or with `python lstm_time_series.py`.
4. Open `results/report.md`.

## Example dataset

`PETR4.SA.csv` contains daily prices of the Petrobras (PETR4) stock: date, open, high, low, close, adjusted close and volume. The default configuration forecasts the closing price (`Close`).

## Credits

Built from the Deep Learning course, Class 3 (LSTM with weatherAUS and PETR4), of the MBA in Data Science and Analytics (USP/Esalq).
