# 10 · Decision Trees and Ensembles (Random Forest, XGBoost, LightGBM)

A ready-to-use tool for **predicting a target**, either a class (classification) or a value (regression), with tree-based models. It builds a leakage-free pipeline, compares the models with baselines through cross-validation and hyperparameter search, evaluates the winner on a test set, explains it with permutation importance and SHAP, and saves a reusable model.

> Part of the [Data-Science](https://github.com/PRBerlanda/Data-Science/tree/main) repository. Each branch holds one model.

---

## What this model does

A **decision tree** splits the data with a sequence of yes/no questions (e.g. "age < 30?") until it reaches groups that are homogeneous in the target. Single trees are easy to read but unstable. **Ensembles** combine many trees:

- **Random Forest**: many trees, each trained on a random sample of rows and variables, averaged together.
- **Gradient boosting (XGBoost, LightGBM)**: trees built one after another, each correcting the errors of the previous ones.

These models capture non-linear relationships and interactions automatically and usually give the best accuracy on tabular data.

## When to use it

- **Prediction is the priority**: churn, default, fraud, demand, price, risk scores…
- Many variables, mixed types, non-linear effects or interactions.
- Both binary/multiclass classification and regression.

**Use another model when** you need coefficients and formal significance tests to explain effects ([04](https://github.com/PRBerlanda/Data-Science/tree/04-Linear-Regression)–[08](https://github.com/PRBerlanda/Data-Science/tree/08-Multilevel-Modeling)) or the data is a time series ([09](https://github.com/PRBerlanda/Data-Science/tree/09-Time-Series-Forecasting)).

## How it works

1. **Load and prepare the data**: numbers stored as text ("$1,200", "1.234,56") are converted, and columns that look like identifiers are removed.
2. **Detect the problem type**: classification (text, boolean or few integer values) or regression.
3. **Leakage check**: flags numeric variables that predict the target almost perfectly on their own, which usually means information from the future.
4. **Train/test split**, stratified for classification, or a test file you provide.
5. **Pipeline without leakage**: median/most-frequent imputation and one-hot encoding are fitted only on the training folds.
6. **Model comparison by cross-validation**: two baselines (majority class or mean, and logistic or linear regression) plus a decision tree, Random Forest, XGBoost and LightGBM, each with a random or grid **hyperparameter search**.
7. **Overfitting check**: training score vs. validation score of the best model.
8. **Final evaluation on the test set**, used only once: accuracy, macro F1, AUC/Gini and confusion matrix (classification), or RMSE, MAE, R² and MAPE (regression).
9. **Explainability**: permutation importance on the test set and SHAP values (summary plot and explanation of one individual case).
10. **Final model** refit on all data, saved as a reusable pipeline, plus optional predictions for new data.

## Inputs

**Data:** one row per case with the target and the explanatory variables, as CSV, Excel, Parquet, Pickle or JSON. Separate X and y files and a ready-made test set are also supported.

**Configuration** (the `CONFIGURATION` cell):

| Parameter | What it controls |
|---|---|
| `FILE`, `SHEET`, `SEPARATOR` | Input file, Excel sheet and CSV separator. |
| `TARGET`, `X_VARIABLES`, `ID_COLUMN`, `IGNORE_COLUMNS` | Target, explanatory variables and columns to exclude. |
| `CONVERT_NUMERIC_TEXT` | Convert numbers stored as text. |
| `Y_FILE`, `TEST_FILE`, `TEST_Y_FILE` | Optional pre-split files. |
| `PROBLEM_TYPE`, `POSITIVE_CLASS` | `"auto"`, `"classification"` or `"regression"`, and the class of interest in binary problems. |
| `MODELS` | Models to compare: `"tree"`, `"random_forest"`, `"xgboost"`, `"lightgbm"`. |
| `SEARCH`, `N_SEARCH_ITER` | `"random"`, `"grid"` or `None`, and the number of random combinations. |
| `FOLDS`, `TEST_PROPORTION`, `METRIC`, `SEED` | Cross-validation folds, test share, selection metric and seed. |
| `COMPUTE_SHAP`, `SHAP_SAMPLE` | SHAP explanations and sample size. |
| `REFIT_ON_ALL_DATA` | Refit the final model on train + test before saving. |
| `PREDICT_FILE` | Optional file with new cases to predict. |

## Outputs

Saved in `results/`:

| File | Content |
|---|---|
| `report.md` | Data summary, leakage check, cross-validation table, overfitting check, test metrics, importances and reuse instructions. |
| `model_comparison_cv.csv` | Cross-validated score, standard deviation, training score, time and best hyperparameters of each model. |
| `permutation_importance.csv` | Loss in the metric when each variable is shuffled. |
| `shap_importance.csv` | Mean absolute SHAP value per (encoded) variable. |
| `final_model.joblib` | The trained pipeline (preprocessing + model), ready to reuse. |
| `new_data_predictions.csv` | Predictions for `PREDICT_FILE`. |
| `01_roc_curve.png` / `01_observed_predicted.png`, `02_confusion_matrix.png`, `03_permutation_importance.png`, `04_shap_summary.png`, `05_shap_individual_case.png` | Charts. |

## How to read the results

- **Baselines first**: a model only adds value if it clearly beats the baselines.
- **Cross-validation score ± standard deviation**: the expected performance and its stability. Small differences between models may not be meaningful.
- **Training much better than validation** (gap above about 10%): overfitting. Prefer simpler hyperparameters or more regularization.
- **Classification metrics**: AUC (0.5 = random, > 0.8 good), macro F1 (balances all classes), confusion matrix (which classes get confused).
- **Regression metrics**: RMSE and MAE in the target's unit, R² as the share of variance explained, MAPE as the average % error.
- **Permutation importance**: how much the test metric worsens when a variable is shuffled. It works for any model.
- **SHAP**: each variable's contribution to each individual prediction. In the summary plot, points to the right push the prediction up. Color shows whether the variable's value was high or low.
- Importance shows what the model **uses**, not causality.
- Reuse the saved model with:

```python
import joblib
package = joblib.load("results/final_model.joblib")
predictions = package["pipeline"].predict(new_data[package["variables"]])
```

## How to run

```bash
pip install -r requirements.txt
```

1. Put your data file next to `trees_ensembles.py`.
2. Edit the `CONFIGURATION` cell (at least `FILE` and `TARGET`).
3. Run the script in Spyder (F5) or with `python trees_ensembles.py`.
4. Open `results/report.md`.

XGBoost, LightGBM and SHAP are optional: the script skips whatever is not installed.

## Example dataset

`titanic1.pkl` is a prepared version of the Titanic passengers dataset (pandas pickle): survival (`survived`, the target), ticket class (`pclass`), age, number of siblings/spouses (`sibsp`) and parents/children (`parch`) aboard, fare, and dummies for sex (`sex_male`) and port of embarkation (`embarked_Q`, `embarked_S`).

## Credits

Built from Classes 01–04 and the practical classes of "Trees, Networks and Ensemble Models" by Prof. João Mello and team (MBA in Data Science and Analytics, USP/Esalq).
