# -*- coding: utf-8 -*-
"""
SIMPLE AND MULTIPLE LINEAR REGRESSION (OLS): WORKING TOOL
MBA in Data Science and Analytics (USP/Esalq). Working version generated with
Claude from the "01 - Simple and Multiple Regression" course script (Prof. Dr.
Luiz Paulo Fávero). Theory: Fávero & Belfiore, Ch. 12.

GOAL
- Explain/predict a QUANTITATIVE variable (Y) from quantitative and/or
  categorical explanatory variables (converted into dummies automatically).

WHAT THE SCRIPT DOES ON ITS OWN
- Dummies (n-1), stepwise procedure, assumption diagnostics (residual
  normality, Breusch-Pagan, VIF, influential points), Box-Cox transformation
  when needed, robust standard errors when there is heteroskedasticity,
  cross-validation and prediction on new data.

WHEN TO USE
- Continuous Y (price, time, score, return...). For binary Y → logistic;
  count Y (0, 1, 2...) → count models; grouped data → multilevel.

HOW TO USE
1. Edit only the CONFIGURATION cell. 2. Run (F5) in Spyder.
3. Results in OUTPUT_FOLDER (tables, charts, report.md, predictions).

Installation (once, in the Anaconda Prompt):
    pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn openpyxl
"""

#%% CONFIGURATION (the only place to edit)

# --- Data ----------------------------------------------------------------
FILE = "empresas.csv"
SHEET = None
SEPARATOR = None
TARGET = "retorno"             # Y (quantitative)
X_VARIABLES = "auto"           # list or "auto" (all except target/ID/ignored)
ID_COLUMN = "empresa"          # identifier (not used in the model). None if absent
IGNORE_COLUMNS = []
CATEGORICAL_VARIABLES = "auto" # "auto" (text/boolean) or list, e.g. ["region", "store_code"]
MISSING_VALUES = "drop"        # rows with NA in the model variables are removed

# --- Modeling -------------------------------------------------------------
STEPWISE = True                # removes non-significant explanatory variables
P_VALUE_THRESHOLD = 0.05
BOX_COX = "auto"               # "auto" (if residuals are not normal and Y > 0), True or False
ROBUST_ERRORS = "auto"         # "auto" (if Breusch-Pagan rejects), True or False
CROSS_VALIDATION = 5           # number of folds to measure out-of-sample error (0 = no)
SEED = 42

# --- Prediction (optional) ------------------------------------------------
PREDICT_FILE = None            # file with new observations (same X columns)

# --- Output ---------------------------------------------------------------
OUTPUT_FOLDER = "results"
SHOW_PLOTS = True
SIGNIFICANCE_LEVEL = 0.05


#%% Packages

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import boxcox
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

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



#%% Modeling functions (shared by the regression tools; do not edit)

def select_x(data, target, x_variables, ignore, extras=()):
    """Define the explanatory variables: the CONFIG list, or all columns (except
    target, ID and ignored ones) when X_VARIABLES = "auto"."""
    if x_variables == "auto":
        exclude = {target, *ignore, *[e for e in extras if e]}
        columns = [c for c in data.columns if c not in exclude]
    else:
        columns = list(x_variables)
    missing_cols = [c for c in columns + [target] if c not in data.columns]
    if missing_cols:
        raise KeyError(f"Columns not found: {missing_cols}")
    return columns


def build_x_matrix(df, variables, categorical="auto", max_categories=20,
                   model_columns=None):
    """
    Build the X matrix with dummies (n-1 categories, as in class).

    - categorical="auto": text, boolean and category columns become dummies;
      numeric columns stay as they are. A list of columns to treat as
      categorical (e.g. numeric region codes) is also accepted.
    - model_columns: when given (prediction on new data), aligns the columns
      with those used in the fit, filling missing dummies with 0.
    Returns X (with constant) and the list of categorical columns used.
    """
    X = df[variables].copy()
    if categorical == "auto":
        cats = [c for c in variables
                if not pd.api.types.is_numeric_dtype(X[c])
                or pd.api.types.is_bool_dtype(X[c])]
    else:
        cats = [c for c in categorical if c in variables]
    too_many = [c for c in cats if X[c].nunique() > max_categories]
    if too_many and model_columns is None:
        raise ValueError(f"Categorical variables with more than {max_categories} categories "
                         f"(possibly an ID or free text): {too_many}")
    for c in cats:
        X[c] = X[c].astype(str)
    X = pd.get_dummies(X, columns=cats, drop_first=model_columns is None,
                       dtype=float)
    X.columns = [str(c).replace(" ", "_") for c in X.columns]
    X = sm.add_constant(X, has_constant="add")
    if model_columns is not None:
        X = X.reindex(columns=model_columns, fill_value=0.0)
    return X.astype(float), cats


def stepwise_backward(fit, X, p_threshold=0.05, keep=("const",),
                      pvalues=None):
    """
    Backward stepwise procedure, same logic as the statstests package used in
    class: fit the model, remove the variable with the HIGHEST p-value above
    the threshold and repeat until all are significant.

    fit    : function that receives X and returns the fitted model.
    pvalues: optional function that extracts a Series of p-values per column
             (used in the multinomial model, which has one equation per category).
    """
    columns = list(X.columns)
    removed = []
    while True:
        model = fit(X[columns])
        p = pvalues(model) if pvalues else model.pvalues
        p = p.drop([k for k in keep if k in p.index], errors="ignore")
        if p.empty or p.max() <= p_threshold:
            return model, columns, removed
        worst = p.idxmax()
        removed.append((worst, float(p.max())))
        columns.remove(worst)


def shapiro_francia(residuals):
    """
    Shapiro-Francia normality test (Royston's approximation, the same as the
    statstests package used in class). H0: the residuals are normal.
    Returns (W, p-value).
    """
    from scipy.stats import norm
    x = np.sort(np.asarray(residuals, dtype=float))
    n = len(x)
    m = norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    W = np.corrcoef(x, m)[0, 1] ** 2
    u, v = np.log(n), np.log(np.log(n))
    mu = -1.2725 + 1.0521 * (v - u)
    sigma = 1.0308 - 0.26758 * (v + 2 / u)
    z = (np.log(1 - W) - mu) / sigma
    return W, float(norm.sf(z))


def normality_test(residuals):
    """Shapiro-Wilk for n < 30 and Shapiro-Francia for n >= 30 (class standard)."""
    from scipy.stats import shapiro
    if len(residuals) < 30:
        stat, p = shapiro(residuals)
        return "Shapiro-Wilk", stat, p
    stat, p = shapiro_francia(residuals)
    return "Shapiro-Francia", stat, p


def coefficient_table(model, exponentiate=False, exp_name="exp(coef)"):
    """Coefficients, standard error, p-value and 95% CI (optionally exponentiated)."""
    ci = model.conf_int(alpha=0.05)
    tab = pd.DataFrame({"coefficient": model.params, "std_error": model.bse,
                        "p_value": model.pvalues,
                        "ci95_low": ci.iloc[:, 0], "ci95_high": ci.iloc[:, 1]})
    if exponentiate:
        tab[exp_name] = np.exp(tab["coefficient"])
        tab[f"{exp_name}_ci_low"] = np.exp(tab["ci95_low"])
        tab[f"{exp_name}_ci_high"] = np.exp(tab["ci95_high"])
    return tab

# (end of modeling functions)


def breusch_pagan(model):
    """
    Breusch-Pagan/Cook-Weisberg test (same construction as in class):
    auxiliary regression of the standardized squared residuals on the fitted
    values. H0: homoskedasticity. Returns (chi², p-value).
    Note: the class function uses chi2.pdf*2; here the chi² tail (chi2.sf)
    is used, which is the correct form of the p-value.
    """
    resid, fitted = model.resid, model.fittedvalues
    up = resid ** 2 / np.mean(resid ** 2)
    aux = sm.OLS(up, sm.add_constant(fitted)).fit()
    chi2 = aux.ess / 2
    return chi2, stats.chi2.sf(chi2, 1)


def inverse_boxcox(value, lmbda):
    """Bring the prediction back from the Box-Cox scale to the original Y scale."""
    if lmbda == 0:
        return np.exp(value)
    return np.power(np.maximum(lmbda * value + 1, 1e-12), 1 / lmbda)


#%% 1. Load data and build X and Y

data = read_data(FILE, SHEET, SEPARATOR)
variables = select_x(data, TARGET, X_VARIABLES, IGNORE_COLUMNS,
                     extras=[ID_COLUMN])
if not pd.api.types.is_numeric_dtype(data[TARGET]):
    raise TypeError(f"The target variable '{TARGET}' must be numeric.")

base = data.dropna(subset=variables + [TARGET]).reset_index(drop=True)
y = base[TARGET].astype(float)
X, categorical = build_x_matrix(base, variables, CATEGORICAL_VARIABLES)

log("## 1. Data")
log(f"- File: `{FILE}` ({data.shape[0]} rows)")
log(f"- Rows removed due to missing values: {len(data) - len(base)}; "
    f"used: {len(base)}")
log(f"- Y: `{TARGET}` | explanatory: {', '.join(variables)}")
log(f"- Categorical variables converted into dummies (n-1): "
    f"{', '.join(categorical) or 'none'}")
if len(base) < 10 * (X.shape[1] - 1):
    log("- Warning: fewer than 10 observations per parameter; the model may "
        "be unstable (overfitting).")
log("\n### Descriptive statistics")
log(md_table(base[[TARGET] + [v for v in variables
                              if v not in categorical]].describe().T))


#%% 2. Full model and stepwise

def fit_ols(Xs, ys=None):
    return sm.OLS(y if ys is None else ys, Xs).fit()

full_model = fit_ols(X)
if STEPWISE:
    model, final_columns, removed = stepwise_backward(
        fit_ols, X, P_VALUE_THRESHOLD)
else:
    model, final_columns, removed = full_model, list(X.columns), []

log("\n## 2. Model")
log(f"- Full model: R² = {full_model.rsquared:.4f}; "
    f"adjusted R² = {full_model.rsquared_adj:.4f}")
if STEPWISE:
    log("- Stepwise removed: " + (", ".join(f"{v} (p={p:.3f})"
                                          for v, p in removed)
                                  or "no variable"))
log(f"- Final model: R² = {model.rsquared:.4f}; adjusted R² = "
    f"{model.rsquared_adj:.4f}; F = {model.fvalue:.2f} "
    f"(p = {model.f_pvalue:.4g}); AIC = {model.aic:.2f}")


#%% 3. Assumption diagnostics

def diagnose(m, label):
    """Residual normality and Breusch-Pagan for model m."""
    test_name, stat, p_norm = normality_test(m.resid)
    chi2_bp, p_bp = breusch_pagan(m)
    log(f"- [{label}] {test_name}: statistic = {stat:.4f}, "
        f"p = {p_norm:.4f} → residuals "
        f"{'normal' if p_norm >= SIGNIFICANCE_LEVEL else 'NOT normal'}")
    log(f"- [{label}] Breusch-Pagan: chi² = {chi2_bp:.3f}, p = {p_bp:.4f}"
        f" → {'homoskedasticity' if p_bp >= SIGNIFICANCE_LEVEL else 'HETEROSKEDASTICITY'}")
    return p_norm, p_bp

log("\n## 3. Assumptions")
p_norm, p_bp = diagnose(model, "linear")

# Multicollinearity (VIF > 10 is a serious problem; > 5 deserves attention)
X_final = X[final_columns]
vif = pd.DataFrame({"VIF": [variance_inflation_factor(X_final.values, i)
                            for i in range(X_final.shape[1])]},
                   index=X_final.columns).drop("const", errors="ignore")
vif["Tolerance"] = 1 / vif["VIF"]
if len(vif):
    log(md_table(vif))
    if (vif["VIF"] > 5).any():
        log("- Warning: VIF > 5 indicates relevant multicollinearity.")

# Influential points (Cook's distance > 4/n)
cook = model.get_influence().cooks_distance[0]
influential = np.where(cook > 4 / len(base))[0]
log(f"- Influential observations (Cook > 4/n): {len(influential)}")


#%% 4. Box-Cox (when the residuals are not normal)
# Box-Cox finds the λ that makes Y as close as possible to normal:
# Y* = (Y^λ - 1)/λ. The model is fitted on Y* and the predictions are brought
# back to the original scale. Requires Y > 0.

lmbda = None
use_bc = (BOX_COX is True or
          (BOX_COX == "auto" and p_norm < SIGNIFICANCE_LEVEL))
if use_bc and (y <= 0).any():
    log("- Box-Cox not applied: Y has values <= 0.")
    use_bc = False

if use_bc:
    y_bc, lmbda = boxcox(y)
    y_bc = pd.Series(y_bc, index=y.index)

    def fit_bc(Xs):
        return sm.OLS(y_bc, Xs).fit()

    if STEPWISE:
        model_bc, columns_bc, removed_bc = stepwise_backward(
            fit_bc, X, P_VALUE_THRESHOLD)
    else:
        model_bc, columns_bc = fit_bc(X), list(X.columns)
    log(f"\n## 4. Box-Cox (λ = {lmbda:.4f})")
    p_norm_bc, p_bp_bc = diagnose(model_bc, "Box-Cox")
    if BOX_COX is True or p_norm_bc > p_norm:
        log("- Box-Cox model adopted as final (residuals closer to "
            "normality).")
        model, final_columns, p_norm, p_bp = model_bc, columns_bc, p_norm_bc, p_bp_bc
    else:
        log("- Box-Cox did not improve normality; the linear model was kept.")
        lmbda = None


#%% 5. Coefficients, confidence intervals and robust errors

coef_tab = coefficient_table(model)
log("\n## 5. Final model coefficients" +
    (f" (Y transformed by Box-Cox, λ = {lmbda:.4f})" if lmbda else ""))
log(md_table(coef_tab, 4))

use_robust = (ROBUST_ERRORS is True or
              (ROBUST_ERRORS == "auto" and p_bp < SIGNIFICANCE_LEVEL))
if use_robust:
    robust = model.get_robustcov_results(cov_type="HC3")
    robust_tab = pd.DataFrame({"coefficient": robust.params,
                               "robust_std_error": robust.bse,
                               "robust_p_value": robust.pvalues},
                              index=model.params.index)
    log("\n### Robust standard errors (Huber-White HC3)")
    log("There is heteroskedasticity: use these p-values for inference.")
    log(md_table(robust_tab, 4))
    robust_tab.to_csv(OUTPUT / "robust_coefficients.csv", encoding="utf-8-sig")


#%% 6. Cross-validation (out-of-sample error, on the original Y scale)

if CROSS_VALIDATION and len(base) >= 2 * CROSS_VALIDATION:
    kf = KFold(n_splits=CROSS_VALIDATION, shuffle=True, random_state=SEED)
    metrics = []
    Xf = X[final_columns]
    for train, test in kf.split(Xf):
        if lmbda:
            yt, lam_fold = boxcox(y.iloc[train])
            m = sm.OLS(yt, Xf.iloc[train]).fit()
            pred = inverse_boxcox(m.predict(Xf.iloc[test]), lam_fold)
        else:
            m = sm.OLS(y.iloc[train], Xf.iloc[train]).fit()
            pred = m.predict(Xf.iloc[test])
        actual = y.iloc[test]
        metrics.append({
            "RMSE": np.sqrt(np.mean((actual - pred) ** 2)),
            "MAE": np.mean(np.abs(actual - pred)),
            "R2": 1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2)})
    cv = pd.DataFrame(metrics)
    log(f"\n## 6. Cross-validation ({CROSS_VALIDATION} folds)")
    log(f"- Out-of-sample R²: {cv['R2'].mean():.4f} ± {cv['R2'].std():.4f}")
    log(f"- RMSE: {cv['RMSE'].mean():.4f} ± {cv['RMSE'].std():.4f} | "
        f"MAE: {cv['MAE'].mean():.4f} ± {cv['MAE'].std():.4f}")
    log("- If the out-of-sample R² is much lower than the fit R², "
        "the model is overfitted.")


#%% 7. Diagnostic charts

fitted = (inverse_boxcox(model.fittedvalues, lmbda) if lmbda
          else model.fittedvalues)
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes[0, 0].scatter(fitted, y, s=12, alpha=0.6)
lim = [min(y.min(), fitted.min()), max(y.max(), fitted.max())]
axes[0, 0].plot(lim, lim, color="red", ls="--")
axes[0, 0].set(title="Observed x Predicted", xlabel="Predicted",
               ylabel=f"Observed ({TARGET})")
axes[0, 1].scatter(model.fittedvalues, model.resid, s=12, alpha=0.6)
axes[0, 1].axhline(0, color="red", ls="--")
axes[0, 1].set(title="Residuals x Fitted (look for no pattern)",
               xlabel="Fitted values", ylabel="Residuals")
sm.qqplot(model.resid, line="s", ax=axes[1, 0])
axes[1, 0].set_title("Residual Q-Q plot")
sns.histplot(model.resid, kde=True, ax=axes[1, 1])
axes[1, 1].set(title="Residual distribution", xlabel="Residual",
               ylabel="Frequency")
save_figure(fig, "01_residual_diagnostics")


#%% 8. Save results and predictions

(OUTPUT / "statsmodels_summary.txt").write_text(str(model.summary()),
                                                encoding="utf-8")
coef_tab.to_csv(OUTPUT / "coefficients.csv", encoding="utf-8-sig")
output_base = base.copy()
output_base["predicted"] = fitted.values
output_base["residual"] = y.values - fitted.values
output_base["cook_distance"] = cook
output_base.to_csv(OUTPUT / "data_with_predictions.csv", index=False,
                   encoding="utf-8-sig")

if PREDICT_FILE:
    new_data = read_data(PREDICT_FILE)
    X_new, _ = build_x_matrix(new_data, variables, categorical or [],
                              model_columns=final_columns)
    prediction = model.get_prediction(X_new).summary_frame(alpha=0.05)
    result = new_data.copy()
    pred_columns = {"mean": "predicted", "obs_ci_lower": "prediction_interval_low",
                    "obs_ci_upper": "prediction_interval_high"}
    for source, target_col in pred_columns.items():
        values = prediction[source].values
        result[target_col] = inverse_boxcox(values, lmbda) if lmbda else values
    result.to_csv(OUTPUT / "new_data_predictions.csv", index=False,
                  encoding="utf-8-sig")
    log(f"\n- Predictions for {len(new_data)} new observations saved to "
        "new_data_predictions.csv (with 95% prediction interval).")

log("\n## 7. How to interpret")
log("- Coefficient: average change in Y for +1 unit of the explanatory "
    "variable, holding the others constant (on the Box-Cox scale, when "
    "applied: interpret the sign and use the predictions for the effect on Y).")
log("- Dummy: average difference in Y relative to the reference category "
    "(the one that does not appear in the table).")
log("- p-value < 0.05: the effect is statistically different from zero. "
    "Also assess whether the effect size is relevant to the business.")

save_report("Report: Linear Regression")
