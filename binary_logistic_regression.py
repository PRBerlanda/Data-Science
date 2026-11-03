# -*- coding: utf-8 -*-
"""
BINARY LOGISTIC REGRESSION: WORKING TOOL
MBA in Data Science and Analytics (USP/Esalq). Working version generated with
Claude from the "2 - Binary and Multinomial Logistic Models" course script
(Prof. Dr. Luiz Paulo Fávero). Theory: Fávero & Belfiore, Ch. 13.

GOAL
- Estimate the PROBABILITY of an event (Y with 2 categories: churn yes/no,
  default, loyalty, failure...) and classify cases.

WHAT THE SCRIPT DOES ON ITS OWN
- Encodes the event (1) and the non-event (0), creates dummies, applies
  stepwise, computes Wald, log-likelihood, pseudo-R², AIC/BIC, odds ratios
  with 95% CI, chooses the cutoff, builds the confusion matrix, sensitivity,
  specificity, ROC/AUC/Gini, cross-validation and probabilities for new data.

WHEN TO USE
- Binary Y. With 3+ categories → model 06 (multinomial).

HOW TO USE
1. Edit only the CONFIGURATION cell. 2. Run (F5) in Spyder.
3. Results in OUTPUT_FOLDER.

Installation (once, in the Anaconda Prompt):
    pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn openpyxl
"""

#%% CONFIGURATION (the only place to edit)

# --- Data ----------------------------------------------------------------
FILE = "dados_fidelidade.csv"
SHEET = None
SEPARATOR = None
TARGET = "fidelidade"          # Y with 2 categories
EVENT = "sim"                  # category coded as 1 ("auto": uses 1 if Y is 0/1;
                               # otherwise, the least frequent category)
X_VARIABLES = "auto"
ID_COLUMN = "id"
IGNORE_COLUMNS = []
CATEGORICAL_VARIABLES = ["sexo", "atendimento", "sortimento",
                         "acessibilidade", "preço"]
# "auto" = only text/boolean columns. Here the 1–5 ratings enter as
# categories (dummies), as in class.

# --- Modeling -------------------------------------------------------------
STEPWISE = True
P_VALUE_THRESHOLD = 0.05
CUTOFF = "auto"            # number (e.g. 0.5) or "auto" (Youden maximum:
                           # sensitivity + specificity - 1)
CROSS_VALIDATION = 5       # stratified folds for out-of-sample AUC (0 = no)
SEED = 42

# --- Prediction (optional) ------------------------------------------------
PREDICT_FILE = None

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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

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


def cutoff_metrics(y_true, prob, cutoff):
    """Sensitivity, specificity and accuracy for a given cutoff."""
    pred = (prob >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"cutoff": cutoff,
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "accuracy": (tp + tn) / len(y_true)}


#%% 1. Load data and encode Y

data = read_data(FILE, SHEET, SEPARATOR)
variables = select_x(data, TARGET, X_VARIABLES, IGNORE_COLUMNS,
                     extras=[ID_COLUMN])
base = data.dropna(subset=variables + [TARGET]).reset_index(drop=True)

y_categories = base[TARGET].unique()
if len(y_categories) != 2:
    raise ValueError(f"Y must have 2 categories; found {len(y_categories)}: "
                     f"{list(y_categories)[:10]}. For 3+, use model 06.")
if EVENT == "auto":
    event = 1 if set(y_categories) == {0, 1} else \
        base[TARGET].value_counts().idxmin()
else:
    event = EVENT
    if event not in set(y_categories):
        raise ValueError(f"EVENT = {event!r} does not exist in Y: {list(y_categories)}")
y = (base[TARGET] == event).astype(int)
X, categorical = build_x_matrix(base, variables, CATEGORICAL_VARIABLES)

log("## 1. Data")
log(f"- File: `{FILE}`; rows used: {len(base)} "
    f"(removed due to missing values: {len(data) - len(base)})")
log(f"- Event (Y = 1): `{TARGET} = {event}` → {y.sum()} cases "
    f"({y.mean():.1%}); non-event: {len(y) - y.sum()}")
log(f"- Explanatory variables: {', '.join(variables)}")
log(f"- Categorical (n-1 dummies): {', '.join(categorical) or 'none'}")
events_per_parameter = min(y.sum(), len(y) - y.sum()) / max(1, X.shape[1] - 1)
if events_per_parameter < 10:
    log(f"- Warning: {events_per_parameter:.1f} events per parameter "
        "(< 10). Estimates may be unstable.")


#%% 2. Maximum likelihood estimation and stepwise

def fit_logit(Xs):
    return sm.Logit(y, Xs).fit(disp=False, maxiter=200)

try:
    full_model = fit_logit(X)
except np.linalg.LinAlgError:
    raise RuntimeError("Singular matrix: there are redundant explanatory "
                       "variables or perfect separation. Review the variables.")
if STEPWISE:
    model, final_columns, removed = stepwise_backward(
        fit_logit, X, P_VALUE_THRESHOLD)
else:
    model, final_columns, removed = full_model, list(X.columns), []

# Likelihood-ratio chi² test (overall model significance)
chi2_model = 2 * (model.llf - model.llnull)
p_model = stats.chi2.sf(chi2_model, model.df_model)

log("\n## 2. Model")
if STEPWISE:
    log("- Stepwise removed: " + (", ".join(f"{v} (p={p:.3f})"
                                          for v, p in removed)
                                  or "no variable"))
log(f"- Log-likelihood: {model.llf:.3f} (null model: {model.llnull:.3f})")
log(f"- Model chi² (likelihood ratio): {chi2_model:.2f} "
    f"with {int(model.df_model)} df, p = {p_model:.4g}")
log(f"- McFadden pseudo-R²: {model.prsquared:.4f} | AIC = {model.aic:.2f}"
    f" | BIC = {model.bic:.2f}")


#%% 3. Coefficients, Wald and odds ratios
# Odds ratio = exp(coef): how much the ODDS of the event are multiplied for +1
# unit of X (or for the dummy's category vs. the reference).

coef_tab = coefficient_table(model, exponentiate=True, exp_name="odds_ratio")
coef_tab.insert(2, "wald_z", model.tvalues)
log("\n## 3. Coefficients and odds ratios (95% CI)")
log(md_table(coef_tab[["coefficient", "wald_z", "p_value", "odds_ratio",
                       "odds_ratio_ci_low", "odds_ratio_ci_high"]], 4))


#%% 4. Cutoff, confusion matrix and sensitivity curve

prob = model.predict(X[final_columns])
grid = pd.DataFrame([cutoff_metrics(y, prob, c)
                     for c in np.round(np.arange(0.01, 1.0, 0.01), 2)])
grid["youden"] = grid["sensitivity"] + grid["specificity"] - 1
cutoff = (float(grid.loc[grid["youden"].idxmax(), "cutoff"])
          if CUTOFF == "auto" else float(CUTOFF))

pred = (prob >= cutoff).astype(int)
matrix = pd.DataFrame(confusion_matrix(y, pred, labels=[1, 0]),
                      index=["Actual: event", "Actual: non-event"],
                      columns=["Predicted: event", "Predicted: non-event"])
m_cut, m_05 = cutoff_metrics(y, prob, cutoff), cutoff_metrics(y, prob, 0.5)

log("\n## 4. Classification")
log(f"- Cutoff used: {cutoff:.2f} "
    f"({'Youden maximum' if CUTOFF == 'auto' else 'set in CONFIG'})")
log(md_table(matrix, 0))
log(f"- Sensitivity (hit rate on events): {m_cut['sensitivity']:.3f}")
log(f"- Specificity (hit rate on non-events): {m_cut['specificity']:.3f}")
log(f"- Accuracy (overall efficiency): {m_cut['accuracy']:.3f}")
log(f"- Reference with cutoff 0.5: sens. {m_05['sensitivity']:.3f}, "
    f"spec. {m_05['specificity']:.3f}, accuracy {m_05['accuracy']:.3f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(grid["cutoff"], grid["sensitivity"], label="Sensitivity")
ax.plot(grid["cutoff"], grid["specificity"], label="Specificity")
ax.axvline(cutoff, ls="--", color="grey", label=f"Cutoff = {cutoff:.2f}")
ax.set(title="Sensitivity and specificity by cutoff", xlabel="Cutoff",
       ylabel="Hit rate")
ax.legend()
save_figure(fig, "01_sensitivity_specificity")


#%% 5. ROC curve, AUC and Gini

auc = roc_auc_score(y, prob)
fpr, tpr, _ = roc_curve(y, prob)
log(f"\n## 5. ROC\n- AUC = {auc:.4f} | Gini = 2·AUC − 1 = {2 * auc - 1:.4f}")

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(fpr, tpr, label=f"Model (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], ls="--", color="grey", label="Random")
ax.set(title="ROC curve", xlabel="1 − Specificity (false positives)",
       ylabel="Sensitivity (true positives)")
ax.legend()
save_figure(fig, "02_roc_curve")


#%% 6. Cross-validation (out-of-sample AUC)

if CROSS_VALIDATION and min(y.sum(), len(y) - y.sum()) >= CROSS_VALIDATION:
    skf = StratifiedKFold(n_splits=CROSS_VALIDATION, shuffle=True,
                          random_state=SEED)
    aucs = []
    Xf = X[final_columns]
    for train, test in skf.split(Xf, y):
        try:
            m = sm.Logit(y.iloc[train], Xf.iloc[train]).fit(disp=False,
                                                            maxiter=200)
            aucs.append(roc_auc_score(y.iloc[test], m.predict(Xf.iloc[test])))
        except Exception:
            continue
    if aucs:
        log(f"\n## 6. Cross-validation ({CROSS_VALIDATION} folds)")
        log(f"- Out-of-sample AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        log("- A fit AUC far above the out-of-sample AUC indicates "
            "overfitting.")


#%% 7. Save results and predictions

(OUTPUT / "statsmodels_summary.txt").write_text(str(model.summary()),
                                                encoding="utf-8")
coef_tab.to_csv(OUTPUT / "coefficients_odds_ratios.csv", encoding="utf-8-sig")
grid.to_csv(OUTPUT / "metrics_by_cutoff.csv", index=False, encoding="utf-8-sig")
output_base = base.copy()
output_base["y_event"] = y
output_base["event_probability"] = prob.values
output_base["classification"] = pred.values
output_base.to_csv(OUTPUT / "data_with_probabilities.csv", index=False,
                   encoding="utf-8-sig")

if PREDICT_FILE:
    new_data = read_data(PREDICT_FILE)
    X_new, _ = build_x_matrix(new_data, variables, categorical or [],
                              model_columns=final_columns)
    new_data["event_probability"] = model.predict(X_new).values
    new_data["classification"] = (new_data["event_probability"] >= cutoff).astype(int)
    new_data.to_csv(OUTPUT / "new_data_predictions.csv", index=False,
                    encoding="utf-8-sig")
    log(f"\n- Probabilities for {len(new_data)} new cases in "
        "new_data_predictions.csv")

log("\n## 7. How to interpret")
log("- Odds ratio > 1 increases the odds of the event; < 1 decreases them. "
    "E.g. OR = 1.8 → 80% higher odds, holding the other variables constant.")
log("- The cutoff is a business decision: raise sensitivity when missing an "
    "event is costly (e.g. fraud); raise specificity when a false alarm is "
    "costly.")
log("- AUC: 0.5 = random; 0.7–0.8 acceptable; 0.8–0.9 good; > 0.9 excellent.")

save_report("Report: Binary Logistic Regression")
