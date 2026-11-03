# -*- coding: utf-8 -*-
"""
MULTINOMIAL LOGISTIC REGRESSION: WORKING TOOL
MBA in Data Science and Analytics (USP/Esalq). Working version generated with
Claude from the "2 - Binary and Multinomial Logistic Models" course script
(Prof. Dr. Luiz Paulo Fávero). Theory: Fávero & Belfiore, Ch. 13.3.

GOAL
- Estimate the probability of each category of a Y with 3 or more
  UNORDERED classes (e.g. not late / late for the 1st class / late for the
  2nd; purchase channel; chosen plan) and classify cases.

WHAT THE SCRIPT DOES ON ITS OWN
- Chooses the reference category, creates dummies, applies stepwise (removes
  the variable that is not significant in ANY equation), computes LL, model
  chi², pseudo-R², relative risk ratios (RRR), confusion matrix,
  cross-validated accuracy and probabilities for new data.

HOW TO USE
1. Edit only the CONFIGURATION cell. 2. Run (F5) in Spyder.

Installation (once, in the Anaconda Prompt):
    pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn openpyxl
"""

#%% CONFIGURATION (the only place to edit)

FILE = "atrasado_multinomial.csv"
SHEET = None
SEPARATOR = None
TARGET = "atrasado"
REFERENCE_CATEGORY = "nao chegou atrasado"  # "auto" = most frequent category
X_VARIABLES = "auto"
ID_COLUMN = "estudante"
IGNORE_COLUMNS = []
CATEGORICAL_VARIABLES = "auto"

STEPWISE = True
P_VALUE_THRESHOLD = 0.05
CROSS_VALIDATION = 5       # stratified folds (0 = no)
SEED = 42
MAX_CLASSES = 15           # protection against a continuous Y or an ID used as target

PREDICT_FILE = None

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
from sklearn.metrics import accuracy_score, confusion_matrix
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


#%% 1. Load data and encode Y (reference = code 0)

data = read_data(FILE, SHEET, SEPARATOR)
variables = select_x(data, TARGET, X_VARIABLES, IGNORE_COLUMNS,
                     extras=[ID_COLUMN])
base = data.dropna(subset=variables + [TARGET]).reset_index(drop=True)

counts = base[TARGET].value_counts()
if len(counts) < 3:
    raise ValueError("Y has fewer than 3 categories: use model 05 (binary).")
if len(counts) > MAX_CLASSES:
    raise ValueError(f"Y has {len(counts)} categories (> {MAX_CLASSES}). "
                     "Check that you chose the right column.")
reference = (counts.idxmax() if REFERENCE_CATEGORY == "auto"
             else REFERENCE_CATEGORY)
if reference not in counts.index:
    raise ValueError(f"Reference {reference!r} does not exist in Y: "
                     f"{list(counts.index)}")
classes = [reference] + sorted([c for c in counts.index if c != reference],
                               key=str)
code = {c: i for i, c in enumerate(classes)}
y = base[TARGET].map(code).astype(int)
X, categorical = build_x_matrix(base, variables, CATEGORICAL_VARIABLES)

log("## 1. Data")
log(f"- File: `{FILE}`; rows used: {len(base)}")
log(f"- Y: `{TARGET}`; reference = **{reference}**")
log(md_table(counts.rename("n").to_frame(), 0))
log(f"- Explanatory variables: {', '.join(variables)} | dummies: "
    f"{', '.join(categorical) or 'none'}")


#%% 2. Estimation and stepwise
# MNLogit fits one equation (logit) for each category vs. the reference.
# A variable only leaves in the stepwise if it is not significant in any equation.

def fit_mnlogit(Xs):
    return sm.MNLogit(y, Xs).fit(disp=False, maxiter=500, method="bfgs")

def smallest_pvalue_per_variable(model):
    return model.pvalues.min(axis=1)

if STEPWISE:
    model, final_columns, removed = stepwise_backward(
        fit_mnlogit, X, P_VALUE_THRESHOLD, pvalues=smallest_pvalue_per_variable)
else:
    model, final_columns, removed = fit_mnlogit(X), list(X.columns), []

chi2_model = 2 * (model.llf - model.llnull)
p_model = stats.chi2.sf(chi2_model, model.df_model)
log("\n## 2. Model")
if STEPWISE:
    log("- Stepwise removed: " + (", ".join(f"{v} (smallest p={p:.3f})"
                                          for v, p in removed)
                                  or "no variable"))
log(f"- Log-likelihood: {model.llf:.3f} (null: {model.llnull:.3f})")
log(f"- Model chi²: {chi2_model:.2f} with {int(model.df_model)} df, "
    f"p = {p_model:.4g}")
log(f"- McFadden pseudo-R²: {model.prsquared:.4f} | AIC = {model.aic:.2f}")


#%% 3. Coefficients per equation and relative risk ratios (RRR)
# RRR = exp(coef): how much the odds of being in category k (vs. reference)
# are multiplied for +1 unit of X.

ci = model.conf_int()
tables = []
for j, cls in enumerate(classes[1:]):
    params = model.params.iloc[:, j]
    pvals = model.pvalues.iloc[:, j]
    ci_j = ci.xs(ci.index.get_level_values(0).unique()[j])
    tab = pd.DataFrame({"equation": f"{cls} vs {reference}",
                        "coefficient": params, "p_value": pvals,
                        "RRR": np.exp(params),
                        "RRR_ci_low": np.exp(ci_j.iloc[:, 0].values),
                        "RRR_ci_high": np.exp(ci_j.iloc[:, 1].values)})
    tables.append(tab)
coef_tab = pd.concat(tables)
coef_tab.index.name = "variable"
log("\n## 3. Coefficients and RRR (95% CI)")
log(md_table(coef_tab.reset_index().set_index(["equation", "variable"]), 4))


#%% 4. Classification (class with the highest probability)

probs = pd.DataFrame(np.asarray(model.predict(X[final_columns])),
                     columns=[f"prob_{c}" for c in classes])
pred = probs.values.argmax(axis=1)
matrix = pd.DataFrame(confusion_matrix(y, pred, labels=range(len(classes))),
                      index=[f"Actual: {c}" for c in classes],
                      columns=[f"Pred: {c}" for c in classes])
accuracy = accuracy_score(y, pred)
log("\n## 4. Classification on the fitting data")
log(md_table(matrix, 0))
log(f"- Accuracy: {accuracy:.3f} (majority class alone: "
    f"{counts.max() / counts.sum():.3f})")

fig, ax = plt.subplots(figsize=(1.5 * len(classes) + 3, 1.2 * len(classes) + 2))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_title("Confusion matrix")
save_figure(fig, "01_confusion_matrix")


#%% 5. Cross-validation (out-of-sample accuracy)

if CROSS_VALIDATION and counts.min() >= CROSS_VALIDATION:
    skf = StratifiedKFold(n_splits=CROSS_VALIDATION, shuffle=True,
                          random_state=SEED)
    hits = []
    Xf = X[final_columns]
    for train, test in skf.split(Xf, y):
        try:
            m = sm.MNLogit(y.iloc[train], Xf.iloc[train]).fit(
                disp=False, maxiter=500, method="bfgs")
            hits.append(accuracy_score(
                y.iloc[test], np.asarray(m.predict(Xf.iloc[test])).argmax(axis=1)))
        except Exception:
            continue
    if hits:
        log(f"\n## 5. Cross-validation ({CROSS_VALIDATION} folds)")
        log(f"- Out-of-sample accuracy: {np.mean(hits):.3f} ± "
            f"{np.std(hits):.3f}")


#%% 6. Save results and predictions

(OUTPUT / "statsmodels_summary.txt").write_text(str(model.summary()),
                                                encoding="utf-8")
coef_tab.to_csv(OUTPUT / "coefficients_rrr.csv", encoding="utf-8-sig")
output_base = pd.concat([base, probs], axis=1)
output_base["predicted_class"] = [classes[i] for i in pred]
output_base.to_csv(OUTPUT / "data_with_probabilities.csv", index=False,
                   encoding="utf-8-sig")

if PREDICT_FILE:
    new_data = read_data(PREDICT_FILE)
    X_new, _ = build_x_matrix(new_data, variables, categorical or [],
                              model_columns=final_columns)
    p_new = np.asarray(model.predict(X_new))
    for i, c in enumerate(classes):
        new_data[f"prob_{c}"] = p_new[:, i]
    new_data["predicted_class"] = [classes[i] for i in p_new.argmax(axis=1)]
    new_data.to_csv(OUTPUT / "new_data_predictions.csv", index=False,
                    encoding="utf-8-sig")
    log(f"\n- Predictions for {len(new_data)} new cases saved.")

log("\n## 6. How to interpret")
log(f"- Each equation compares one category with the reference ({reference}).")
log("- RRR > 1: +1 unit of X increases the odds of that category relative "
    "to the reference; RRR < 1: decreases them.")
log("- Compare the accuracy with that of the majority class: the model only "
    "adds value if it beats that baseline.")

save_report("Report: Multinomial Logistic Regression")
