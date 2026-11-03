# -*- coding: utf-8 -*-
"""
DECISION TREES AND ENSEMBLES (RANDOM FOREST, XGBOOST, LIGHTGBM)
CLASSIFICATION AND REGRESSION: WORKING TOOL
MBA in Data Science and Analytics (USP/Esalq). Working version generated with
Claude from Classes 01–04 and the practical classes of "Trees, Networks and
Ensemble Models" (Prof. João Mello and team).

GOAL
- Predict a target variable (categorical → classification; numeric →
  regression) with tree-based models, choosing the best one by
  cross-validation and explaining the predictions (permutation importance
  and SHAP).

WHAT THE SCRIPT DOES ON ITS OWN
- Detects the problem type, cleans numbers stored as text ("$1,200"), makes a
  stratified train/test split, builds a Pipeline WITHOUT data leakage
  (imputation and one-hot fitted only on the training data), compares with
  baselines, tunes hyperparameters by cross-validation, evaluates on the
  test set, measures overfitting, explains the model, saves the trained
  pipeline and predicts new data.

HOW TO USE
1. Edit only the CONFIGURATION cell. 2. Run (F5) in Spyder.

Installation (once, in the Anaconda Prompt):
    pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib seaborn openpyxl pyarrow joblib
    (xgboost, lightgbm and shap are optional: the script skips whatever is not installed)
"""

#%% CONFIGURATION (the only place to edit)

# --- Data ----------------------------------------------------------------
FILE = "titanic1.pkl"
SHEET = None
SEPARATOR = None
TARGET = "survived"
X_VARIABLES = "auto"            # list or "auto" (all except target/ID/ignored)
ID_COLUMN = None
IGNORE_COLUMNS = []
CONVERT_NUMERIC_TEXT = True     # converts columns such as "$1,200" or "1.234,5"

# Pre-split datasets (optional, e.g. HAR): X and y in different files
Y_FILE = None                   # file with only the target column (same order)
TEST_FILE = None                # ready-made test set (X), instead of a random split
TEST_Y_FILE = None              # test set target (if separate)

# --- Problem --------------------------------------------------------------
PROBLEM_TYPE = "auto"           # "auto", "classification" or "regression"
POSITIVE_CLASS = "auto"         # binary: class of interest ("auto" = 1 or the minority class)
MAX_CLASSES = 30

# --- Modeling -------------------------------------------------------------
MODELS = ["tree", "random_forest", "xgboost", "lightgbm"]
SEARCH = "random"               # "random" (RandomizedSearchCV), "grid" or None
N_SEARCH_ITER = 20              # combinations tested in the random search
FOLDS = 5
TEST_PROPORTION = 0.25
METRIC = "auto"                 # "auto" (AUC binary, macro F1 multiclass,
                                # RMSE regression) or a scikit-learn scorer name
SEED = 42
N_JOBS = -1                     # cores used by cross-validation (-1 = all)

# --- Interpretation and output -------------------------------------------
COMPUTE_SHAP = True
SHAP_SAMPLE = 500
REFIT_ON_ALL_DATA = True        # trains the final model on train + test
PREDICT_FILE = None
OUTPUT_FOLDER = "results"
SHOW_PLOTS = True


#%% Packages

from pathlib import Path
import re
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, get_scorer,
                             mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = XGBRegressor = None
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = LGBMRegressor = None
try:
    import shap
except ImportError:
    shap = None


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


def text_to_number(series):
    """
    Convert text with currency/separators into numbers when >= 90% of the
    values become valid numbers. The format is decided by the whole column:
    Brazilian if there is a decimal comma ("1.234,56", "12,5") or thousand
    dots ("44.411.238"); otherwise, English ("$3,300", "1,234.5").
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return series
    t = series.astype(str).str.strip().str.replace(r"[^\d,.\-]", "", regex=True)
    brazilian = (t.str.contains(r",\d{1,2}$").any()
                 or t.str.match(r"^-?\d{1,3}(\.\d{3}){2,}$").any())
    if brazilian:
        t = t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        t = t.str.replace(",", "", regex=False)
    num = pd.to_numeric(t, errors="coerce")
    valid = series.notna() & (series.astype(str).str.strip() != "")
    return num if valid.any() and num[valid].notna().mean() >= 0.9 else series


#%% 1. Load and prepare the data

data = read_data(FILE, SHEET, SEPARATOR)
if Y_FILE:
    y_sep = read_data(Y_FILE)
    data[TARGET] = (y_sep.iloc[:, 0] if isinstance(y_sep, pd.DataFrame)
                    else y_sep).values
if TARGET not in data.columns:
    raise KeyError(f"Target '{TARGET}' not found.")

def prepare(df):
    """Select columns and convert numbers stored as text."""
    df = df.copy()
    if CONVERT_NUMERIC_TEXT:
        for c in df.columns:
            df[c] = text_to_number(df[c])
    return df

data = prepare(data)
if X_VARIABLES == "auto":
    variables = [c for c in data.columns
                 if c not in [TARGET, ID_COLUMN, *IGNORE_COLUMNS]]
else:
    variables = list(X_VARIABLES)
data = data.dropna(subset=[TARGET]).reset_index(drop=True)

# Problem type
target = data[TARGET]
if PROBLEM_TYPE == "auto":
    classification = (not pd.api.types.is_numeric_dtype(target)
                      or pd.api.types.is_bool_dtype(target)
                      or (target.nunique() <= 10 and (target.dropna() % 1 == 0).all()))
else:
    classification = PROBLEM_TYPE == "classification"
if classification and target.nunique() > MAX_CLASSES:
    raise ValueError(f"Target with {target.nunique()} classes: use PROBLEM_TYPE='regression'.")

numeric = [c for c in variables if pd.api.types.is_numeric_dtype(data[c])]
categorical = [c for c in variables if c not in numeric]
# Column-name patterns that usually indicate identifiers (English and Portuguese)
id_pattern = re.compile(r"^(id|.*[_ ]id|id[_ ].*|code|codigo|código|cod_.*|serial.*|user id)$", re.I)
near_id = [c for c in variables
           if (c in categorical and data[c].nunique() > 0.95 * len(data))
           or (id_pattern.match(str(c)) and data[c].nunique() > 0.95 * len(data))]
if near_id:
    log(f"- Removed because they look like identifiers: {near_id}")
    categorical = [c for c in categorical if c not in near_id]
    numeric = [c for c in numeric if c not in near_id]
    variables = numeric + categorical

log("## 1. Data")
log(f"- File: `{FILE}` | {len(data)} rows | target: `{TARGET}`")
log(f"- Problem: **{'classification' if classification else 'regression'}**")
log(f"- Explanatory variables: {len(variables)} ({len(numeric)} numeric, "
    f"{len(categorical)} categorical)")
missing = data[variables].isna().mean()
if (missing > 0).any():
    log("- Missing values (imputed inside the pipeline): " + ", ".join(
        f"{c} {v:.0%}" for c, v in missing[missing > 0].items()))

if classification:
    encoder = LabelEncoder().fit(target.astype(str))
    y_all = pd.Series(encoder.transform(target.astype(str)))
    classes = list(encoder.classes_)
    binary = len(classes) == 2
    if binary:
        positive = (("1" if "1" in classes else target.astype(str).value_counts().idxmin())
                    if POSITIVE_CLASS == "auto" else str(POSITIVE_CLASS))
        y_all = (target.astype(str) == positive).astype(int)
        other = [c for c in target.astype(str).unique() if c != positive][0]
        classes = [other, str(positive)]
    log("- Target distribution: " + ", ".join(
        f"{k} ({v:.1%})" for k, v in target.astype(str).value_counts(normalize=True).items()))
else:
    y_all = target.astype(float)
    binary = False


#%% 2. Data leakage check
# A variable that predicts the target almost perfectly on its own is usually
# information from the future or derived from the target itself.

suspects = []
for c in numeric:
    x = data[c]
    if x.nunique() < 2:
        continue
    try:
        if classification and binary:
            auc_1 = roc_auc_score(y_all, x.fillna(x.median()))
            if max(auc_1, 1 - auc_1) > 0.98:
                suspects.append(f"{c} (AUC alone = {max(auc_1, 1 - auc_1):.3f})")
        elif not classification:
            r = abs(np.corrcoef(x.fillna(x.median()), y_all)[0, 1])
            if r > 0.98:
                suspects.append(f"{c} (|correlation| = {r:.3f})")
    except Exception:
        continue
log("\n## 2. Data leakage")
log("- Suspicious variables: " + (", ".join(suspects) if suspects else
                                  "none") +
    (" → confirm they would be available at prediction time; "
     "if not, add them to IGNORE_COLUMNS." if suspects else ""))


#%% 3. Train and test

X_all = data[variables]
if TEST_FILE:
    test_df = prepare(read_data(TEST_FILE))
    if TEST_Y_FILE:
        yt = read_data(TEST_Y_FILE)
        test_df[TARGET] = (yt.iloc[:, 0] if isinstance(yt, pd.DataFrame) else yt).values
    X_train, y_train = X_all, y_all
    X_test = test_df[variables]
    yt_raw = test_df[TARGET]
    if classification:
        y_test = ((yt_raw.astype(str) == positive).astype(int) if binary
                  else pd.Series(encoder.transform(yt_raw.astype(str))))
    else:
        y_test = yt_raw.astype(float)
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=TEST_PROPORTION, random_state=SEED,
        stratify=y_all if classification else None)
log(f"\n## 3. Train and test\n- Train: {len(X_train)} | test: {len(X_test)}"
    f"{' (test set provided)' if TEST_FILE else ' (stratified split)' if classification else ''}")

if METRIC == "auto":
    metric = ("roc_auc" if binary else "f1_macro") if classification \
        else "neg_root_mean_squared_error"
else:
    metric = METRIC
cv = (StratifiedKFold(FOLDS, shuffle=True, random_state=SEED) if classification
      else KFold(FOLDS, shuffle=True, random_state=SEED))
log(f"- Selection metric ({FOLDS}-fold cross-validation): `{metric}`")


#%% 4. Preprocessing pipeline (no leakage)
# Everything that "learns" from the data (median, most frequent category,
# one-hot categories) is fitted ONLY on the training part of each fold.

def preprocessor(scale=False):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                   min_frequency=0.01))]),
         categorical)], remainder="drop", verbose_feature_names_out=False)


def catalog():
    """Available models and hyperparameter search spaces."""
    c = {}
    if classification:
        c["baseline_majority"] = (DummyClassifier(strategy="prior"), {})
        c["baseline_logistic"] = (LogisticRegression(max_iter=2000), {})
        c["tree"] = (DecisionTreeClassifier(random_state=SEED),
                     {"max_depth": [2, 3, 4, 5, 6, 8, 10, None],
                      "min_samples_leaf": [1, 5, 10, 20, 50],
                      "ccp_alpha": [0, 0.0005, 0.001, 0.005, 0.01]})
        c["random_forest"] = (RandomForestClassifier(random_state=SEED, n_jobs=1),
                              {"n_estimators": [100, 200, 400],
                               "max_depth": [4, 6, 8, 12, None],
                               "min_samples_leaf": [1, 3, 5, 10],
                               "max_features": ["sqrt", 0.3, 0.5]})
        if XGBClassifier:
            c["xgboost"] = (XGBClassifier(random_state=SEED, n_jobs=1,
                                          eval_metric="logloss", verbosity=0),
                            {"n_estimators": [100, 200, 400],
                             "max_depth": [2, 3, 4, 6],
                             "learning_rate": [0.03, 0.05, 0.1, 0.2],
                             "subsample": [0.7, 0.85, 1.0],
                             "colsample_bytree": [0.6, 0.8, 1.0]})
        if LGBMClassifier:
            c["lightgbm"] = (LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1),
                             {"n_estimators": [100, 200, 400],
                              "num_leaves": [7, 15, 31, 63],
                              "learning_rate": [0.03, 0.05, 0.1],
                              "min_child_samples": [5, 10, 20, 50],
                              "colsample_bytree": [0.6, 0.8, 1.0]})
    else:
        c["baseline_mean"] = (DummyRegressor(strategy="mean"), {})
        c["baseline_linear"] = (LinearRegression(), {})
        c["tree"] = (DecisionTreeRegressor(random_state=SEED),
                     {"max_depth": [2, 3, 4, 5, 6, 8, 10, None],
                      "min_samples_leaf": [1, 5, 10, 20, 50],
                      "ccp_alpha": [0, 0.001, 0.01, 0.1]})
        c["random_forest"] = (RandomForestRegressor(random_state=SEED, n_jobs=1),
                              {"n_estimators": [100, 200, 400],
                               "max_depth": [4, 6, 8, 12, None],
                               "min_samples_leaf": [1, 3, 5, 10],
                               "max_features": [1.0, 0.5, 0.3]})
        if XGBRegressor:
            c["xgboost"] = (XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0),
                            {"n_estimators": [100, 200, 400],
                             "max_depth": [2, 3, 4, 6],
                             "learning_rate": [0.03, 0.05, 0.1, 0.2],
                             "subsample": [0.7, 0.85, 1.0],
                             "colsample_bytree": [0.6, 0.8, 1.0]})
        if LGBMRegressor:
            c["lightgbm"] = (LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1),
                             {"n_estimators": [100, 200, 400],
                              "num_leaves": [7, 15, 31, 63],
                              "learning_rate": [0.03, 0.05, 0.1],
                              "min_child_samples": [5, 10, 20, 50],
                              "colsample_bytree": [0.6, 0.8, 1.0]})
    return c


#%% 5. Fit the models with cross-validation

model_catalog = catalog()
selected = [k for k in model_catalog if k.startswith("baseline")] + \
    [k for k in MODELS if k in model_catalog]
unavailable = [k for k in MODELS if k not in model_catalog]
if unavailable:
    log(f"- Models not installed (skipped): {unavailable}")

results, fitted = [], {}
for name in selected:
    estimator, grid = model_catalog[name]
    pipe = Pipeline([("prep", preprocessor(scale=name.startswith("baseline"))),
                     ("model", estimator)])
    grid = {f"model__{k}": v for k, v in grid.items()}
    start = time.time()
    if grid and SEARCH:
        search = (RandomizedSearchCV(pipe, grid, n_iter=N_SEARCH_ITER, scoring=metric,
                                     cv=cv, random_state=SEED, n_jobs=1,
                                     return_train_score=True)
                  if SEARCH == "random" else
                  GridSearchCV(pipe, grid, scoring=metric, cv=cv, n_jobs=N_JOBS,
                               return_train_score=True))
        search.fit(X_train, y_train)
        i = search.best_index_
        cv_mean = search.cv_results_["mean_test_score"][i]
        cv_std = search.cv_results_["std_test_score"][i]
        train_cv = search.cv_results_["mean_train_score"][i]
        best_pipe = search.best_estimator_
        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    else:
        r = cross_validate(pipe, X_train, y_train, scoring=metric, cv=cv,
                           n_jobs=N_JOBS, return_train_score=True)
        cv_mean, cv_std = r["test_score"].mean(), r["test_score"].std()
        train_cv = r["train_score"].mean()
        best_pipe = pipe.fit(X_train, y_train)
        params = {}
    fitted[name] = best_pipe
    sign = -1 if metric.startswith("neg_") else 1
    results.append({"model": name, "cv_mean": sign * cv_mean,
                    "cv_std": cv_std, "train_mean": sign * train_cv,
                    "seconds": round(time.time() - start, 1),
                    "hyperparameters": params})
    print(f"  {name}: {sign * cv_mean:.4f} ± {cv_std:.4f}")

table = pd.DataFrame(results).set_index("model")
higher_is_better = not metric.startswith("neg_")
table = table.sort_values("cv_mean", ascending=not higher_is_better)
candidates = [m for m in table.index if not m.startswith("baseline")]
best = candidates[0] if candidates else table.index[0]
metric_name = metric.replace("neg_", "")

log(f"\n## 4. Cross-validation ({metric_name}; "
    f"{'higher' if higher_is_better else 'lower'} is better)")
log(md_table(table.drop(columns="hyperparameters"), 4))
log(f"- Best model: **{best}** | hyperparameters: "
    f"{table.loc[best, 'hyperparameters']}")
train_m, cv_m = table.loc[best, "train_mean"], table.loc[best, "cv_mean"]
gap = ((train_m - cv_m) if higher_is_better else (cv_m - train_m)) / max(abs(cv_m), 1e-12)
log(f"- Train = {train_m:.4f} x validation = {cv_m:.4f}: validation is "
    f"{gap:.1%} worse than training "
    + ("(warning: sign of overfitting)" if gap > 0.10 else "(acceptable)"))
baseline_names = [m for m in table.index if m.startswith("baseline")]
if baseline_names:
    log(f"- Baselines: " + ", ".join(f"{b} = {table.loc[b, 'cv_mean']:.4f}"
                                    for b in baseline_names)
        + ". A model only adds value if it beats the baselines.")


#%% 6. Final evaluation on the TEST set (used only once)

final_model = fitted[best]
log("\n## 5. Test performance")
if classification:
    pred = final_model.predict(X_test)
    prob = final_model.predict_proba(X_test)
    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="macro")
    log(f"- Accuracy: {acc:.4f} | macro F1: {f1m:.4f}")
    if binary:
        auc = roc_auc_score(y_test, prob[:, 1])
        log(f"- AUC: {auc:.4f} | Gini: {2 * auc - 1:.4f}")
        fpr, tpr, _ = roc_curve(y_test, prob[:, 1])
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot(fpr, tpr, label=f"{best} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], ls="--", color="grey")
        ax.set(title="ROC curve (test)", xlabel="False positive rate",
               ylabel="Sensitivity")
        ax.legend()
        save_figure(fig, "01_roc_curve")
    labels = classes
    cls_report = classification_report(y_test, pred, target_names=[str(r) for r in labels],
                                       output_dict=True, zero_division=0)
    log(md_table(pd.DataFrame(cls_report).T.drop(columns="support", errors="ignore"), 3))
    matrix = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(1.0 * len(labels) + 3, 0.8 * len(labels) + 2.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(title="Confusion matrix (test)", xlabel="Predicted", ylabel="Actual")
    save_figure(fig, "02_confusion_matrix")
else:
    pred = final_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    log(f"- RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    if (y_test != 0).all():
        log(f"- MAPE: {np.mean(np.abs((y_test - pred) / y_test)) * 100:.2f}%")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pred, y_test, s=10, alpha=0.5)
    lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lim, lim, ls="--", color="red")
    ax.set(title=f"Observed x predicted (test): {best}", xlabel="Predicted",
           ylabel=f"Observed ({TARGET})")
    save_figure(fig, "01_observed_predicted")


#%% 7. Interpretation: permutation importance and SHAP
# Permutation: how much the metric worsens when a variable is shuffled in the
# test set (model-agnostic). SHAP: each variable's contribution to each
# individual prediction (positive = pushes the prediction up).

# With many variables, permutation gets expensive: smaller sample and fewer repeats.
many_vars = len(variables) > 100
perm_sample = X_test.sample(min(len(X_test), 800 if many_vars else 2000),
                            random_state=SEED)
perm = permutation_importance(final_model, perm_sample, y_test.loc[perm_sample.index]
                              if hasattr(y_test, "loc") else y_test,
                              scoring=metric, n_repeats=2 if many_vars else 5,
                              random_state=SEED,
                              n_jobs=N_JOBS)
importance = (pd.DataFrame({"mean_importance": perm.importances_mean,
                            "std": perm.importances_std}, index=variables)
              .sort_values("mean_importance", ascending=False))
log("\n## 6. Permutation importance (top 15)")
log(md_table(importance.head(15), 4))
top = importance.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 0.3 * len(top) + 1.5))
ax.barh(top.index.astype(str), top["mean_importance"], xerr=top["std"],
        color="steelblue")
ax.set(title="Permutation importance (test)", xlabel=f"Loss in {metric_name}")
save_figure(fig, "03_permutation_importance")

def shap_contributions(estimator, Xs):
    """
    SHAP values per observation. XGBoost and LightGBM have a native
    computation (pred_contribs), faster and immune to version
    incompatibilities; scikit-learn trees use the shap package's TreeExplainer.
    """
    class_name = type(estimator).__name__
    if class_name.startswith("XGB"):
        import xgboost as xgb
        c = estimator.get_booster().predict(xgb.DMatrix(Xs), pred_contribs=True)
        c = c[..., :-1]                       # last column = base value
    elif class_name.startswith("LGBM"):
        c = estimator.predict(Xs, pred_contrib=True)
        if not binary and classification:     # multiclass: blocks per class
            k = len(classes)
            c = np.asarray(c).reshape(len(Xs), k, -1)[..., :-1]
        else:
            c = np.asarray(c)[:, :-1]
    else:
        if shap is None:
            raise ImportError("shap package not installed")
        c = shap.TreeExplainer(estimator).shap_values(Xs)
        if isinstance(c, list):
            c = np.stack(c, axis=1)
    c = np.asarray(c)
    if c.ndim == 3:                           # (n, classes, variables) or (n, variables, classes)
        class_axis = 1 if c.shape[1] == len(classes) else 2
        c = np.take(c, 1 if binary else 0, axis=class_axis)
    return c


if COMPUTE_SHAP and best in ("tree", "random_forest", "xgboost", "lightgbm"):
    try:
        prep = final_model.named_steps["prep"]
        estimator = final_model.named_steps["model"]
        sample = X_test.sample(min(len(X_test), SHAP_SAMPLE), random_state=SEED)
        Xs = prep.transform(sample)
        Xs = Xs.toarray() if hasattr(Xs, "toarray") else np.asarray(Xs)
        names = list(prep.get_feature_names_out())
        values = shap_contributions(estimator, Xs)
        shap_target = (f" (class {classes[1]})" if binary
                       else f" (class {classes[0]})" if classification else "")
        if shap is not None:
            plt.figure()
            shap.summary_plot(values, pd.DataFrame(Xs, columns=names), show=False,
                              max_display=15)
            fig = plt.gcf()
        else:
            mean_abs = pd.Series(np.abs(values).mean(axis=0), index=names)
            mean_abs = mean_abs.sort_values().tail(15)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(mean_abs.index, mean_abs.values, color="steelblue")
            ax.set(xlabel="mean |SHAP|")
        fig.suptitle("SHAP: impact of the variables on the predictions" + shap_target, y=1.02)
        save_figure(fig, "04_shap_summary")
        case = pd.Series(values[0], index=names).sort_values(key=np.abs).tail(10)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(case.index.astype(str), case.values,
                color=np.where(case.values >= 0, "indianred", "steelblue"))
        ax.set(title="SHAP: explanation of the 1st case in the test sample" + shap_target,
               xlabel="Contribution to the prediction (model scale)")
        save_figure(fig, "05_shap_individual_case")
        shap_mean = (pd.Series(np.abs(values).mean(axis=0), index=names)
                     .sort_values(ascending=False).rename("mean_abs_shap"))
        shap_mean.to_csv(OUTPUT / "shap_importance.csv", encoding="utf-8-sig")
        log("\n### SHAP (mean |value|, top 10)" + shap_target)
        log(md_table(shap_mean.head(10).to_frame(), 4))
    except Exception as error:
        log(f"- SHAP not computed: {type(error).__name__}: {error}")


#%% 8. Final model, predictions and saving

if REFIT_ON_ALL_DATA:
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([pd.Series(np.asarray(y_train)), pd.Series(np.asarray(y_test))],
                       ignore_index=True)
    final_model = final_model.fit(X_full.reset_index(drop=True), y_full)
package = {"pipeline": final_model, "variables": variables,
           "classification": classification, "classes": classes if classification else None,
           "metric": metric, "model": best}
joblib.dump(package, OUTPUT / "final_model.joblib")
table.to_csv(OUTPUT / "model_comparison_cv.csv", encoding="utf-8-sig")
importance.to_csv(OUTPUT / "permutation_importance.csv", encoding="utf-8-sig")

if PREDICT_FILE:
    new_data = prepare(read_data(PREDICT_FILE))
    missing_cols = [v for v in variables if v not in new_data.columns]
    for v in missing_cols:
        new_data[v] = np.nan
    if classification:
        probs = final_model.predict_proba(new_data[variables])
        if binary:
            new_data[f"prob_{classes[1]}"] = probs[:, 1]
        else:
            for i, c in enumerate(classes):
                new_data[f"prob_{c}"] = probs[:, i]
        new_data["predicted_class"] = [classes[i] for i in probs.argmax(axis=1)]
    else:
        new_data["predicted"] = final_model.predict(new_data[variables])
    new_data.to_csv(OUTPUT / "new_data_predictions.csv", index=False, encoding="utf-8-sig")
    log(f"\n- Predictions for {len(new_data)} new cases saved"
        + (f" (absent columns treated as missing: {missing_cols})" if missing_cols else ""))

log("\n## 7. How to reuse the model")
log("```python\nimport joblib\npackage = joblib.load('results/final_model.joblib')\n"
    "pipe = package['pipeline']\npredictions = pipe.predict(new_data[package['variables']])\n```")
log("- The pipeline already includes imputation and one-hot: pass the raw data.")

save_report("Report: Trees and Ensembles")
