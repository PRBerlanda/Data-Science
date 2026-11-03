# -*- coding: utf-8 -*-
"""
NEURAL NETWORKS (MLP) FOR TABULAR DATA: CLASSIFICATION AND REGRESSION
MBA in Data Science and Analytics (USP/Esalq). Working version generated with
Claude from the Deep Learning course (Classes 1 and 2: neuron/MLP with wine,
California Housing and Breast Cancer; Class 3.1.1: dense network in Keras)
and from "aula4_script05_keras_titanic" (Trees, Networks and Ensembles).

GOAL
- Predict a target (class or value) with a dense-layer neural network
  (multilayer perceptron), comparing it with a reference linear model.

WHAT THE SCRIPT DOES ON ITS OWN
- Detects classification x regression, builds a Pipeline without leakage
  (imputation, standardization and one-hot fitted only on the training data),
  searches the architecture (number of neurons/layers, regularization,
  learning rate) by cross-validation with early stopping, compares with
  logistic/linear regression, evaluates on the test set, computes
  permutation importance, shows the learning curve and saves the model.

ENGINES
- "sklearn" (default): MLPClassifier/MLPRegressor (no TensorFlow required).
- "keras": dense network in TensorFlow/Keras (requires `pip install tensorflow`).

Installation (once, in the Anaconda Prompt):
    pip install pandas numpy scikit-learn matplotlib seaborn openpyxl joblib
    (optional) pip install tensorflow
"""

#%% CONFIGURATION (the only place to edit)

FILE = "winequality-red.csv"
SHEET = None
SEPARATOR = None
TARGET = "quality"
X_VARIABLES = "auto"
ID_COLUMN = None
IGNORE_COLUMNS = []
PROBLEM_TYPE = "regression"     # "auto", "classification" or "regression"
POSITIVE_CLASS = "auto"         # binary: class of interest

ENGINE = "sklearn"              # "sklearn" or "keras"
# Architectures tested (each tuple = neurons per hidden layer)
ARCHITECTURES = [(8,), (16,), (32,), (16, 8), (32, 16), (64, 32)]
L2_REGULARIZATION = [1e-4, 1e-3, 1e-2]    # alpha (penalizes large weights)
LEARNING_RATES = [1e-3, 3e-3, 1e-2]
N_SEARCH_ITER = 12
MAX_EPOCHS = 500
PATIENCE = 20                   # epochs without improvement before stopping
FOLDS = 5
TEST_PROPORTION = 0.2
METRIC = "auto"                 # AUC binary, macro F1 multiclass, RMSE regression
SEED = 42

PREDICT_FILE = None
OUTPUT_FOLDER = "results"
SHOW_PLOTS = True


#%% Packages

from pathlib import Path
import os
import re
import random
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)
from sklearn.model_selection import (KFold, RandomizedSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


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


#%% 1. Data and problem type

data = read_data(FILE, SHEET, SEPARATOR).dropna(subset=[TARGET]).reset_index(drop=True)
variables = ([c for c in data.columns if c not in [TARGET, ID_COLUMN, *IGNORE_COLUMNS]]
             if X_VARIABLES == "auto" else list(X_VARIABLES))
target = data[TARGET]
if PROBLEM_TYPE == "auto":
    classification = (not pd.api.types.is_numeric_dtype(target)
                      or (target.nunique() <= 10 and (target % 1 == 0).all()))
else:
    classification = PROBLEM_TYPE == "classification"

# Column-name patterns that usually indicate identifiers (English and Portuguese)
id_pattern = re.compile(r"^(id|.*[_ ]id|id[_ ].*|code|codigo|código|cod_.*|serial.*)$", re.I)
near_id = [c for c in variables if id_pattern.match(str(c))
           and data[c].nunique() > 0.95 * len(data)]
variables = [c for c in variables if c not in near_id]
numeric = [c for c in variables if pd.api.types.is_numeric_dtype(data[c])]
categorical = [c for c in variables if c not in numeric]

binary = False
if classification:
    labels = target.astype(str)
    if labels.nunique() == 2:
        binary = True
        positive = (("1" if "1" in set(labels) else labels.value_counts().idxmin())
                    if POSITIVE_CLASS == "auto" else str(POSITIVE_CLASS))
        y = (labels == positive).astype(int)
        classes = [[c for c in labels.unique() if c != positive][0], positive]
    else:
        enc = LabelEncoder().fit(labels)
        y = pd.Series(enc.transform(labels))
        classes = list(enc.classes_)
else:
    y = target.astype(float)

metric = (("roc_auc" if binary else "f1_macro") if classification
          else "neg_root_mean_squared_error") if METRIC == "auto" else METRIC

log("## 1. Data")
log(f"- File: `{FILE}` | {len(data)} rows | target `{TARGET}` → "
    f"**{'classification' if classification else 'regression'}**")
log(f"- Explanatory variables: {len(numeric)} numeric, {len(categorical)} categorical")
log(f"- Engine: {ENGINE} | metric: {metric}")
if near_id:
    log(f"- Removed because they look like identifiers: {near_id}")

X_train, X_test, y_train, y_test = train_test_split(
    data[variables], y, test_size=TEST_PROPORTION, random_state=SEED,
    stratify=y if classification else None)
cv = (StratifiedKFold(FOLDS, shuffle=True, random_state=SEED) if classification
      else KFold(FOLDS, shuffle=True, random_state=SEED))


#%% 2. Preprocessing
# Neural networks are sensitive to scale: standardization (mean 0, std 1)
# speeds up and stabilizes training. It is fitted only on the training data
# (no leakage).

prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), categorical)])


#%% 3. Reference model (linear)

reference = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=3000)
                                       if classification else LinearRegression())])
r_ref = cross_validate(reference, X_train, y_train, cv=cv, scoring=metric)
sign = -1 if metric.startswith("neg_") else 1
ref_cv = sign * r_ref["test_score"].mean()


#%% 4. Neural network: architecture search with cross-validation

if ENGINE == "keras":
    import tensorflow as tf
    tf.random.set_seed(SEED)
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

    # The "mixin" tells scikit-learn whether this is a classifier or a regressor
    # (needed for metrics such as AUC and for stratified cross-validation).
    Mixin = ClassifierMixin if classification else RegressorMixin

    class KerasNetwork(Mixin, BaseEstimator):
        """Minimal adapter to use a Keras network inside scikit-learn."""

        def __init__(self, layers=(32, 16), alpha=1e-3, rate=1e-3):
            self.layers, self.alpha, self.rate = layers, alpha, rate

        def _build(self, n_inputs):
            reg = tf.keras.regularizers.l2(self.alpha)
            model = tf.keras.Sequential([tf.keras.Input(shape=(n_inputs,))])
            for n in self.layers:
                model.add(tf.keras.layers.Dense(n, activation="relu",
                                                kernel_regularizer=reg))
            if classification:
                n_out = 1 if binary else len(classes)
                model.add(tf.keras.layers.Dense(
                    n_out, activation="sigmoid" if binary else "softmax"))
                loss = "binary_crossentropy" if binary else "sparse_categorical_crossentropy"
            else:
                model.add(tf.keras.layers.Dense(1))
                loss = "mse"
            model.compile(optimizer=tf.keras.optimizers.Adam(self.rate), loss=loss)
            return model

        def fit(self, X, y):
            self.model_ = self._build(X.shape[1])
            stop = tf.keras.callbacks.EarlyStopping(patience=PATIENCE,
                                                    restore_best_weights=True)
            self.history_ = self.model_.fit(
                np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32"),
                validation_split=0.1, epochs=MAX_EPOCHS, batch_size=32,
                callbacks=[stop], verbose=0).history
            if classification:
                self.classes_ = np.arange(2 if binary else len(classes))
            return self

        def predict_proba(self, X):
            p = self.model_.predict(np.asarray(X, dtype="float32"), verbose=0)
            return np.hstack([1 - p, p]) if binary else p

        def predict(self, X):
            if classification:
                return self.predict_proba(X).argmax(axis=1)
            return self.model_.predict(np.asarray(X, dtype="float32"), verbose=0).ravel()

    network = KerasNetwork()
    grid = {"model__layers": ARCHITECTURES, "model__alpha": L2_REGULARIZATION,
            "model__rate": LEARNING_RATES}
    n_jobs = 1
else:
    Estimator = MLPClassifier if classification else MLPRegressor
    network = Estimator(max_iter=MAX_EPOCHS, early_stopping=True, n_iter_no_change=PATIENCE,
                        validation_fraction=0.1, random_state=SEED)
    grid = {"model__hidden_layer_sizes": ARCHITECTURES,
            "model__alpha": L2_REGULARIZATION,
            "model__learning_rate_init": LEARNING_RATES}
    n_jobs = -1

pipe = Pipeline([("prep", prep), ("model", network)])
search = RandomizedSearchCV(pipe, grid, n_iter=N_SEARCH_ITER, scoring=metric, cv=cv,
                            random_state=SEED, n_jobs=n_jobs, return_train_score=True)
search.fit(X_train, y_train)
i = search.best_index_
net_cv = sign * search.cv_results_["mean_test_score"][i]
net_std = search.cv_results_["std_test_score"][i]
net_train = sign * search.cv_results_["mean_train_score"][i]
best = search.best_estimator_
params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

metric_name = metric.replace("neg_", "")
log(f"\n## 2. Cross-validation ({FOLDS} folds, {metric_name})")
log(f"- Linear reference: {ref_cv:.4f} ± {r_ref['test_score'].std():.4f}")
log(f"- Neural network: {net_cv:.4f} ± {net_std:.4f} (train: {net_train:.4f})")
log(f"- Best architecture: {params}")
gain = (net_cv - ref_cv) * (1 if sign == 1 else -1)
log(f"- Gain over the reference: {gain:+.4f} → "
    + ("the network adds value." if gain > 0 else
       "the network does NOT beat the linear model; prefer the linear one (simpler)."))


#%% 5. Test evaluation

log("\n## 3. Test")
if classification:
    pred = best.predict(X_test)
    log(f"- Accuracy: {accuracy_score(y_test, pred):.4f} | macro F1: "
        f"{f1_score(y_test, pred, average='macro'):.4f}")
    if binary:
        log(f"- AUC: {roc_auc_score(y_test, best.predict_proba(X_test)[:, 1]):.4f}")
    fig, ax = plt.subplots(figsize=(0.9 * len(classes) + 3, 0.8 * len(classes) + 2.5))
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues",
                cbar=False, xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set(title="Confusion matrix (test)", xlabel="Predicted", ylabel="Actual")
    save_figure(fig, "01_confusion_matrix")
else:
    pred = best.predict(X_test)
    log(f"- RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.4f} | MAE: "
        f"{mean_absolute_error(y_test, pred):.4f} | R²: {r2_score(y_test, pred):.4f}")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pred, y_test, s=10, alpha=0.5)
    lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lim, lim, ls="--", color="red")
    ax.set(title="Observed x predicted (test)", xlabel="Predicted",
           ylabel=f"Observed ({TARGET})")
    save_figure(fig, "01_observed_predicted")

# Learning curve (loss per epoch)
inner_model = best.named_steps["model"]
curve = (inner_model.loss_curve_ if hasattr(inner_model, "loss_curve_")
         else inner_model.history_["loss"])
validation = (getattr(inner_model, "validation_scores_", None)
              if ENGINE == "sklearn" else inner_model.history_.get("val_loss"))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(curve, label="Training loss")
if validation is not None:
    ax2 = ax.twinx() if ENGINE == "sklearn" else ax
    ax2.plot(validation, color="darkorange",
             label="Validation score" if ENGINE == "sklearn" else "Validation loss")
    ax2.legend(loc="upper center")
ax.set(title="Learning curve", xlabel="Epoch", ylabel="Loss")
ax.legend(loc="upper right")
save_figure(fig, "02_learning_curve")
log(f"- Epochs trained (early stopping): {len(curve)}")


#%% 6. Variable importance (permutation on the test set)

perm = permutation_importance(best, X_test, y_test, scoring=metric, n_repeats=5,
                              random_state=SEED)
importance = (pd.DataFrame({"importance": perm.importances_mean,
                            "std": perm.importances_std}, index=variables)
              .sort_values("importance", ascending=False))
log("\n## 4. Permutation importance")
log(md_table(importance.head(15), 4))


#%% 7. Save model and predictions

if ENGINE == "keras":
    inner_model.model_.save(OUTPUT / "keras_network.keras")
    joblib.dump({"prep": best.named_steps["prep"], "variables": variables},
                OUTPUT / "preprocessor.joblib")
else:
    joblib.dump({"pipeline": best, "variables": variables,
                 "classes": classes if classification else None},
                OUTPUT / "final_model.joblib")
importance.to_csv(OUTPUT / "permutation_importance.csv", encoding="utf-8-sig")
pd.DataFrame(search.cv_results_).to_csv(OUTPUT / "architecture_search.csv", index=False,
                                        encoding="utf-8-sig")

if PREDICT_FILE:
    new_data = read_data(PREDICT_FILE)
    if classification:
        p = best.predict_proba(new_data[variables])
        new_data["predicted_class"] = [classes[k] for k in p.argmax(axis=1)]
        if binary:
            new_data[f"prob_{classes[1]}"] = p[:, 1]
    else:
        new_data["predicted"] = best.predict(new_data[variables])
    new_data.to_csv(OUTPUT / "new_data_predictions.csv", index=False, encoding="utf-8-sig")
    log(f"\n- Predictions for {len(new_data)} new cases saved.")

log("\n## 5. How to interpret")
log("- Neural networks shine with lots of data and non-linear relationships; "
    "on small tables, linear models or tree ensembles often tie or win, "
    "which is why the comparison with the reference is made.")
log("- Learning curve: validation loss rising while training loss falls = "
    "overfitting (increase regularization or shrink the network).")

save_report("Report: Neural Networks (MLP) for Tabular Data")
