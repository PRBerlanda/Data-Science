# -*- coding: utf-8 -*-
"""
MODELS FOR COUNT DATA (POISSON, NEGATIVE BINOMIAL, ZIP, ZINB)
MBA in Data Science and Analytics (USP/Esalq). Working version generated with
Claude from the "03 - Models for Count Data" course script (Prof. Dr. Luiz
Paulo Fávero). Theory: Fávero & Belfiore, Ch. 14 + appendix.

GOAL
- Model a COUNT Y (0, 1, 2, ...): number of complaints, claims, defects,
  visits, violations, purchases per month...

WHAT THE SCRIPT DOES ON ITS OWN (MODEL = "auto")
1. Fits a Poisson model and tests for OVERDISPERSION (Cameron and Trivedi).
   With overdispersion → Negative Binomial (NB2).
2. Tests for EXCESS ZEROS by comparing with the zero-inflated version
   (ZIP or ZINB) using the Vuong test.
3. Reports the comparison (LL, AIC, BIC), the incidence rate ratios (IRR)
   with 95% CI, the observed x predicted distribution and the predictions.

HOW TO USE
1. Edit only the CONFIGURATION cell. 2. Run (F5) in Spyder.

Installation (once, in the Anaconda Prompt):
    pip install pandas numpy scipy statsmodels matplotlib seaborn openpyxl
"""

#%% CONFIGURATION (the only place to edit)

FILE = "corruption.csv"
SHEET = None
SEPARATOR = None
TARGET = "violations"            # integer count >= 0
X_VARIABLES = ["staff", "post", "corruption"]   # or "auto"
ID_COLUMN = "country"
IGNORE_COLUMNS = ["code"]
CATEGORICAL_VARIABLES = "auto"
EXPOSURE_COLUMN = None           # e.g. "months_observed" (rate per unit of
                                 # exposure). None = same exposure for everyone

MODEL = "auto"                   # "auto", "poisson", "negbin", "zip" or "zinb"
INFLATION_VARIABLES = ["corruption"]  # explanatory variables of the "structural
                                 # zero" part (logit) of ZI models; "auto" = same as X
STEPWISE = False                 # the class uses the full set of variables
P_VALUE_THRESHOLD = 0.05

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
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.count_model import (ZeroInflatedNegativeBinomialP,
                                              ZeroInflatedPoisson)

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


def cameron_trivedi(y, mu):
    """
    Cameron and Trivedi (1990) overdispersion test, as in class:
    auxiliary regression of ((y - mu)² - y)/mu on mu, without a constant.
    H0: equidispersion (Poisson is adequate). Returns (t, p-value).
    """
    ystar = ((y - mu) ** 2 - y) / mu
    aux = sm.OLS(ystar, mu).fit()
    return float(np.ravel(aux.tvalues)[0]), float(np.ravel(aux.pvalues)[0])


def loglik_obs(model, y):
    """Per-observation log-likelihood (needed by the Vuong test)."""
    if hasattr(model.model, "loglikeobs"):
        return np.asarray(model.model.loglikeobs(model.params))
    return stats.poisson.logpmf(y, model.fittedvalues)


def vuong(m_base, m_zi, y):
    """
    Vuong test (non-nested): zero-inflated model x base model.
    Statistic > 0 and p < 0.05 → the zero-inflated model is preferable.
    """
    diff = loglik_obs(m_zi, y) - loglik_obs(m_base, y)
    v = np.sqrt(len(diff)) * diff.mean() / diff.std(ddof=1)
    return float(v), float(stats.norm.sf(v))


def count_probabilities(model, name, X, Xi, max_k, expo):
    """Mean predicted probability of each count 0..max_k."""
    ks = np.arange(max_k + 1)
    if name in ("ZIP", "ZINB"):
        kw = {"exog_infl": Xi}
        if expo is not None:
            kw["exposure"] = expo
        return np.asarray(model.predict(X, which="prob", y_values=ks, **kw)).mean(axis=0)
    mu = np.asarray(model.predict(X, **({"exposure": expo} if expo is not None else {})))
    if name == "Poisson":
        return np.array([stats.poisson.pmf(k, mu).mean() for k in ks])
    alpha = model.params["alpha"]
    n = 1 / alpha
    p = n / (n + mu)
    return np.array([stats.nbinom.pmf(k, n, p).mean() for k in ks])


#%% 1. Load data and validate Y

data = read_data(FILE, SHEET, SEPARATOR)
variables = select_x(data, TARGET, X_VARIABLES, IGNORE_COLUMNS,
                     extras=[ID_COLUMN, EXPOSURE_COLUMN])
used_columns = variables + [TARGET] + ([EXPOSURE_COLUMN] if EXPOSURE_COLUMN else [])
base = data.dropna(subset=used_columns).reset_index(drop=True)
y = base[TARGET]
if not pd.api.types.is_numeric_dtype(y) or (y < 0).any() or (y % 1 != 0).any():
    raise ValueError("Y must be a count: integers >= 0.")
y = y.astype(int)
exposure = base[EXPOSURE_COLUMN].astype(float).values if EXPOSURE_COLUMN else None

X, categorical = build_x_matrix(base, variables, CATEGORICAL_VARIABLES)
if STEPWISE:
    _, x_columns, removed = stepwise_backward(
        lambda Xs: sm.GLM(y, Xs, family=sm.families.Poisson(),
                          exposure=exposure).fit(), X, P_VALUE_THRESHOLD)
    X = X[x_columns]

if INFLATION_VARIABLES == "auto":
    Xi = X.copy()
else:
    Xi, _ = build_x_matrix(base, list(INFLATION_VARIABLES), CATEGORICAL_VARIABLES)

mean_y, var_y = y.mean(), y.var()
log("## 1. Data")
log(f"- File: `{FILE}`; rows used: {len(base)}")
log(f"- Y: `{TARGET}` | mean = {mean_y:.3f} | variance = "
    f"{var_y:.3f} | zeros = {(y == 0).mean():.1%}")
log(f"- Explanatory variables (X): {', '.join(variables)} | dummies: "
    f"{', '.join(categorical) or 'none'}")
log(f"- Zero-inflation explanatory variables: {', '.join(Xi.columns[1:])}")
if var_y > 1.5 * mean_y:
    log("- Variance much larger than the mean: preliminary sign of "
        "overdispersion (the formal test follows).")
if STEPWISE:
    log("- Stepwise (Poisson) removed: " +
        (", ".join(v for v, _ in removed) or "none"))


#%% 2. Poisson and overdispersion test

kw_expo = {"exposure": exposure} if exposure is not None else {}
poisson = sm.GLM(y, X, family=sm.families.Poisson(), **kw_expo).fit()
t_ct, p_ct = cameron_trivedi(y.values, poisson.fittedvalues.values)
overdispersion = p_ct < SIGNIFICANCE_LEVEL

log("\n## 2. Poisson and overdispersion")
log(f"- Poisson: LL = {poisson.llf:.3f}")
log(f"- Cameron and Trivedi: t = {t_ct:.3f}, p = {p_ct:.4g} → "
    f"{'OVERDISPERSION (prefer Negative Binomial)' if overdispersion else 'equidispersion (Poisson is adequate)'}")


#%% 3. Negative Binomial (NB2)

negbin = sm.NegativeBinomial(y, X, loglike_method="nb2", **kw_expo).fit(
    disp=False, maxiter=500)
alpha = float(negbin.params["alpha"])
lr = 2 * (negbin.llf - poisson.llf)
p_lr = 0.5 * stats.chi2.sf(lr, 1)   # alpha = 0 lies on the boundary
log("\n## 3. Negative Binomial (NB2)")
log(f"- LL = {negbin.llf:.3f}; dispersion parameter α (phi) = {alpha:.4f}")
log(f"- Likelihood-ratio test α = 0 (Poisson x NegBin): "
    f"LR = {lr:.2f}, p = {p_lr:.4g}")


#%% 4. Zero-inflated models and Vuong test

models = {"Poisson": poisson, "NegBin": negbin}
vuongs = {}
try:
    zip_m = ZeroInflatedPoisson(y, X, exog_infl=Xi, inflation="logit",
                                **kw_expo).fit(disp=False, maxiter=1000,
                                               method="bfgs")
    models["ZIP"] = zip_m
    vuongs["ZIP vs Poisson"] = vuong(poisson, zip_m, y.values)
except Exception as error:
    log(f"- ZIP did not converge: {error}")
try:
    zinb_m = ZeroInflatedNegativeBinomialP(y, X, exog_infl=Xi, p=2,
                                           inflation="logit", **kw_expo).fit(
        disp=False, maxiter=2000, method="bfgs")
    models["ZINB"] = zinb_m
    vuongs["ZINB vs NegBin"] = vuong(negbin, zinb_m, y.values)
except Exception as error:
    log(f"- ZINB did not converge: {error}")

log("\n## 4. Zero inflation (Vuong test)")
for name, (v, p) in vuongs.items():
    log(f"- {name}: V = {v:.3f}, p = {p:.4g} → "
        f"{'zero-inflated is better' if p < SIGNIFICANCE_LEVEL else 'no evidence of excess zeros'}")


#%% 5. Model choice and comparison

if MODEL == "auto":
    family = "NegBin" if overdispersion else "Poisson"
    zi = "ZINB" if family == "NegBin" else "ZIP"
    vuong_key = f"{zi} vs {family}"
    chosen = (zi if vuong_key in vuongs
              and vuongs[vuong_key][1] < SIGNIFICANCE_LEVEL else family)
else:
    chosen = {"poisson": "Poisson", "negbin": "NegBin",
              "zip": "ZIP", "zinb": "ZINB"}[MODEL]
    if chosen not in models:
        raise RuntimeError(f"The {chosen} model could not be fitted.")
final = models[chosen]

comparison = pd.DataFrame({n: {"LL": m.llf, "AIC": m.aic, "BIC": m.bic,
                               "parameters": len(m.params)}
                           for n, m in models.items()}).T
log("\n## 5. Comparison and chosen model")
log(md_table(comparison, 3))
log(f"- Final model: **{chosen}** "
    f"({'automatic' if MODEL == 'auto' else 'set in CONFIG'})")


#%% 6. Coefficients and IRR
# IRR (incidence rate ratio) = exp(coef): the expected count is multiplied
# by IRR for +1 unit of X. In ZI models, the "inflate_" parameters form a
# logit of the probability of being a "structural zero".

tab = coefficient_table(final, exponentiate=True, exp_name="IRR")
inflation = tab.index.str.startswith("inflate_")
tab.loc[inflation | (tab.index == "alpha"),
        ["IRR", "IRR_ci_low", "IRR_ci_high"]] = np.nan
log("\n## 6. Final model coefficients (IRR with 95% CI)")
log(md_table(tab[["coefficient", "p_value", "IRR", "IRR_ci_low",
                  "IRR_ci_high"]], 4))
if inflation.any():
    log("- `inflate_` rows: logit coefficients of the probability of a "
        "structural zero (positive = higher chance of a certain zero).")


#%% 7. Observed x predicted distribution

max_k = int(min(y.max(), np.percentile(y, 99) + 1, 40))
observed = np.array([(y == k).mean() for k in range(max_k + 1)])
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(range(max_k + 1), observed, color="lightgrey", label="Observed")
for name, m in models.items():
    try:
        ax.plot(range(max_k + 1),
                count_probabilities(m, name, X, Xi, max_k, exposure),
                marker="o", ms=3, label=name)
    except Exception:
        continue
ax.set(title="Distribution of Y: observed x predicted by model",
       xlabel=f"{TARGET}", ylabel="Proportion")
ax.legend()
save_figure(fig, "01_observed_x_predicted")


#%% 8. Save results and predictions

def predict(model, name, Xn, Xin, expo):
    kw = {"exposure": expo} if expo is not None else {}
    if name in ("ZIP", "ZINB"):
        return np.asarray(model.predict(Xn, exog_infl=Xin, **kw))
    return np.asarray(model.predict(Xn, **kw))

(OUTPUT / "statsmodels_summary.txt").write_text(str(final.summary()),
                                                encoding="utf-8")
tab.to_csv(OUTPUT / "coefficients_irr.csv", encoding="utf-8-sig")
comparison.to_csv(OUTPUT / "model_comparison.csv", encoding="utf-8-sig")
output_base = base.copy()
for name, m in models.items():
    output_base[f"predicted_{name}"] = predict(m, name, X, Xi, exposure)
output_base.to_csv(OUTPUT / "data_with_predictions.csv", index=False,
                   encoding="utf-8-sig")

if PREDICT_FILE:
    new_data = read_data(PREDICT_FILE)
    Xn, _ = build_x_matrix(new_data, variables, categorical or [],
                           model_columns=list(X.columns))
    if INFLATION_VARIABLES == "auto":
        Xin = Xn
    else:
        Xin, _ = build_x_matrix(new_data, list(INFLATION_VARIABLES),
                                CATEGORICAL_VARIABLES,
                                model_columns=list(Xi.columns))
    expo_n = new_data[EXPOSURE_COLUMN].values if EXPOSURE_COLUMN else None
    new_data[f"expected_count_{chosen}"] = predict(final, chosen, Xn, Xin, expo_n)
    new_data.to_csv(OUTPUT / "new_data_predictions.csv", index=False,
                    encoding="utf-8-sig")
    log(f"\n- Predictions for {len(new_data)} new cases saved.")

log("\n## 7. How to interpret")
log("- IRR = 1.20 → the expected count is 20% higher for +1 unit of X "
    "(or for the dummy vs. the reference); IRR = 0.80 → 20% lower.")
log("- Ignoring overdispersion underestimates standard errors (artificially "
    "low p-values): that is why the test comes before the choice.")
log("- ZI models have two processes: one decides whether the observation is "
    "a 'certain zero'; the other generates the count for the rest.")

save_report("Report: Models for Count Data")
