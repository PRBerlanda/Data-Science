# -*- coding: utf-8 -*-
"""
MULTILEVEL MODELING (HIERARCHICAL LINEAR MODELS HLM2 AND HLM3)
MBA in Data Science and Analytics (USP/Esalq). Working version generated with
Claude from the "04 - Multilevel Modeling" course script (Prof. Dr. Luiz Paulo
Fávero). Theory: Fávero & Belfiore, Ch. 16.

GOAL
- Model a quantitative Y when observations are NESTED in groups
  (students in schools, customers in stores, employees in branches, repeated
  measures over time within individuals...).
- HLM2: 2 levels (observation → group).
- HLM3: 3 levels (measure over time → individual → group), enabled by setting
  MIDDLE_GROUP.

WHAT THE SCRIPT DOES ON ITS OWN
- Null model: ICC (how much of the variance lies between groups) and a
  likelihood-ratio test against the ordinary OLS regression.
- Tests random slopes (an effect of X that varies by group) and keeps the
  significant ones, fits the final model with fixed effects (95% CI),
  variance components, random effects per group and predictions.

HOW TO USE
1. Edit only the CONFIGURATION cell. 2. Run (F5) in Spyder.

Installation (once, in the Anaconda Prompt):
    pip install pandas numpy scipy statsmodels matplotlib seaborn openpyxl
"""

#%% CONFIGURATION (the only place to edit)

FILE = "desempenho_aluno_escola.csv"
SHEET = None
SEPARATOR = None
TARGET = "desempenho"
TOP_GROUP = "escola"               # highest level (required)
MIDDLE_GROUP = None                # e.g. "student" → HLM3 (repeated measures)
X_VARIABLES = ["horas", "texp"]    # fixed effects (level 1 and level 2); "auto" = all
IGNORE_COLUMNS = ["estudante"]
CATEGORICAL_VARIABLES = "auto"     # "auto" (text) or list
INTERACTIONS = [("horas", "texp")] # interaction pairs (e.g. cross-level interaction)

RANDOM_SLOPES = "auto"             # "auto" (tests each numeric X and keeps the
                                   # significant ones), list (e.g. ["hours"]) or []
MAX_SLOPES = 2                     # limit in "auto" mode (convergence)
ESTIMATION_METHOD = "ML"           # "ML" (allows comparing models by LR) or "REML"

PREDICT_FILE = None                # new data with the X and group columns

OUTPUT_FOLDER = "results"
SHOW_PLOTS = True
SIGNIFICANCE_LEVEL = 0.05


#%% Packages

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

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


def safe_name(name):
    """Column name usable in formulas (no spaces/symbols)."""
    clean = re.sub(r"\W", "_", str(name))
    return f"v_{clean}" if clean[0].isdigit() else clean


def lr_test(restricted, full):
    """Likelihood-ratio test between nested models."""
    dof = max(1, len(full.params) - len(restricted.params))
    lr = 2 * (full.llf - restricted.llf)
    return lr, dof, stats.chi2.sf(max(lr, 0), dof)


#%% 1. Load and prepare the data

data = read_data(FILE, SHEET, SEPARATOR)
group_cfg = [TOP_GROUP] + ([MIDDLE_GROUP] if MIDDLE_GROUP else [])
if X_VARIABLES == "auto":
    variables = [c for c in data.columns
                 if c not in [TARGET, *group_cfg, *IGNORE_COLUMNS]]
else:
    variables = list(X_VARIABLES)
missing_cols = [c for c in variables + group_cfg + [TARGET]
                if c not in data.columns]
if missing_cols:
    raise KeyError(f"Columns not found: {missing_cols}")

base = data.dropna(subset=variables + group_cfg + [TARGET]).reset_index(drop=True)
name_map = {c: safe_name(c) for c in base.columns}
df = base.rename(columns=name_map)
target, g_top = name_map[TARGET], name_map[TOP_GROUP]
g_mid = name_map[MIDDLE_GROUP] if MIDDLE_GROUP else None
xs = [name_map[v] for v in variables]
cats = ([v for v in xs if not pd.api.types.is_numeric_dtype(df[v])]
        if CATEGORICAL_VARIABLES == "auto"
        else [name_map[c] for c in CATEGORICAL_VARIABLES])
terms = [f"C({v})" if v in cats else v for v in xs]
terms += [f"{name_map[a]}:{name_map[b]}" for a, b in INTERACTIONS]
fixed_formula = f"{target} ~ " + (" + ".join(terms) if terms else "1")
reml = ESTIMATION_METHOD.upper() == "REML"
model_type = "HLM3" if g_mid else "HLM2"

sizes = df.groupby(g_top).size()
log("## 1. Data and structure")
log(f"- File: `{FILE}`; observations: {len(df)}")
log(f"- Model: **{model_type}** | top level: `{TOP_GROUP}` "
    f"({df[g_top].nunique()} groups; {sizes.min()} to {sizes.max()} "
    "observations per group)")
if g_mid:
    log(f"- Middle level: `{MIDDLE_GROUP}` "
        f"({df[[g_top, g_mid]].drop_duplicates().shape[0]} units)")
log(f"- Fixed effects: `{fixed_formula}`")
log(f"- Estimation: {'REML' if reml else 'maximum likelihood (ML)'}")
if df[g_top].nunique() < 10:
    log("- Warning: fewer than 10 groups at the top level; the variances "
        "of the random effects will be imprecise.")


def fit(formula, slopes):
    """Fit the mixed model with random intercepts (and slopes)."""
    re_formula = "1" + "".join(f" + {s}" for s in slopes)
    vc = None
    if g_mid:
        vc = {g_mid: f"0 + C({g_mid})"}
        for s in slopes:
            vc[f"{g_mid}_{s}"] = f"0 + C({g_mid}):{s}"
    model = sm.MixedLM.from_formula(formula, groups=g_top, re_formula=re_formula,
                                    vc_formula=vc, data=df)
    return model.fit(reml=reml, method=["lbfgs", "powell"])


#%% 2. Null model: ICC and test against OLS
# ICC = between-group variance / total variance. A high ICC means that a large
# part of the variation in Y is explained just by "which group" the
# observation belongs to, and ordinary regression ignores that.

null = fit(f"{target} ~ 1", [])
ols_null = sm.OLS.from_formula(f"{target} ~ 1", df).fit()
var_top = float(null.cov_re.iloc[0, 0])
var_mid = float(null.vcomp[0]) if g_mid else 0.0
var_res = float(null.scale)
total = var_top + var_mid + var_res
lr0, dof0, p0 = lr_test(ols_null, null)

log("\n## 2. Null model")
log(f"- ICC level `{TOP_GROUP}`: {var_top / total:.4f}")
if g_mid:
    log(f"- ICC level `{MIDDLE_GROUP}`: {var_mid / total:.4f}")
log(f"- LR multilevel null model x OLS null: {lr0:.2f} ({dof0} df), "
    f"p = {p0:.4g} → "
    f"{'the multilevel structure is needed' if p0 < SIGNIFICANCE_LEVEL else 'OLS may be enough'}")


#%% 3. Random slopes
# Random slope = the effect of X changes from group to group
# (e.g. the effect of study hours is different in each school).

# A random slope only makes sense for variables that VARY WITHIN the groups
# (level 1). Group-level variables (e.g. the school teacher's experience) are
# constant within the group and stay only as fixed effects.
numeric = [v for v in xs if v not in cats
           and df.groupby(g_top)[v].nunique().max() > 1]
if RANDOM_SLOPES == "auto":
    base_int = fit(fixed_formula, [])
    candidates = []
    for v in numeric:
        try:
            m = fit(fixed_formula, [v])
            lr, dof, p = lr_test(base_int, m)
            candidates.append((v, lr, p))
        except Exception:
            continue
    slopes = [v for v, _, p in sorted(candidates, key=lambda c: c[2])
              if p < SIGNIFICANCE_LEVEL][:MAX_SLOPES]
    log("\n## 3. Random slope tests")
    for v, lr, p in candidates:
        name = [k for k, s in name_map.items() if s == v][0]
        log(f"- `{name}`: LR = {lr:.2f}, p = {p:.4g}")
else:
    slopes = [name_map[v] for v in RANDOM_SLOPES]
log("- Random slopes in the final model: "
    + (", ".join(slopes) or "none (random intercepts only)"))


#%% 4. Final model

final = fit(fixed_formula, slopes)
ols_final = sm.OLS.from_formula(fixed_formula, df).fit()
lr_f, dof_f, p_f = lr_test(ols_final, final)

ci = final.conf_int()
fixed_names = final.fe_params.index
fixed_tab = pd.DataFrame({"coefficient": final.fe_params,
                          "std_error": final.bse_fe,
                          "p_value": final.pvalues[fixed_names],
                          "ci95_low": ci.loc[fixed_names, 0],
                          "ci95_high": ci.loc[fixed_names, 1]})
reverse = {s: o for o, s in name_map.items()}
fixed_tab.index = [re.sub(r"\w+", lambda m: reverse.get(m.group(0), m.group(0)), i)
                   for i in fixed_tab.index]

components = {"residual (level 1)": float(final.scale)}
for i, name in enumerate(final.cov_re.index):
    label = "intercept" if i == 0 else reverse.get(name, name)
    components[f"{TOP_GROUP}: {label}"] = float(final.cov_re.iloc[i, i])
if g_mid:
    for name, value in zip(final.model.exog_vc.names, final.vcomp):
        components[f"{MIDDLE_GROUP}: {name}"] = float(value)
var_tab = pd.Series(components, name="variance").to_frame()

log("\n## 4. Final model")
log(f"- Log-likelihood: {final.llf:.3f} (OLS with the same fixed effects: "
    f"{ols_final.llf:.3f})")
log(f"- LR multilevel x OLS: {lr_f:.2f} ({dof_f} df), p = {p_f:.4g}")
log("\n### Fixed effects (95% CI)")
log(md_table(fixed_tab, 4))
log("\n### Variance components (random effects)")
log(md_table(var_tab, 4))


#%% 5. Random effects per group and charts

effects = pd.DataFrame(final.random_effects).T
effects.columns = ["intercept" if i == 0 else reverse.get(c, c)
                   for i, c in enumerate(effects.columns)]
effects.index.name = TOP_GROUP
# In HLM3, the middle-level effects also appear here; the chart and the table
# show only the top-level ones (intercept and slopes).
effects = effects[["intercept"] + [reverse.get(s, s) for s in slopes]]

fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(effects))))
order = effects["intercept"].sort_values()
ax.barh(order.index.astype(str), order.values,
        color=np.where(order.values >= 0, "steelblue", "indianred"))
ax.axvline(0, color="black", lw=0.8)
ax.set(title=f"Random intercept by {TOP_GROUP} (deviation from the mean)",
       xlabel=f"Effect on {TARGET}", ylabel=TOP_GROUP)
save_figure(fig, "01_random_intercepts")

predicted = final.fittedvalues
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(predicted, df[target], s=8, alpha=0.5, label="Multilevel")
ax.scatter(ols_final.fittedvalues, df[target], s=8, alpha=0.3, label="OLS")
lim = [df[target].min(), df[target].max()]
ax.plot(lim, lim, ls="--", color="grey")
ax.set(title="Observed x predicted", xlabel="Predicted",
       ylabel=f"Observed ({TARGET})")
ax.legend()
save_figure(fig, "02_observed_predicted")


#%% 6. Save results and predictions

(OUTPUT / "statsmodels_summary.txt").write_text(str(final.summary()),
                                                encoding="utf-8")
fixed_tab.to_csv(OUTPUT / "fixed_effects.csv", encoding="utf-8-sig")
var_tab.to_csv(OUTPUT / "variance_components.csv", encoding="utf-8-sig")
effects.to_csv(OUTPUT / "random_effects_by_group.csv", encoding="utf-8-sig")
output_base = base.copy()
output_base["predicted_multilevel"] = predicted.values
output_base["predicted_ols"] = ols_final.fittedvalues.values
output_base.to_csv(OUTPUT / "data_with_predictions.csv", index=False,
                   encoding="utf-8-sig")

if PREDICT_FILE:
    new_data = read_data(PREDICT_FILE)
    nd = new_data.rename(columns=name_map)
    import patsy
    right_side = fixed_formula.split("~", 1)[1]
    design = patsy.dmatrix(right_side, df, return_type="dataframe").design_info
    X_new = patsy.dmatrix(design, nd, return_type="dataframe")
    fixed_part = X_new.values @ final.fe_params.values
    random_part = np.zeros(len(nd))
    for i, g in enumerate(nd[g_top]):
        if g in final.random_effects:
            re_g = final.random_effects[g]
            random_part[i] = re_g.iloc[0] + sum(re_g.get(s, 0) * nd[s].iloc[i]
                                                for s in slopes)
    new_data["predicted_fixed"] = fixed_part
    new_data["predicted_with_group_effect"] = fixed_part + random_part
    new_data.to_csv(OUTPUT / "new_data_predictions.csv", index=False,
                    encoding="utf-8-sig")
    log(f"\n- Predictions for {len(new_data)} new cases saved (new groups "
        "get only the fixed part).")

log("\n## 5. How to interpret")
log("- Fixed effects: average effect of X on Y, accounting for the group "
    "structure (correct standard errors for nested data).")
log("- Variance components: how much Y varies between groups (intercepts)"
    " and how much the effect of X varies between groups (slopes).")
log("- Random effects per group: how far above/below the mean each group is, "
    "after accounting for the model's variables (useful for ranking "
    "schools, stores, branches).")

save_report(f"Report: Multilevel Modeling ({model_type})")
