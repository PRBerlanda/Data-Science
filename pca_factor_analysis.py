# -*- coding: utf-8 -*-
"""
PRINCIPAL COMPONENT FACTOR ANALYSIS (PCA): WORKING TOOL
MBA in Data Science and Analytics (USP/Esalq). Working version generated with
Claude from the course scripts by Prof. Dr. Wilson Tarantin Junior (Grades,
São Paulo Atlas, Houses, Loans). Theory: Fávero & Belfiore, Ch. 10.

GOAL
- Summarize many correlated quantitative variables into a few FACTORS
  (linear combinations that are uncorrelated with each other).
- Common uses: build indices/rankings, reduce dimensionality before other
  models, remove multicollinearity.

WHEN TO USE
- Metric variables with relevant correlations among them (significant
  Bartlett test and acceptable KMO).
- Exploratory technique: with new observations, rerun the analysis.

HOW TO USE
1. Edit only the CONFIGURATION cell.
2. Run (F5) or cell by cell in Spyder.
3. Results in OUTPUT_FOLDER (tables, charts and report.md).

Installation (once, in the Anaconda Prompt):
    pip install pandas numpy scipy factor_analyzer matplotlib seaborn openpyxl
"""

#%% CONFIGURATION (the only place to edit)

# --- Data ----------------------------------------------------------------
FILE = "notas_fatorial.xlsx"
SHEET = None
SEPARATOR = None
ID_COLUMN = "estudante"   # row identifier (None if absent)
VARIABLES = "auto"        # list of columns or "auto" (all numeric columns)
IGNORE_COLUMNS = []       # numeric columns that must not be used
MISSING_VALUES = "drop"   # "drop" or "mean"

# --- Factors --------------------------------------------------------------
N_FACTORS = "auto"        # integer or "auto" (Kaiser criterion: eigenvalue > 1)
ROTATION = None           # None (as in class) or "varimax"
LOADING_THRESHOLD = 0.5   # minimum loading to highlight the variable in the factor

# --- Ranking (optional) ---------------------------------------------------
CREATE_RANKING = True     # sum of factors weighted by explained variance
RANKING_FACTORS = "auto"  # "auto" (retained factors) or list, e.g. [1] or [1, 2]
ORIENT_FACTORS = True     # flips the factor sign if most loadings are negative,
                          # so that "higher" = "more of the characteristic"

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
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (calculate_bartlett_sphericity,
                                             calculate_kmo)

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


#%% 1. Load the data and select variables

data = read_data(FILE, SHEET, SEPARATOR)
log("## 1. Data")
log(f"- File: `{FILE}`")
log(f"- Shape: {data.shape[0]} rows x {data.shape[1]} columns")

if VARIABLES == "auto":
    variables = [c for c in data.select_dtypes("number").columns
                 if c != ID_COLUMN and c not in IGNORE_COLUMNS]
else:
    variables = list(VARIABLES)

missing_cols = [v for v in variables if v not in data.columns]
if missing_cols:
    raise KeyError(f"Columns not found: {missing_cols}")
non_numeric = [v for v in variables
               if not pd.api.types.is_numeric_dtype(data[v])]
if non_numeric:
    raise TypeError(f"PCA uses quantitative variables. Check: {non_numeric}")
if len(variables) < 3:
    raise ValueError("Use at least 3 variables for factor analysis.")

base = data.copy()
n_missing = int(base[variables].isna().any(axis=1).sum())
if n_missing:
    if MISSING_VALUES == "mean":
        base[variables] = base[variables].fillna(base[variables].mean())
    else:
        base = base.dropna(subset=variables).reset_index(drop=True)

constants = [v for v in variables if base[v].std() == 0]
if constants:
    raise ValueError(f"Constant variables (no variation): {constants}")

X = base[variables]
log(f"- Variables ({len(variables)}): {', '.join(variables)}")
log(f"- Rows with missing values: {n_missing} ({MISSING_VALUES}); "
    f"observations analyzed: {len(X)}")
if len(X) < 5 * len(variables):
    log("- Warning: fewer than 5 observations per variable; results "
        "may be unstable.")


#%% 2. Pearson correlation matrix
# PCA only makes sense if the variables are correlated with each other: the
# shared correlation is what becomes a factor.

correlations = X.corr()
fig, ax = plt.subplots(figsize=(0.7 * len(variables) + 4,
                                0.6 * len(variables) + 3))
sns.heatmap(correlations, annot=len(variables) <= 15, fmt=".2f",
            cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
ax.set_title("Pearson correlation matrix")
save_figure(fig, "01_correlations")


#%% 3. Dataset adequacy: Bartlett and KMO
# Bartlett: H0 = the correlation matrix is the identity (no correlation).
#   p < 0.05 → there is enough correlation → PCA is adequate.
# KMO: proportion of common variance (0 to 1). Usual reference:
#   >= 0.9 marvelous; 0.8 meritorious; 0.7 middling; 0.6 mediocre; 0.5 miserable; < 0.5 unacceptable.

chi2_bartlett, p_bartlett = calculate_bartlett_sphericity(X)
kmo_per_variable, kmo_overall = calculate_kmo(X)

def classify_kmo(value):
    """Classify the KMO using Kaiser's scale."""
    for threshold, label in [(0.9, "marvelous"), (0.8, "meritorious"),
                             (0.7, "middling"), (0.6, "mediocre"),
                             (0.5, "miserable")]:
        if value >= threshold:
            return label
    return "unacceptable"

log("\n## 2. Dataset adequacy")
log(f"- Bartlett: chi² = {chi2_bartlett:.2f}, p-value = {p_bartlett:.4g} "
    f"→ {'adequate' if p_bartlett < SIGNIFICANCE_LEVEL else 'NOT adequate'}")
log(f"- Overall KMO = {kmo_overall:.3f} ({classify_kmo(kmo_overall)})")
kmo_tab = pd.Series(kmo_per_variable, index=variables, name="KMO").to_frame()
log(md_table(kmo_tab))
if p_bartlett >= SIGNIFICANCE_LEVEL:
    log("- Warning: Bartlett is not significant. The factors tend not to "
        "summarize the variables well; interpret with caution.")


#%% 4. Eigenvalues, explained variance and number of factors
# Each eigenvalue is the variance captured by one factor. Kaiser criterion:
# keep factors with eigenvalue > 1 (they explain more than a single variable).

fa_full = FactorAnalyzer(n_factors=len(variables), method="principal",
                         rotation=None).fit(X)
eigenvalues = fa_full.get_eigenvalues()[0]
eigen_table = pd.DataFrame({
    "eigenvalue": eigenvalues,
    "variance_%": eigenvalues / len(variables) * 100,
})
eigen_table["cumulative_variance_%"] = eigen_table["variance_%"].cumsum()
eigen_table.index = [f"Factor {i + 1}" for i in range(len(eigenvalues))]

n_kaiser = max(1, int((eigenvalues > 1).sum()))
n_factors = n_kaiser if N_FACTORS == "auto" else int(N_FACTORS)

log("\n## 3. Eigenvalues and explained variance")
log(md_table(eigen_table))
log(f"- Kaiser criterion (eigenvalue > 1): {n_kaiser} factor(s)")
log(f"- Factors retained: {n_factors} "
    f"({'automatic' if N_FACTORS == 'auto' else 'set in CONFIG'}), "
    f"explaining {eigen_table['cumulative_variance_%'].iloc[n_factors - 1]:.1f}% "
    "of the total variance")

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(eigen_table.index, eigen_table["variance_%"], color="steelblue")
ax.plot(eigen_table.index, eigen_table["cumulative_variance_%"],
        marker="o", color="darkorange", label="Cumulative")
ax.axvline(n_factors - 0.5, ls="--", color="grey")
ax.set(title="Variance explained by factor (scree plot)",
       xlabel="Factor", ylabel="Explained variance (%)")
ax.tick_params(axis="x", rotation=45)
ax.legend()
save_figure(fig, "02_explained_variance")


#%% 5. Retained factors: factor loadings and communalities
# Factor loading = correlation between the variable and the factor.
# Communality = share of the variable's variance explained by the retained
#   factors (sum of squared loadings). Low values: the variable is "poorly
#   represented" with this number of factors.

fa = FactorAnalyzer(n_factors=n_factors, method="principal",
                    rotation=ROTATION if n_factors > 1 else None).fit(X)
factor_names = [f"Factor {i + 1}" for i in range(n_factors)]
loadings = pd.DataFrame(fa.loadings_, index=variables, columns=factor_names)
weights = pd.DataFrame(fa.weights_, index=variables, columns=factor_names)
scores = pd.DataFrame(fa.transform(X), columns=factor_names)

if ORIENT_FACTORS:
    for f in factor_names:
        if loadings[f].sum() < 0:
            loadings[f] *= -1
            weights[f] *= -1
            scores[f] *= -1

communalities = pd.Series((loadings ** 2).sum(axis=1), name="communality")

log("\n## 4. Factor loadings and communalities")
log(f"- Rotation: {ROTATION or 'none'}")
log(md_table(loadings.join(communalities)))

low_communality = communalities[communalities < 0.5].index.tolist()
if low_communality:
    log(f"- Communality < 0.5 (poorly represented): "
        f"{', '.join(low_communality)}")

log("\n### Variables with the highest loadings on each factor "
    f"(|loading| >= {LOADING_THRESHOLD})")
for f in factor_names:
    strong = loadings[f][loadings[f].abs() >= LOADING_THRESHOLD].sort_values(ascending=False)
    description = ", ".join(f"{v} ({c:+.2f})" for v, c in strong.items()) or "none"
    log(f"- **{f}**: {description}")

fig, ax = plt.subplots(figsize=(1.3 * n_factors + 4, 0.45 * len(variables) + 2))
sns.heatmap(loadings, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
ax.set_title("Factor loadings")
save_figure(fig, "03_factor_loadings")

if n_factors >= 2:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(loadings.iloc[:, 0], loadings.iloc[:, 1], color="steelblue")
    for v in variables:
        ax.annotate(v, (loadings.loc[v].iloc[0], loadings.loc[v].iloc[1]),
                    fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1),
           title="Loading plot (Factor 1 x Factor 2)",
           xlabel=f"Factor 1 ({eigen_table['variance_%'].iloc[0]:.1f}%)",
           ylabel=f"Factor 2 ({eigen_table['variance_%'].iloc[1]:.1f}%)")
    save_figure(fig, "04_loading_plot")


#%% 6. Factor scores and ranking
# Scores = value of each observation on each factor (mean 0, std 1).
# Ranking used in class: sum of the scores weighted by each factor's explained
# variance. Check the direction: after ORIENT_FACTORS, a higher value = more
# of the characteristic represented by the positive loadings.

result = base.copy()
for f in factor_names:
    result[f"score_{f.replace(' ', '_')}"] = scores[f].values

if CREATE_RANKING:
    rank_factors = (list(range(1, n_factors + 1)) if RANKING_FACTORS == "auto"
                    else list(RANKING_FACTORS))
    rank_factors = [i for i in rank_factors if 1 <= i <= n_factors]
    score_total = sum(scores[f"Factor {i}"].values
                      * eigen_table["variance_%"].iloc[i - 1] / 100
                      for i in rank_factors)
    result["ranking_score"] = score_total
    result = result.sort_values("ranking_score", ascending=False)
    result.insert(0, "ranking_position", range(1, len(result) + 1))
    log("\n## 5. Ranking")
    log(f"- Factors used: {rank_factors} (weighted by variance)")
    top_columns = ([ID_COLUMN] if ID_COLUMN in result.columns else []) + \
        ["ranking_position", "ranking_score"]
    log("Top 10:")
    log(md_table(result[top_columns].head(10).set_index("ranking_position")))


#%% 7. Save results

result.to_csv(OUTPUT / "data_with_scores.csv", index=False, encoding="utf-8-sig")
loadings.join(communalities).to_csv(OUTPUT / "loadings_communalities.csv",
                                    encoding="utf-8-sig")
weights.to_csv(OUTPUT / "factor_score_weights.csv", encoding="utf-8-sig")
eigen_table.to_csv(OUTPUT / "eigenvalues.csv", encoding="utf-8-sig")
correlations.to_csv(OUTPUT / "correlations.csv", encoding="utf-8-sig")

log("\n## 6. How to interpret")
log("- Name each factor by looking at the variables with the highest loadings "
    "(e.g. 'performance in quantitative subjects').")
log("- The weights (factor_score_weights.csv) let you compute the score of new "
    "observations: standardize the variables (Z-score with this dataset's mean "
    "and std) and multiply by the weights.")
log("- The factors are orthogonal: they can enter a regression together "
    "without multicollinearity.")

save_report("Report: Factor Analysis (PCA)")
