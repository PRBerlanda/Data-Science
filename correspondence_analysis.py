# -*- coding: utf-8 -*-
"""
SIMPLE (ANACOR) AND MULTIPLE (MCA) CORRESPONDENCE ANALYSIS: WORKING TOOL
MBA in Data Science and Analytics (USP/Esalq). Working version generated with
Claude from the course scripts by Prof. Dr. Wilson Tarantin Junior (Investment
Profile, Investment Profile by Marital Status, Heart Disease, Management
(Likert), Student Adaptation) and the ANACOR/MCA summary (Claude).
Theory: Fávero & Belfiore, Ch. 11.

GOAL
- Check whether CATEGORICAL variables are associated and WHICH categories go
  together, summarizing everything in a perceptual map.
- 2 variables → ANACOR (chosen automatically). 3 or more → MCA.

WHEN TO USE
- Surveys, customer profiles, Likert scales, cross-tabulation of attributes.
- Quantitative variables are automatically binned into ranges.
- Exploratory technique: it produces no equation or prediction.

HOW TO USE
1. Edit only the CONFIGURATION cell.
2. Run (F5) or cell by cell in Spyder.
3. Results in OUTPUT_FOLDER (tables, maps and report.md).

Installation (once, in the Anaconda Prompt):
    pip install pandas numpy scipy statsmodels prince matplotlib seaborn openpyxl
"""

#%% CONFIGURATION (the only place to edit)

# --- Data ----------------------------------------------------------------
FILE = "dados_cor_acm.xlsx"
SHEET = None
SEPARATOR = None
ID_COLUMN = None          # row identifier (not used in the analysis)
VARIABLES = "auto"        # list of columns or "auto" (all except the ID)
IGNORE_COLUMNS = []
FOCUS_VARIABLE = "doenca.card"  # optional: colors the observation map and
                                # details residuals of pairs with it. None = no focus

# --- Quantitative variables -----------------------------------------------
MAX_CATEGORIES = 10       # numeric with up to N distinct values = categorical
N_BINS = 3                # numeric with more values become N bins (quantiles)

# --- Analysis -------------------------------------------------------------
DROP_UNASSOCIATED = True  # MCA: remove variables with no significant
                          # association with any other (rule used in class)
N_DIMENSIONS = 2          # dimensions computed (the map uses the first 2)
SIGNIFICANCE_LEVEL = 0.05

# --- Output ---------------------------------------------------------------
OUTPUT_FOLDER = "results"
SHOW_PLOTS = True
SHOW_OBSERVATIONS = True  # observation points on the map


#%% Packages

from pathlib import Path
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2_contingency
import prince

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


def label_points(ax, df, x="x", y="y", label="category"):
    """Write each category's name next to its point on the map."""
    for _, row in df.iterrows():
        ax.annotate(row[label], (row[x], row[y]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")


def map_axes(ax, inertia_pct, title):
    """Reference lines at zero and % of inertia on each axis."""
    ax.axhline(0, color="lightgrey", ls="--", lw=0.8)
    ax.axvline(0, color="lightgrey", ls="--", lw=0.8)
    ax.set(title=title,
           xlabel=f"Dimension 1 ({inertia_pct[0]:.1f}% of inertia)",
           ylabel=f"Dimension 2 ({inertia_pct[1]:.1f}% of inertia)")


def adjusted_residuals(table):
    """Adjusted standardized residuals of a contingency table."""
    return sm.stats.Table(table).standardized_resids


#%% 1. Load the data

data = read_data(FILE, SHEET, SEPARATOR)
log("## 1. Data")
log(f"- File: `{FILE}`")
log(f"- Shape: {data.shape[0]} rows x {data.shape[1]} columns")

variables = ([c for c in data.columns
              if c != ID_COLUMN and c not in IGNORE_COLUMNS]
             if VARIABLES == "auto" else list(VARIABLES))
missing_cols = [v for v in variables if v not in data.columns]
if missing_cols:
    raise KeyError(f"Columns not found: {missing_cols}")
if len(variables) < 2:
    raise ValueError("At least 2 variables are required.")

base = data[variables].copy()
n_missing = int(base.isna().any(axis=1).sum())
base = base.dropna().reset_index(drop=True)
log(f"- Rows removed due to missing values: {n_missing}; analyzed: {len(base)}")


#%% 2. Prepare the categorical variables
# The technique only accepts categories. Numeric variables with many values
# are split into N_BINS equal-size bins (quantiles), as in class (pd.qcut).

labels_3 = ["low", "medium", "high"]
binned = []
for v in variables:
    series = base[v]
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > MAX_CATEGORIES:
        labels = (labels_3 if N_BINS == 3
                  else [f"bin{i + 1}" for i in range(N_BINS)])
        base[v] = pd.qcut(series, N_BINS, labels=[f"{v}_{r}" for r in labels],
                          duplicates="drop")
        binned.append(v)
    base[v] = base[v].astype(str)

id_like = [v for v in variables if base[v].nunique() > 0.5 * len(base)]
if id_like:
    raise ValueError("These columns look like identifiers (almost one value per "
                     f"row). Use ID_COLUMN or IGNORE_COLUMNS: {id_like}")
single_category = [v for v in variables if base[v].nunique() < 2]
if single_category:
    raise ValueError(f"Variables with a single category: {single_category}")

log(f"- Variables ({len(variables)}): {', '.join(variables)}")
log(f"- Quantitative variables binned into {N_BINS} ranges: "
    f"{', '.join(binned) or 'none'}")
log("\n### Categories and frequencies")
for v in variables:
    log(f"- **{v}**: " + ", ".join(f"{k} ({n})" for k, n in
                                   base[v].value_counts().items()))


#%% 3. Chi-square tests for all pairs
# H0: the variables are associated at random (independence).
# p < 0.05 → there is association. Validity: chi² requires EXPECTED
# frequencies >= 5 (the script reports the % of cells below that).

rows = []
for v1, v2 in combinations(variables, 2):
    tab = pd.crosstab(base[v1], base[v2])
    chi2, p, dof, expected = chi2_contingency(tab)
    rows.append({"var1": v1, "var2": v2, "chi2": chi2, "dof": dof,
                 "p_value": p,
                 "%_expected<5": (expected < 5).mean() * 100})
tests = pd.DataFrame(rows).sort_values("p_value")
log("\n## 2. Chi-square tests (pairs)")
log(md_table(tests.set_index(["var1", "var2"]), 4))
if (tests["%_expected<5"] > 20).any():
    log("- Warning: some pairs have more than 20% of expected frequencies "
        "below 5. Consider merging rare categories.")

significant = tests[tests["p_value"] < SIGNIFICANCE_LEVEL]
associated = set(significant["var1"]) | set(significant["var2"])
unassociated = [v for v in variables if v not in associated]

if len(variables) == 2 and not len(significant):
    log("- The two variables are NOT associated (p >= "
        f"{SIGNIFICANCE_LEVEL}). The perceptual map must not be "
        "interpreted as association.")
elif unassociated and len(variables) > 2:
    log(f"- No significant association with any other variable: "
        f"{', '.join(unassociated)}")
    if DROP_UNASSOCIATED:
        variables = [v for v in variables if v not in unassociated]
        log("  → removed from the MCA (DROP_UNASSOCIATED = True).")
        if FOCUS_VARIABLE in unassociated:
            log(f"  → the focus variable `{FOCUS_VARIABLE}` left the analysis.")
if len(variables) < 2:
    raise ValueError("Fewer than 2 associated variables remain; there is no map to build.")

technique = "ANACOR" if len(variables) == 2 else "MCA"
log(f"\n- Technique applied: **{technique}** with {len(variables)} variables")


#%% 4. Adjusted standardized residuals: WHERE the association is
# Adjusted residual > 1.96 → the two categories appear together more than
# expected by chance (5% significance). < -1.96 → they appear together less.

residual_pairs = ([(variables[0], variables[1])] if technique == "ANACOR" else
                  [(a, b) for a, b in combinations(variables, 2)
                   if FOCUS_VARIABLE in (a, b)])
log("\n## 3. Adjusted standardized residuals")
if not residual_pairs:
    log("- Set FOCUS_VARIABLE to detail the residuals of a variable of "
        "interest against the others.")
for v1, v2 in residual_pairs:
    tab = pd.crosstab(base[v1], base[v2])
    res = adjusted_residuals(tab)
    res.to_csv(OUTPUT / f"residuals_{v1}_x_{v2}.csv", encoding="utf-8-sig")
    highlight = [(i, j, res.loc[i, j]) for i in res.index for j in res.columns
                 if res.loc[i, j] > 1.96]
    log(f"- **{v1} x {v2}**: significant positive associations: "
        + ("; ".join(f"{v1}={i} ↔ {v2}={j} ({r:.2f})"
                     for i, j, r in highlight)
           or "none"))
    fig, ax = plt.subplots(figsize=(1.1 * tab.shape[1] + 3,
                                    0.6 * tab.shape[0] + 2))
    sns.heatmap(res, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-4, vmax=4, ax=ax)
    ax.set_title(f"Adjusted residuals: {v1} x {v2} (|r| > 1.96 = significant)")
    save_figure(fig, f"residuals_{v1}_x_{v2}")


#%% 5. Fit the model and get the eigenvalues (inertia)

if technique == "ANACOR":
    table = pd.crosstab(base[variables[0]], base[variables[1]])
    n_dim = max(1, min(N_DIMENSIONS, min(table.shape) - 1))
    model = prince.CA(n_components=n_dim).fit(table)
    eigenvalues = np.asarray(model.eigenvalues_)
    total_inertia = float(model.total_inertia_)
    inertia_pct = eigenvalues / total_inertia * 100
    row_coords = model.row_coordinates(table)
    col_coords = model.column_coordinates(table)
    categories = pd.concat([
        pd.DataFrame({"variable": variables[0], "category": table.index,
                      "x": row_coords.iloc[:, 0].values,
                      "y": row_coords.iloc[:, 1].values if n_dim > 1 else 0}),
        pd.DataFrame({"variable": variables[1], "category": table.columns,
                      "x": col_coords.iloc[:, 0].values,
                      "y": col_coords.iloc[:, 1].values if n_dim > 1 else 0}),
    ], ignore_index=True)
    # Each observation sits at the mean of its two categories' coordinates
    obs = base[variables].copy()
    for v in variables:
        cat_map = categories[categories["variable"] == v].set_index("category")
        obs[f"x_{v}"] = obs[v].map(cat_map["x"])
        obs[f"y_{v}"] = obs[v].map(cat_map["y"])
    obs_coords = pd.DataFrame({
        "x": obs[[f"x_{v}" for v in variables]].mean(axis=1),
        "y": obs[[f"y_{v}" for v in variables]].mean(axis=1)})
    log("\n## 4. ANACOR: eigenvalues and inertia")
    log(f"- Total principal inertia = chi²/N = {total_inertia:.4f}")
else:
    mca_data = base[variables]
    J = sum(mca_data[v].nunique() for v in variables)
    Q = len(variables)
    n_dim = max(1, min(N_DIMENSIONS, J - Q))
    model = prince.MCA(n_components=n_dim).fit(mca_data)
    eigenvalues = np.asarray(model.eigenvalues_)
    total_inertia = float(model.total_inertia_)
    inertia_pct = eigenvalues / total_inertia * 100
    # Standard coordinates (binary matrix), used in the map shown in class
    std_coords = model.column_coordinates(mca_data) / np.sqrt(eigenvalues)

    def split_label(label):
        """Split prince's label ('var_category') into variable and category."""
        for v in sorted(variables, key=len, reverse=True):
            if label.startswith(v + "_"):
                return v, label[len(v):].lstrip("_")
        return label, label

    categories = pd.DataFrame({"x": std_coords.iloc[:, 0].values,
                               "y": std_coords.iloc[:, 1].values if n_dim > 1 else 0})
    categories[["variable", "category"]] = [split_label(r) for r in std_coords.index]
    repeated = categories["category"].duplicated(keep=False)
    categories.loc[repeated, "category"] = (categories["variable"] + ": "
                                            + categories["category"])
    obs_coords = model.row_coordinates(mca_data).iloc[:, :2]
    obs_coords.columns = ["x", "y"][:obs_coords.shape[1]]
    log("\n## 4. MCA: eigenvalues and inertia")
    log(f"- Categories (J) = {J}; variables (Q) = {Q}; possible dimensions "
        f"= J - Q = {J - Q}")
    log(f"- Total principal inertia = (J - Q)/Q = {total_inertia:.4f}; "
        f"mean inertia per dimension = {total_inertia / (J - Q):.4f}")

eigen_tab = pd.DataFrame({"eigenvalue": eigenvalues,
                          "inertia_%": inertia_pct,
                          "cumulative_inertia_%": np.cumsum(inertia_pct)},
                         index=[f"Dim {i + 1}" for i in range(len(eigenvalues))])
log(md_table(eigen_tab, 4))
if len(inertia_pct) < 2:
    inertia_pct = np.append(inertia_pct, 0.0)


#%% 6. Perceptual map
# Categories close to each other on the map are associated. Read it together
# with the residuals: the map shows proximity; the residuals confirm significance.

fig, ax = plt.subplots(figsize=(9, 6))
if SHOW_OBSERVATIONS and "y" in obs_coords:
    ax.scatter(obs_coords["x"], obs_coords["y"], s=6, color="lightgrey",
               alpha=0.5, label="Observations")
sns.scatterplot(data=categories, x="x", y="y", hue="variable", s=50, ax=ax)
label_points(ax, categories)
map_axes(ax, inertia_pct, f"Perceptual map: {technique}")
ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
save_figure(fig, "perceptual_map")

if FOCUS_VARIABLE in variables and "y" in obs_coords:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x=obs_coords["x"], y=obs_coords["y"],
                    hue=base[FOCUS_VARIABLE].values, s=15, ax=ax)
    map_axes(ax, inertia_pct, f"Observations colored by {FOCUS_VARIABLE}")
    save_figure(fig, "observations_map_focus")


#%% 7. Save results

categories.to_csv(OUTPUT / "category_coordinates.csv", index=False,
                  encoding="utf-8-sig")
obs_output = pd.concat([base.reset_index(drop=True),
                        obs_coords.reset_index(drop=True).add_prefix("coord_")],
                       axis=1)
obs_output.to_csv(OUTPUT / "observations_with_coordinates.csv", index=False,
                  encoding="utf-8-sig")
tests.to_csv(OUTPUT / "chi2_tests.csv", index=False, encoding="utf-8-sig")
eigen_tab.to_csv(OUTPUT / "eigenvalues_inertia.csv", encoding="utf-8-sig")

log("\n## 5. How to interpret")
log("1. If the chi² p-value < 0.05, the variables are associated.")
log("2. Category pairs with adjusted residual > 1.96 go together.")
log("3. The map summarizes the associations; the sum of the % of inertia of "
    "the two axes shows how much the 2D map represents. In MCA this "
    "percentage is usually low: read the map as a summary, not as proof.")

save_report(f"Report: {technique}")
