# -*- coding: utf-8 -*-
"""
CLUSTER ANALYSIS: WORKING TOOL
MBA in Data Science and Analytics (USP/Esalq). Working version generated with
Claude from the course scripts by Prof. Dr. Wilson Tarantin Junior (Credit Card,
Countries, Regional, Entrance Exam). Theory: Fávero & Belfiore, Ch. 9.

GOAL
- Group observations that are similar to each other (and different from the
  other groups) using QUANTITATIVE variables. Unsupervised, exploratory technique.

WHEN TO USE
- Segment customers, stores, regions, products, etc. without a prior label.
- Variables must be metric. For categorical variables, use ANACOR/MCA.
- It is not a predictive model: with new observations, rerun the analysis.

HOW TO USE
1. Edit only the CONFIGURATION cell (file, columns, options).
2. Run the whole script (F5) or cell by cell (Ctrl+Enter) in Spyder.
3. Results are saved in the folder set in OUTPUT_FOLDER (tables, charts and
   report.md with the interpretation).

Installation (once, in the Anaconda Prompt):
    pip install pandas numpy scipy scikit-learn matplotlib seaborn openpyxl
"""

#%% CONFIGURATION (the only place to edit)

# --- Data ----------------------------------------------------------------
# File path (.csv, .xlsx, .xls, .parquet, .pkl, .json).
# Relative path = relative to this script's folder.
FILE = "cartao_credito.csv"
SHEET = None            # Excel sheet name/index (None = first sheet)
SEPARATOR = None        # CSV separator (None = detect automatically)

ID_COLUMN = "Sl_No"     # identifier column (not used in the calculation). None if absent
VARIABLES = ["Avg_Credit_Limit", "Total_Credit_Cards", "Total_visits_bank",
             "Total_visits_online", "Total_calls_made"]
# VARIABLES = "auto"    # "auto" = all numeric columns, except ID_COLUMN
IGNORE_COLUMNS = ["Customer Key"]   # numeric columns that must NOT be used

# --- Preprocessing --------------------------------------------------------
STANDARDIZE = "auto"    # True, False or "auto" (standardize if scales differ)
MISSING_VALUES = "drop"       # "drop" rows with NA or "mean" (impute the mean)

# --- Method ---------------------------------------------------------------
METHOD = "both"         # "kmeans", "hierarchical" or "both"
N_CLUSTERS = "auto"     # integer (e.g. 3) or "auto" (highest silhouette)
K_MIN, K_MAX = 2, 10    # range tested in "auto" mode
DISTANCE = "euclidean"        # hierarchical: euclidean, cityblock, chebyshev, cosine
LINKAGE = "complete"          # hierarchical: single, complete, average, ward
SEED = 100              # k-means reproducibility

# --- Output ---------------------------------------------------------------
OUTPUT_FOLDER = "results"     # relative to this script's folder
SHOW_PLOTS = True             # False = only save the charts (batch run)
DENDROGRAM_LIMIT = 3000       # above this, the dendrogram uses a random sample
SIGNIFICANCE_LEVEL = 0.05


#%% Packages

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.stats import f_oneway, zscore
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=FutureWarning)


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


#%% 1. Load and inspect the data

data = read_data(FILE, SHEET, SEPARATOR)
log("## 1. Data")
log(f"- File: `{FILE}`")
log(f"- Shape: {data.shape[0]} rows x {data.shape[1]} columns")

if ID_COLUMN is not None and ID_COLUMN in data.columns:
    duplicated_ids = data[ID_COLUMN].duplicated().sum()
    if duplicated_ids:
        log(f"- Warning: {duplicated_ids} repeated IDs in `{ID_COLUMN}`.")

data.info()


#%% 2. Select variables and handle missing values

if VARIABLES == "auto":
    numeric = data.select_dtypes("number").columns.tolist()
    variables = [c for c in numeric
                 if c != ID_COLUMN and c not in IGNORE_COLUMNS]
else:
    variables = list(VARIABLES)

missing_cols = [v for v in variables if v not in data.columns]
if missing_cols:
    raise KeyError(f"Columns not found in the file: {missing_cols}")

non_numeric = [v for v in variables
               if not pd.api.types.is_numeric_dtype(data[v])]
if non_numeric:
    raise TypeError("Cluster analysis uses quantitative variables. Remove or convert: "
                    f"{non_numeric} (for categorical variables, use ANACOR/MCA).")
if len(variables) < 2:
    raise ValueError("Provide at least 2 variables for clustering.")

base = data.copy()
n_missing = int(base[variables].isna().any(axis=1).sum())
if n_missing:
    if MISSING_VALUES == "mean":
        base[variables] = base[variables].fillna(base[variables].mean())
    else:
        base = base.dropna(subset=variables).reset_index(drop=True)

log(f"- Variables used ({len(variables)}): {', '.join(variables)}")
log(f"- Rows with missing values: {n_missing} "
    f"(handling: {MISSING_VALUES})")
log(f"- Observations analyzed: {len(base)}")

constants = [v for v in variables if base[v].std(ddof=1) == 0]
if constants:
    raise ValueError(f"Variables without variation (constant): {constants}")

log("\n### Descriptive statistics")
log(md_table(base[variables].describe().T))


#%% 3. Standardization (Z-score)
# Distances are sensitive to scale: a variable in thousands of dollars
# "dominates" another measured in number of visits. The Z-score puts all of
# them on mean 0 and standard deviation 1.

std_devs = base[variables].std(ddof=1)
different_scales = (std_devs.max() / std_devs.min()) > 3
standardize = different_scales if STANDARDIZE == "auto" else bool(STANDARDIZE)

X = (base[variables].apply(zscore, ddof=1) if standardize
     else base[variables].copy())

log("\n## 2. Preprocessing")
log(f"- Ratio between the largest and smallest standard deviation: "
    f"{std_devs.max() / std_devs.min():.1f}")
log(f"- Z-score standardization applied: {'yes' if standardize else 'no'}")


#%% 4. Choosing the number of clusters (elbow + silhouette)
# Elbow: WCSS (within-cluster sum of squares) drops as k increases; the "knee"
# of the curve suggests k. Silhouette: measures how much closer each point is
# to its own cluster than to the neighboring one (-1 to 1; higher is better).

k_max = min(K_MAX, len(X) - 1)
k_range = range(max(2, K_MIN), k_max + 1)
wcss, silhouettes = [], []
for k in k_range:
    km = KMeans(n_clusters=k, init="random", n_init=10,
                random_state=SEED).fit(X)
    wcss.append(km.inertia_)
    silhouettes.append(silhouette_score(X, km.labels_))

k_evaluation = pd.DataFrame({"k": list(k_range), "WCSS": wcss,
                             "silhouette": silhouettes}).set_index("k")
k_suggested = int(k_evaluation["silhouette"].idxmax())
k_final = k_suggested if N_CLUSTERS == "auto" else int(N_CLUSTERS)

log("\n## 3. Number of clusters")
log(md_table(k_evaluation))
log(f"- Highest mean silhouette: k = {k_suggested} "
    f"({k_evaluation.loc[k_suggested, 'silhouette']:.3f})")
log(f"- k used: {k_final} "
    f"({'automatic' if N_CLUSTERS == 'auto' else 'set in CONFIG'})")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(k_evaluation.index, k_evaluation["WCSS"], marker="o")
axes[0].set(title="Elbow method", xlabel="Number of clusters (k)",
            ylabel="WCSS")
axes[1].plot(k_evaluation.index, k_evaluation["silhouette"], marker="o",
             color="purple")
axes[1].axvline(k_final, ls="--", color="grey")
axes[1].set(title="Mean silhouette", xlabel="Number of clusters (k)",
            ylabel="Silhouette")
save_figure(fig, "01_elbow_silhouette")


#%% 5. Agglomerative hierarchical clustering (dendrogram)

if METHOD in ("hierarchical", "both"):
    if LINKAGE == "ward" and DISTANCE != "euclidean":
        raise ValueError("'ward' linkage requires Euclidean distance.")

    sample = X
    if len(X) > DENDROGRAM_LIMIT:
        sample = X.sample(DENDROGRAM_LIMIT, random_state=SEED)
        log(f"- Dendrogram built from a sample of {DENDROGRAM_LIMIT}"
            " observations (large dataset).")

    links = sch.linkage(sample, method=LINKAGE, metric=DISTANCE)
    fig, ax = plt.subplots(figsize=(12, 5))
    sch.dendrogram(links, no_labels=len(sample) > 60,
                   color_threshold=links[-(k_final - 1), 2], ax=ax)
    ax.set(title=f"Dendrogram ({LINKAGE} linkage, {DISTANCE})",
           xlabel="Observations", ylabel="Distance")
    save_figure(fig, "02_dendrogram")

    hierarchical = AgglomerativeClustering(n_clusters=k_final,
                                           metric=DISTANCE,
                                           linkage=LINKAGE).fit(X)
    base["cluster_hierarchical"] = (hierarchical.labels_ + 1).astype(str)
    log("\n## 4. Hierarchical")
    log(f"- Linkage: {LINKAGE} | distance: {DISTANCE}")
    log("- Cluster sizes: "
        + str(base["cluster_hierarchical"].value_counts()
              .sort_index().to_dict()))


#%% 6. K-means (non-hierarchical)

if METHOD in ("kmeans", "both"):
    kmeans = KMeans(n_clusters=k_final, init="random", n_init=10,
                    random_state=SEED).fit(X)
    base["cluster_kmeans"] = (kmeans.labels_ + 1).astype(str)
    log("\n## 5. K-means")
    log(f"- Final mean silhouette: {silhouette_score(X, kmeans.labels_):.3f}")
    log("- Cluster sizes: "
        + str(base["cluster_kmeans"].value_counts()
              .sort_index().to_dict()))

    if METHOD == "both":
        agreement = pd.crosstab(base["cluster_hierarchical"],
                                base["cluster_kmeans"])
        log("\nHierarchical (rows) x k-means (columns) comparison:")
        log(md_table(agreement, 0))

cluster_column = ("cluster_kmeans" if METHOD in ("kmeans", "both")
                  else "cluster_hierarchical")


#%% 7. ANOVA: which variables differentiate the clusters?
# H0: the variable's means are equal in all clusters.
# p-value < 0.05 → the variable helps separate the groups. The larger the F,
# the larger the variable's contribution to forming the clusters.

rows = []
groups = base.groupby(cluster_column)
for v in variables:
    samples = [g[v].values for _, g in groups]
    f_stat, p = f_oneway(*samples)
    rows.append({"variable": v, "F": f_stat, "p_value": p,
                 "differentiates_clusters": p < SIGNIFICANCE_LEVEL})
anova = pd.DataFrame(rows).set_index("variable").sort_values("F",
                                                             ascending=False)
log("\n## 6. ANOVA by variable")
log(md_table(anova, 4))
not_differentiating = anova.index[~anova["differentiates_clusters"]].tolist()
if not_differentiating:
    log(f"- Do not differentiate the clusters (p >= {SIGNIFICANCE_LEVEL}): "
        f"{', '.join(not_differentiating)}. Consider removing them and rerunning.")


#%% 8. Cluster profile (means on the original scale)

profile = base.groupby(cluster_column)[variables].mean()
profile.insert(0, "n_observations", base[cluster_column].value_counts())
log("\n## 7. Cluster profile (original means)")
log(md_table(profile, 2))

# Z-score heatmap: compares each cluster with the overall mean (0)
profile_z = (base[variables].apply(zscore, ddof=1)
             .groupby(base[cluster_column]).mean())
fig, ax = plt.subplots(figsize=(1.2 * len(variables) + 3, 0.6 * k_final + 2))
sns.heatmap(profile_z, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set(title="Cluster profile (mean in standard deviations)",
       xlabel="Variable", ylabel="Cluster")
save_figure(fig, "03_cluster_profile")

# 2D view: projection on the first 2 principal components
components = PCA(n_components=2).fit(X)
coords = components.transform(X)
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=base[cluster_column],
                palette="tab10", s=25, ax=ax)
var_exp = components.explained_variance_ratio_ * 100
ax.set(title="Clusters projected in 2 dimensions (PCA)",
       xlabel=f"Component 1 ({var_exp[0]:.1f}% of variance)",
       ylabel=f"Component 2 ({var_exp[1]:.1f}% of variance)")
save_figure(fig, "04_clusters_2d")


#%% 9. Save results

base.to_csv(OUTPUT / "data_with_clusters.csv", index=False, encoding="utf-8-sig")
profile.to_csv(OUTPUT / "cluster_profile.csv", encoding="utf-8-sig")
anova.to_csv(OUTPUT / "anova_variables.csv", encoding="utf-8-sig")
k_evaluation.to_csv(OUTPUT / "number_of_clusters_evaluation.csv", encoding="utf-8-sig")

log("\n## 8. How to interpret")
log("- Use the profile (means and heatmap) to name each cluster: positive "
    "values on the heatmap = above the overall mean.")
log("- Silhouette < 0.25 indicates weak group structure; between 0.25 and "
    "0.5, fair; above 0.5, good.")
log("- Clustering is exploratory: validate the groups with people who know "
    "the business before using them in decisions.")

save_report("Report: Cluster Analysis")
