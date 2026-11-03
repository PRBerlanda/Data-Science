# 01 · Cluster Analysis (Hierarchical + K-means)

A ready-to-use tool that groups observations that are similar to each other, and different from the other groups, based on **quantitative variables**. It runs both agglomerative hierarchical clustering and k-means, picks the number of clusters automatically, tests which variables really separate the groups, and writes a report you can read without opening the code.

> Part of the [Data-Science](https://github.com/PRBerlanda/Data-Science/tree/main) repository. Each branch holds one model.

---

## What this model does

Cluster analysis is an **unsupervised, exploratory** technique: there is no target variable. The algorithm measures the distance between observations and builds groups (clusters) so that observations in the same group are close and groups are far apart. The result is a label for each observation (cluster 1, 2, 3, …) plus a profile that describes what makes each group different.

## When to use it

- Segmenting customers, stores, regions, products or employees when no label exists yet.
- Building personas or tiers from behavior (spending, frequency, channel usage…).
- As a first step before other models, to discover structure in the data.

**Do not use it when**
- The variables are categorical. Use [03 · Correspondence Analysis](https://github.com/PRBerlanda/Data-Science/tree/03-Correspondence-Analysis) instead.
- You need to predict a known outcome. Use a supervised model (04–12).
- You need to score new observations with a fixed rule. Clustering is descriptive; with new data, rerun the analysis.

## How it works

1. **Load the data** (CSV, Excel, Parquet, Pickle or JSON) and select the variables, either listed by you or all numeric columns.
2. **Handle missing values**: drop the rows or impute the mean.
3. **Standardize (Z-score)** when the variables are on very different scales, so that no variable dominates the distance just because of its unit.
4. **Choose the number of clusters (k)**: k-means is run for a range of k, recording the **elbow** (within-cluster sum of squares) and the **silhouette**. The k with the highest mean silhouette is used, unless you fix it.
5. **Hierarchical clustering**: builds the dendrogram with the chosen distance and linkage, then cuts it into k groups.
6. **K-means**: assigns each observation to the nearest centroid. When both methods run, a cross-table compares their assignments.
7. **ANOVA per variable**: tests whether each variable's mean differs across clusters, showing which variables actually drive the segmentation.
8. **Profile**: means per cluster on the original scale, a heatmap in standard deviations and a 2D projection (PCA) of the clusters.

## Inputs

**Data:** one row per observation, with at least two numeric columns. An ID column is optional and is kept in the output but never used in the calculation.

**Configuration** (edit only the `CONFIGURATION` cell at the top of the script):

| Parameter | What it controls |
|---|---|
| `FILE`, `SHEET`, `SEPARATOR` | Input file, Excel sheet and CSV separator (auto-detected when `None`). |
| `ID_COLUMN` | Identifier column, excluded from the calculation. |
| `VARIABLES` | List of variables, or `"auto"` for all numeric columns. |
| `IGNORE_COLUMNS` | Numeric columns that must not be used (e.g. account numbers). |
| `STANDARDIZE` | `True`, `False` or `"auto"` (standardizes when scales differ). |
| `MISSING_VALUES` | `"drop"` rows or impute the `"mean"`. |
| `METHOD` | `"kmeans"`, `"hierarchical"` or `"both"`. |
| `N_CLUSTERS` | A fixed integer, or `"auto"` (highest silhouette). |
| `K_MIN`, `K_MAX` | Range of k tested in automatic mode. |
| `DISTANCE`, `LINKAGE` | Distance metric and linkage method for the hierarchical step. |
| `SEED` | Reproducibility of k-means. |
| `OUTPUT_FOLDER`, `SHOW_PLOTS` | Where results go and whether charts pop up. |

## Outputs

Everything is saved in `results/`:

| File | Content |
|---|---|
| `report.md` | Full report: data summary, choice of k, cluster sizes, ANOVA, profile and interpretation tips. |
| `data_with_clusters.csv` | Original data plus the cluster label(s) for each observation. |
| `cluster_profile.csv` | Number of observations and mean of each variable per cluster. |
| `anova_variables.csv` | F statistic and p-value per variable. |
| `number_of_clusters_evaluation.csv` | WCSS and silhouette for each k tested. |
| `01_elbow_silhouette.png` … `04_clusters_2d.png` | Elbow/silhouette, dendrogram, profile heatmap and 2D projection. |

## How to read the results

- **Silhouette** (−1 to 1) measures how well each observation fits its cluster compared with the nearest other cluster. As a rule of thumb: below 0.25 the structure is weak, 0.25–0.5 is fair, above 0.5 is good.
- **Elbow chart**: the point where adding clusters stops reducing the WCSS sharply is a natural candidate for k. Use it together with the silhouette and with business sense.
- **ANOVA**: a variable with p-value < 0.05 differs significantly between clusters. The larger the F, the more it contributes to the separation. Variables that do not differentiate can be removed and the analysis rerun.
- **Profile heatmap** (values in standard deviations): positive cells mean the cluster is above the overall mean on that variable; negative cells mean below. Use this to **name** each cluster (e.g. "high limit, digital users").
- **Hierarchical × k-means table**: strong agreement between the two methods suggests stable groups.
- Clusters are a description, not a truth: validate them with people who know the business before acting on them.

## How to run

```bash
pip install -r requirements.txt
```

1. Put your data file in the same folder as `cluster_analysis.py` (or set an absolute path in `FILE`).
2. Edit the `CONFIGURATION` cell.
3. Run the script, either in Spyder (F5, or cell by cell with Ctrl+Enter) or from the terminal with `python cluster_analysis.py`.
4. Open `results/report.md`.

## Example dataset

`cartao_credito.csv` contains credit card customers: an ID (`Sl_No`), a customer key, the average credit limit, the number of cards, and the number of bank visits, online visits and calls made. The default configuration segments these customers by their limit and contact-channel behavior.

## Credits

Built from the course scripts by Prof. Dr. Wilson Tarantin Junior (MBA in Data Science and Analytics, USP/Esalq). Theory: Fávero & Belfiore, *Manual de Análise de Dados*, Ch. 9.
