# 02 · Principal Component Factor Analysis (PCA)

A ready-to-use tool that summarizes many correlated quantitative variables into a few **factors**. It checks whether the data is suitable for factor analysis, decides how many factors to keep, shows how each variable relates to each factor and, optionally, builds a **ranking** from the factor scores.

> Part of the [Data-Science](https://github.com/PRBerlanda/Data-Science/tree/main) repository. Each branch holds one model.

---

## What this model does

When several variables move together (they are correlated), part of the information they carry is redundant. Principal component factor analysis finds new variables, the **factors**, that are linear combinations of the originals, are uncorrelated with each other and capture as much of the shared variance as possible. A handful of factors can then replace many variables.

## When to use it

- Building an **index or ranking** from several indicators (e.g. a performance score from grades in several subjects, a development index from socioeconomic indicators).
- **Reducing dimensionality** before another model (clustering, regression).
- **Removing multicollinearity**: the factors are orthogonal and can enter a regression together.

**Do not use it when**
- The variables are not correlated with each other. The Bartlett test and KMO will flag this.
- The variables are categorical. Use [03 · Correspondence Analysis](https://github.com/PRBerlanda/Data-Science/tree/03-Correspondence-Analysis).
- You have very few observations per variable (fewer than about 5).

## How it works

1. **Load the data** and select the numeric variables (listed or automatic).
2. **Correlation matrix**: Pearson correlations, shown as a heatmap.
3. **Adequacy tests**:
   - *Bartlett's sphericity test* checks whether the correlations are, as a whole, different from zero.
   - *KMO (Kaiser-Meyer-Olkin)* measures the share of common variance, overall and per variable.
4. **Eigenvalues**: each eigenvalue is the variance captured by one factor. The **Kaiser criterion** keeps factors with eigenvalue > 1, unless you fix the number of factors.
5. **Loadings and communalities** of the retained factors, optionally with varimax rotation. Factor signs can be oriented so that "higher" means "more of the characteristic".
6. **Factor scores** for each observation and, if enabled, a **ranking score**: the sum of the scores weighted by each factor's share of explained variance.

## Inputs

**Data:** one row per observation, at least three numeric variables. An ID column is optional.

**Configuration** (the `CONFIGURATION` cell):

| Parameter | What it controls |
|---|---|
| `FILE`, `SHEET`, `SEPARATOR` | Input file, Excel sheet and CSV separator. |
| `ID_COLUMN` | Row identifier (kept in the output, not used in the analysis). |
| `VARIABLES`, `IGNORE_COLUMNS` | Variables to use (`"auto"` = all numeric) and columns to exclude. |
| `MISSING_VALUES` | `"drop"` rows or impute the `"mean"`. |
| `N_FACTORS` | Fixed number, or `"auto"` (Kaiser criterion). |
| `ROTATION` | `None` or `"varimax"` (easier to interpret when several factors are kept). |
| `LOADING_THRESHOLD` | Minimum absolute loading to list a variable under a factor in the report. |
| `CREATE_RANKING`, `RANKING_FACTORS` | Whether to build the ranking and which factors enter it. |
| `ORIENT_FACTORS` | Flips factor signs so that most loadings are positive. |

## Outputs

Saved in `results/`:

| File | Content |
|---|---|
| `report.md` | Adequacy tests, eigenvalues, loadings, communalities, top of the ranking and interpretation tips. |
| `data_with_scores.csv` | Original data plus one score column per factor and, if enabled, the ranking score and position. |
| `loadings_communalities.csv` | Loading of each variable on each factor and its communality. |
| `factor_score_weights.csv` | Weights to compute the factor scores of new observations. |
| `eigenvalues.csv` | Eigenvalue, % of variance and cumulative % per factor. |
| `correlations.csv` | Correlation matrix. |
| `01_correlations.png` … `04_loading_plot.png` | Correlation heatmap, scree plot, loading heatmap and loading plot. |

## How to read the results

- **Bartlett p-value < 0.05**: the correlations are strong enough for factor analysis. If it is not significant, the factors will not summarize the variables well.
- **KMO**: ≥ 0.9 marvelous, 0.8 meritorious, 0.7 middling, 0.6 mediocre, 0.5 miserable, below 0.5 unacceptable. A low KMO for a single variable suggests removing it.
- **Eigenvalue / explained variance**: how much of the total information each factor keeps. The cumulative % shows how much the retained factors summarize.
- **Loading**: the correlation between a variable and a factor (−1 to 1). Look at the variables with the largest absolute loadings to **name** the factor (e.g. "quantitative skills").
- **Communality**: the share of a variable's variance explained by the retained factors. Below 0.5 means the variable is poorly represented.
- **Factor score**: the position of each observation on a factor (mean 0, standard deviation 1). Positive means above average on that dimension.
- **Ranking score**: a single number combining the factors, weighted by their importance. Check the direction of each factor before reading "higher" as "better".
- To score **new observations**, standardize their variables with this dataset's mean and standard deviation and multiply by the weights in `factor_score_weights.csv`.

## How to run

```bash
pip install -r requirements.txt
```

1. Put your data file next to `pca_factor_analysis.py` (or use an absolute path in `FILE`).
2. Edit the `CONFIGURATION` cell.
3. Run the script in Spyder (F5) or with `python pca_factor_analysis.py`.
4. Open `results/report.md`.

## Example dataset

`notas_fatorial.xlsx` contains students (`estudante`) and their grades in four subjects: finance (`finanças`), costs (`custos`), marketing and actuarial science (`atuária`). The default configuration extracts the factors behind the grades and ranks the students.

## Credits

Built from the course scripts by Prof. Dr. Wilson Tarantin Junior (MBA in Data Science and Analytics, USP/Esalq). Theory: Fávero & Belfiore, *Manual de Análise de Dados*, Ch. 10.
