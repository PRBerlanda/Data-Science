# 03 · Correspondence Analysis (ANACOR and MCA)

A ready-to-use tool that checks whether **categorical variables** are associated and shows **which categories go together** on a perceptual map. With two variables it runs simple correspondence analysis (ANACOR); with three or more it runs multiple correspondence analysis (MCA). The choice is automatic.

> Part of the [Data-Science](https://github.com/PRBerlanda/Data-Science/tree/main) repository. Each branch holds one model.

---

## What this model does

Correspondence analysis turns contingency tables (counts of how often categories occur together) into a map. Categories that tend to appear together are drawn close to each other, and categories that rarely co-occur are drawn far apart. Before drawing the map, the tool tests whether the association is real (chi-square) and pinpoints where it is (adjusted residuals).

## When to use it

- Surveys and questionnaires, including Likert scales.
- Customer profiles described by categories (age range, plan, region, channel…).
- Exploring which attributes are linked to an outcome category (e.g. which symptoms go with a diagnosis).

**Do not use it when**
- Your variables are mostly quantitative. Use [02 · PCA](https://github.com/PRBerlanda/Data-Science/tree/02-PCA-Factor-Analysis) or [01 · Cluster](https://github.com/PRBerlanda/Data-Science/tree/01-Cluster-Analysis). Numeric variables are accepted here, but they are binned into ranges.
- You need an equation or a prediction. This technique is exploratory.

## How it works

1. **Load the data** and select the variables (all columns except the ID, or a list).
2. **Prepare the categories**: numeric variables with many distinct values are split into equal-size bins (low / medium / high by default). Columns that look like identifiers are rejected.
3. **Chi-square test for every pair of variables** (H0: independence), reporting the % of expected frequencies below 5, the test's validity condition.
4. **Variable filter (MCA)**: variables with no significant association with any other variable can be dropped, following the rule used in class.
5. **Adjusted standardized residuals** for the relevant pairs (both variables in ANACOR; pairs with the *focus variable* in MCA). These show exactly which category pairs drive the association.
6. **Fit ANACOR or MCA**, compute eigenvalues and inertia, and get coordinates for every category and every observation.
7. **Perceptual map**, plus an optional map of the observations colored by the focus variable.

## Inputs

**Data:** one row per observation, with categorical columns (text or codes). Numeric columns are binned automatically.

**Configuration** (the `CONFIGURATION` cell):

| Parameter | What it controls |
|---|---|
| `FILE`, `SHEET`, `SEPARATOR` | Input file, Excel sheet and CSV separator. |
| `ID_COLUMN`, `VARIABLES`, `IGNORE_COLUMNS` | Identifier, variables to use (`"auto"` = all) and columns to exclude. |
| `FOCUS_VARIABLE` | Optional variable of interest: its residuals are detailed and the observation map is colored by it. |
| `MAX_CATEGORIES` | A numeric column with up to this many distinct values is treated as categorical. |
| `N_BINS` | Number of bins for numeric columns with more distinct values. |
| `DROP_UNASSOCIATED` | MCA only: drop variables with no significant association. |
| `N_DIMENSIONS` | Number of dimensions computed (the map uses the first two). |
| `SIGNIFICANCE_LEVEL` | Significance level for the tests. |
| `SHOW_OBSERVATIONS` | Draw observation points on the map. |

## Outputs

Saved in `results/`:

| File | Content |
|---|---|
| `report.md` | Categories and frequencies, chi-square tests, significant residuals, inertia and interpretation guide. |
| `chi2_tests.csv` | Chi-square, degrees of freedom, p-value and % of low expected frequencies for each pair. |
| `residuals_<var1>_x_<var2>.csv` / `.png` | Adjusted residuals table and heatmap for each analyzed pair. |
| `eigenvalues_inertia.csv` | Eigenvalue and % of inertia per dimension. |
| `category_coordinates.csv` | Map coordinates of every category. |
| `observations_with_coordinates.csv` | Data (with binned variables) plus each observation's coordinates. |
| `perceptual_map.png`, `observations_map_focus.png` | The perceptual map and the observation map. |

## How to read the results

1. **Chi-square p-value < 0.05**: the two variables are associated. If more than 20% of expected frequencies are below 5, merge rare categories before trusting the test.
2. **Adjusted residual > 1.96**: those two categories occur together **more** than expected by chance (5% significance). **Below −1.96**: they occur together **less** than expected.
3. **Perceptual map**: categories that are close together are associated. Read the map together with the residuals: the map shows proximity, the residuals confirm significance.
4. **% of inertia** on each axis shows how much of the association the 2D map represents. In MCA this percentage is usually low, so read the map as a summary rather than as proof.
5. **Focus map**: if observations of one category of the focus variable cluster in one region, the categories in that region describe that group's profile.

## How to run

```bash
pip install -r requirements.txt
```

1. Put your data file next to `correspondence_analysis.py`.
2. Edit the `CONFIGURATION` cell.
3. Run the script in Spyder (F5) or with `python correspondence_analysis.py`.
4. Open `results/report.md`.

## Example dataset

`dados_cor_acm.xlsx` is a heart disease dataset with patient attributes: age (`idade`), sex (`sexo`), chest pain type (`dor.peito`), resting blood pressure (`ps.descanco`), cholesterol (`colesterol`), fasting blood sugar (`acucar.sangue`), resting ECG (`ecg.descanco`), maximum heart rate (`bc.max`), exercise-induced angina (`angina.exerc`) and the diagnosis (`doenca.card`), which is the default focus variable. The quantitative attributes are binned automatically.

## Credits

Built from the course scripts by Prof. Dr. Wilson Tarantin Junior (MBA in Data Science and Analytics, USP/Esalq). Theory: Fávero & Belfiore, *Manual de Análise de Dados*, Ch. 11.
