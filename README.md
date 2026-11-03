# 07 · Models for Count Data (Poisson, Negative Binomial, ZIP, ZINB)

A ready-to-use tool for targets that are **counts** (0, 1, 2, …). It fits a Poisson model, tests for overdispersion, moves to a Negative Binomial when needed, tests for excess zeros with zero-inflated versions, compares all candidates, and reports incidence rate ratios and predictions.

> Part of the [Data-Science](https://github.com/PRBerlanda/Data-Science/tree/main) repository. Each branch holds one model.

---

## What this model does

Count data cannot be negative, is made of integers and is often skewed with many zeros, so linear regression is a poor fit. Count models describe the **expected count** as

`E[Y] = exp(β0 + β1·X1 + β2·X2 + …)`

and assume a distribution suited to counts:

| Model | When it fits |
|---|---|
| **Poisson** | Mean ≈ variance (equidispersion). |
| **Negative Binomial (NB2)** | Variance > mean (overdispersion), the most common case in practice. |
| **ZIP / ZINB** | More zeros than the base model expects. Two processes: one decides whether the observation is a "certain zero", the other generates the count. |

## When to use it

- Number of complaints, claims, defects, visits, violations, accidents, purchases per month, absences…
- Rates per unit of exposure (e.g. events per month observed), using an exposure column.

**Use another model when** Y is continuous ([04](https://github.com/PRBerlanda/Data-Science/tree/04-Linear-Regression)) or binary ([05](https://github.com/PRBerlanda/Data-Science/tree/05-Binary-Logistic-Regression)).

## How it works

1. **Validate Y**: it must contain integers ≥ 0. The report shows its mean, variance and share of zeros.
2. **Poisson** model (GLM), optionally with stepwise selection.
3. **Cameron & Trivedi overdispersion test**. A significant result means the Poisson standard errors are too small, so a Negative Binomial is preferred.
4. **Negative Binomial (NB2)** with its dispersion parameter α, plus a likelihood-ratio test of α = 0.
5. **Zero-inflated models (ZIP and ZINB)** with their own set of inflation variables, compared with the base models by the **Vuong test**.
6. **Automatic choice** (`MODEL = "auto"`): Poisson or NegBin depending on overdispersion, then the zero-inflated version if the Vuong test favors it. You can also force a model.
7. **Comparison table** (LL, AIC, BIC), coefficients with **IRR** and 95% CI, **observed × predicted distribution** chart, and predictions.

## Inputs

**Data:** one row per observation, a non-negative integer target and the explanatory variables.

**Configuration** (the `CONFIGURATION` cell):

| Parameter | What it controls |
|---|---|
| `FILE`, `SHEET`, `SEPARATOR` | Input file, Excel sheet and CSV separator. |
| `TARGET` | The count variable. |
| `X_VARIABLES`, `ID_COLUMN`, `IGNORE_COLUMNS` | Explanatory variables and columns to exclude. |
| `CATEGORICAL_VARIABLES` | `"auto"` or a list of columns to treat as categories. |
| `EXPOSURE_COLUMN` | Optional exposure (time, area, population) to model rates. |
| `MODEL` | `"auto"`, `"poisson"`, `"negbin"`, `"zip"` or `"zinb"`. |
| `INFLATION_VARIABLES` | Explanatory variables of the "certain zero" part of ZI models (`"auto"` = same as X). |
| `STEPWISE`, `P_VALUE_THRESHOLD` | Stepwise selection (on the Poisson model) and its threshold. |
| `PREDICT_FILE` | Optional file with new observations. |

## Outputs

Saved in `results/`:

| File | Content |
|---|---|
| `report.md` | Descriptive statistics of Y, overdispersion and Vuong tests, model comparison, coefficients/IRR and interpretation. |
| `model_comparison.csv` | LL, AIC, BIC and number of parameters for every fitted model. |
| `coefficients_irr.csv` | Coefficient, p-value and IRR with CI for the chosen model. |
| `statsmodels_summary.txt` | Full summary of the chosen model. |
| `data_with_predictions.csv` | Original data plus the expected count from every model. |
| `new_data_predictions.csv` | Expected count for `PREDICT_FILE` from the chosen model. |
| `01_observed_x_predicted.png` | Observed share of each count vs. the share predicted by each model. |

## How to read the results

- **Variance much larger than the mean** is an early sign of overdispersion. The Cameron & Trivedi test confirms it formally.
- **Ignoring overdispersion** underestimates standard errors and makes p-values look artificially small. That is why the test comes before the choice.
- **Vuong test** with a positive statistic and p < 0.05 means the zero-inflated model fits better.
- **AIC / BIC**: lower is better when comparing models on the same data.
- **IRR (incidence rate ratio) = exp(coefficient)**: the expected count is multiplied by the IRR for +1 unit of X. For example, IRR = 1.20 means a 20% higher expected count; IRR = 0.80 means 20% lower.
- **`inflate_` rows** (ZI models): logit coefficients of the probability of being a structural zero. A positive coefficient means a higher chance of a certain zero.
- **Observed × predicted chart**: the best model follows the observed bars, especially at zero and in the tail.

## How to run

```bash
pip install -r requirements.txt
```

1. Put your data file next to `count_data_models.py`.
2. Edit the `CONFIGURATION` cell (at least `FILE`, `TARGET` and `X_VARIABLES`).
3. Run the script in Spyder (F5) or with `python count_data_models.py`.
4. Open `results/report.md`.

## Example dataset

`corruption.csv` is the classic diplomatic parking violations dataset: for each country (`country`, `code`) and period, the number of unpaid parking violations (`violations`, the target), the number of diplomats (`staff`), whether the observation is after the law change (`post`: yes/no) and the country's corruption index (`corruption`).

## Credits

Built from the "03 - Models for Count Data" course script by Prof. Dr. Luiz Paulo Fávero (MBA in Data Science and Analytics, USP/Esalq). Theory: Fávero & Belfiore, *Manual de Análise de Dados*, Ch. 14.
