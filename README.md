# 04 · Simple and Multiple Linear Regression (OLS)

A ready-to-use tool that **explains and predicts a continuous variable** from quantitative and/or categorical explanatory variables. Beyond fitting the model, it runs stepwise selection, checks every classical assumption, applies a Box-Cox transformation or robust standard errors when needed, measures out-of-sample error and predicts new data with prediction intervals.

> Part of the [Data-Science](https://github.com/PRBerlanda/Data-Science/tree/main) repository. Each branch holds one model.

---

## What this model does

Ordinary least squares (OLS) regression estimates a linear equation

`Y = β0 + β1·X1 + β2·X2 + … + error`

choosing the coefficients that minimize the sum of squared errors. Each coefficient tells how much Y changes, on average, when one explanatory variable increases by one unit and the others stay constant.

## When to use it

- Y is **continuous**: price, revenue, time, score, return, consumption…
- You want to **explain** the effect of each factor (with significance tests), not only predict.

**Use another model when** Y is binary ([05](https://github.com/PRBerlanda/Data-Science/tree/05-Binary-Logistic-Regression)), has 3+ unordered classes ([06](https://github.com/PRBerlanda/Data-Science/tree/06-Multinomial-Logistic-Regression)), is a count 0, 1, 2… ([07](https://github.com/PRBerlanda/Data-Science/tree/07-Count-Data-Models)), the observations are nested in groups ([08](https://github.com/PRBerlanda/Data-Science/tree/08-Multilevel-Modeling)), or predictive accuracy matters more than interpretation ([10](https://github.com/PRBerlanda/Data-Science/tree/10-Trees-and-Ensembles)).

## How it works

1. **Load the data** and build X: categorical variables become dummies (n − 1 categories; the omitted one is the reference).
2. **Full model**, then **backward stepwise**: the variable with the highest p-value above the threshold is removed, one at a time, until all remaining ones are significant.
3. **Assumption diagnostics**:
   - residual normality (Shapiro-Wilk for n < 30, Shapiro-Francia otherwise);
   - homoskedasticity (Breusch-Pagan);
   - multicollinearity (VIF and tolerance);
   - influential observations (Cook's distance > 4/n).
4. **Box-Cox** (automatic when residuals are not normal and Y > 0): finds the λ that brings Y closest to normal, refits, and keeps the transformed model only if it improves normality. Predictions are always returned on the original scale.
5. **Robust standard errors (HC3)** when Breusch-Pagan detects heteroskedasticity.
6. **K-fold cross-validation**: R², RMSE and MAE measured on data the model did not see.
7. **Diagnostic charts** and, if a file is provided, **predictions for new observations** with a 95% prediction interval.

## Inputs

**Data:** one row per observation with a numeric target and the explanatory variables. Text columns are treated as categorical automatically.

**Configuration** (the `CONFIGURATION` cell):

| Parameter | What it controls |
|---|---|
| `FILE`, `SHEET`, `SEPARATOR` | Input file, Excel sheet and CSV separator. |
| `TARGET` | The continuous Y. |
| `X_VARIABLES` | List of explanatory variables or `"auto"` (all except target, ID and ignored). |
| `ID_COLUMN`, `IGNORE_COLUMNS` | Columns kept out of the model. |
| `CATEGORICAL_VARIABLES` | `"auto"` (text/boolean) or a list, e.g. numeric codes that are really categories. |
| `STEPWISE`, `P_VALUE_THRESHOLD` | Turn stepwise selection on/off and set its threshold. |
| `BOX_COX` | `"auto"`, `True` or `False`. |
| `ROBUST_ERRORS` | `"auto"` (when heteroskedastic), `True` or `False`. |
| `CROSS_VALIDATION`, `SEED` | Number of folds (0 = off) and random seed. |
| `PREDICT_FILE` | Optional file with new observations (same X columns) to predict. |

## Outputs

Saved in `results/`:

| File | Content |
|---|---|
| `report.md` | Data summary, stepwise steps, fit statistics, assumption tests, coefficients, cross-validation and interpretation. |
| `coefficients.csv` | Coefficient, standard error, p-value and 95% CI for each term. |
| `robust_coefficients.csv` | Robust (HC3) standard errors and p-values, when used. |
| `statsmodels_summary.txt` | Full statsmodels summary of the final model. |
| `data_with_predictions.csv` | Original data plus fitted value, residual and Cook's distance. |
| `new_data_predictions.csv` | Predictions and 95% prediction interval for `PREDICT_FILE`. |
| `01_residual_diagnostics.png` | Observed × predicted, residuals × fitted, Q-Q plot and residual histogram. |

## How to read the results

- **R² / adjusted R²**: share of the variance of Y explained by the model. Adjusted R² penalizes extra variables.
- **F test p-value < 0.05**: the model as a whole explains Y better than the mean alone.
- **Coefficient**: average change in Y for +1 unit of X, holding the others constant. For a **dummy**, it is the average difference relative to the reference category. When Box-Cox is applied, coefficients are on the transformed scale: interpret the sign, and use the predictions to see the effect on Y.
- **p-value < 0.05**: the effect is statistically different from zero. Also ask whether the size of the effect matters for the business.
- **95% CI**: the range of plausible values for the coefficient.
- **Normality / Breusch-Pagan**: when violated, p-values and intervals may be unreliable. The script reacts with Box-Cox and robust errors.
- **VIF > 5** deserves attention, **> 10** is a serious multicollinearity problem.
- **Cross-validated R² much lower than the fit R²** means overfitting.
- **Prediction interval**: the range where a new individual observation is expected to fall. It is wider than the confidence interval of the mean.

## How to run

```bash
pip install -r requirements.txt
```

1. Put your data file next to `linear_regression.py`.
2. Edit the `CONFIGURATION` cell (at least `FILE` and `TARGET`).
3. Run the script in Spyder (F5) or with `python linear_regression.py`.
4. Open `results/report.md`.

## Example dataset

`empresas.csv` contains companies (`empresa`) with their stock return (`retorno`, the target) and four candidate explanatory variables: disclosure index, indebtedness (`endividamento`), total assets (`ativos`) and liquidity (`liquidez`).

## Credits

Built from the "01 - Simple and Multiple Regression" course script by Prof. Dr. Luiz Paulo Fávero (MBA in Data Science and Analytics, USP/Esalq). Theory: Fávero & Belfiore, *Manual de Análise de Dados*, Ch. 12.
