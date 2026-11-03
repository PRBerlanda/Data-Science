# 05 · Binary Logistic Regression

A ready-to-use tool that estimates the **probability of an event** (a target with two categories) and classifies cases. It handles the event coding, dummies, stepwise selection, odds ratios with confidence intervals, cutoff choice, confusion matrix, ROC/AUC, cross-validation and scoring of new data.

> Part of the [Data-Science](https://github.com/PRBerlanda/Data-Science/tree/main) repository. Each branch holds one model.

---

## What this model does

Logistic regression models the log-odds of the event as a linear function of the explanatory variables:

`ln( p / (1 − p) ) = β0 + β1·X1 + β2·X2 + …`

and turns it back into a probability between 0 and 1. It is estimated by maximum likelihood. Exponentiated coefficients are **odds ratios**, which say how the odds of the event change for each variable.

## When to use it

- The target has **two outcomes**: churn yes/no, default, loyalty, fraud, purchase, failure, approval…
- You want both a **probability** for each case and an **interpretable** effect for each factor.

**Use another model when** Y has 3+ unordered classes ([06](https://github.com/PRBerlanda/Data-Science/tree/06-Multinomial-Logistic-Regression)) or when you only care about predictive accuracy and can give up interpretability ([10](https://github.com/PRBerlanda/Data-Science/tree/10-Trees-and-Ensembles), [11](https://github.com/PRBerlanda/Data-Science/tree/11-Neural-Networks-Tabular)).

## How it works

1. **Encode Y**: the category set in `EVENT` becomes 1 and the other becomes 0 (automatic: 1 if Y is already 0/1, otherwise the least frequent category).
2. **Build X** with dummies (n − 1) for categorical variables. Ordinal ratings can be forced to be categorical.
3. **Fit the logit by maximum likelihood**, then apply **backward stepwise** until all variables are significant.
4. **Overall significance**: likelihood-ratio chi-square, McFadden pseudo-R², AIC and BIC.
5. **Coefficients, Wald z, odds ratios** and 95% CIs.
6. **Cutoff**: sensitivity and specificity are computed for every cutoff from 0.01 to 0.99. The default is the **Youden maximum** (sensitivity + specificity − 1), or you can fix a value.
7. **Confusion matrix, ROC curve, AUC and Gini.**
8. **Stratified cross-validation** of the AUC.
9. **Scoring of new data** (probability and class) if a file is provided.

## Inputs

**Data:** one row per case, a two-category target and the explanatory variables.

**Configuration** (the `CONFIGURATION` cell):

| Parameter | What it controls |
|---|---|
| `FILE`, `SHEET`, `SEPARATOR` | Input file, Excel sheet and CSV separator. |
| `TARGET`, `EVENT` | Target column and the category that counts as the event (or `"auto"`). |
| `X_VARIABLES`, `ID_COLUMN`, `IGNORE_COLUMNS` | Explanatory variables and columns to exclude. |
| `CATEGORICAL_VARIABLES` | `"auto"` (text/boolean) or a list, e.g. 1–5 ratings to be treated as categories. |
| `STEPWISE`, `P_VALUE_THRESHOLD` | Stepwise selection and its threshold. |
| `CUTOFF` | Fixed probability (e.g. 0.5) or `"auto"` (Youden). |
| `CROSS_VALIDATION`, `SEED` | Folds for out-of-sample AUC and random seed. |
| `PREDICT_FILE` | Optional file with new cases to score. |

## Outputs

Saved in `results/`:

| File | Content |
|---|---|
| `report.md` | Event coding, stepwise steps, model fit, odds ratios, classification metrics, ROC and interpretation. |
| `coefficients_odds_ratios.csv` | Coefficient, Wald z, p-value, CI and odds ratio with CI. |
| `statsmodels_summary.txt` | Full statsmodels summary. |
| `metrics_by_cutoff.csv` | Sensitivity, specificity, accuracy and Youden index for every cutoff. |
| `data_with_probabilities.csv` | Original data plus event flag, predicted probability and classification. |
| `new_data_predictions.csv` | Probability and class for `PREDICT_FILE`. |
| `01_sensitivity_specificity.png`, `02_roc_curve.png` | Cutoff chart and ROC curve. |

## How to read the results

- **Likelihood-ratio chi-square p < 0.05**: the model is better than having no explanatory variables.
- **McFadden pseudo-R²**: a relative fit measure. It is not comparable to the R² of linear regression, and values between 0.2 and 0.4 already indicate a good fit.
- **Odds ratio (OR)**: OR > 1 increases the odds of the event, OR < 1 decreases them. For example, OR = 1.8 means 80% higher odds for +1 unit of X (or for that category vs. the reference), holding the rest constant. If the CI includes 1, the effect is not significant.
- **Sensitivity**: share of real events the model catches. **Specificity**: share of non-events correctly identified. **Accuracy**: overall hit rate.
- **Cutoff** is a business decision. Lower it to catch more events when missing one is costly (fraud, disease). Raise it when false alarms are costly.
- **AUC** (independent of the cutoff): 0.5 = random, 0.7–0.8 acceptable, 0.8–0.9 good, > 0.9 excellent. **Gini** = 2·AUC − 1.
- **Cross-validated AUC much lower than the fit AUC** means overfitting.

## How to run

```bash
pip install -r requirements.txt
```

1. Put your data file next to `binary_logistic_regression.py`.
2. Edit the `CONFIGURATION` cell (at least `FILE`, `TARGET` and `EVENT`).
3. Run the script in Spyder (F5) or with `python binary_logistic_regression.py`.
4. Open `results/report.md`.

## Example dataset

`dados_fidelidade.csv` is a retail customer survey: an ID, the loyalty outcome (`fidelidade`: `sim` = yes, `nao` = no), sex, age and 1–5 ratings for service (`atendimento`), product range (`sortimento`), accessibility (`acessibilidade`) and price (`preço`). The ratings are treated as categories in the default configuration.

## Credits

Built from the "2 - Binary and Multinomial Logistic Models" course script by Prof. Dr. Luiz Paulo Fávero (MBA in Data Science and Analytics, USP/Esalq). Theory: Fávero & Belfiore, *Manual de Análise de Dados*, Ch. 13.
