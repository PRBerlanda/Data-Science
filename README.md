# 06 · Multinomial Logistic Regression

A ready-to-use tool that estimates the **probability of each category** of a target with **three or more unordered classes** and classifies cases. It chooses the reference category, runs a stepwise selection adapted to multiple equations, reports relative risk ratios, evaluates the classification and scores new data.

> Part of the [Data-Science](https://github.com/PRBerlanda/Data-Science/tree/main) repository. Each branch holds one model.

---

## What this model does

Multinomial logistic regression generalizes the binary logit. With K categories, one category is the **reference** and the model estimates K − 1 equations, each comparing one category with the reference:

`ln( P(category k) / P(reference) ) = β0k + β1k·X1 + β2k·X2 + …`

The probabilities of all categories add up to 1, and each case is assigned to the category with the highest probability.

## When to use it

- The target has **3+ categories without a natural order**: purchase channel, chosen plan, product type, transport mode, reason for contact…
- You want to know how each factor shifts the odds of each category relative to a baseline.

**Use another model when** the target has only two categories ([05](https://github.com/PRBerlanda/Data-Science/tree/05-Binary-Logistic-Regression)), when it is a count ([07](https://github.com/PRBerlanda/Data-Science/tree/07-Count-Data-Models)), or when predictive accuracy is the priority ([10](https://github.com/PRBerlanda/Data-Science/tree/10-Trees-and-Ensembles)).

## How it works

1. **Encode Y**: the reference category (set by you, or the most frequent one) gets code 0, and the others follow in alphabetical order.
2. **Build X** with dummies for categorical variables.
3. **Fit the MNLogit** by maximum likelihood.
4. **Stepwise selection**: a variable is removed only if it is **not significant in any** of the K − 1 equations (its smallest p-value is above the threshold).
5. **Model fit**: log-likelihood, likelihood-ratio chi-square, McFadden pseudo-R² and AIC.
6. **Coefficients and relative risk ratios (RRR)** with 95% CIs, equation by equation.
7. **Classification**: confusion matrix and accuracy compared with simply predicting the majority class.
8. **Stratified cross-validation** of the accuracy and, optionally, **scoring of new data**.

## Inputs

**Data:** one row per case, a target with 3+ categories and the explanatory variables.

**Configuration** (the `CONFIGURATION` cell):

| Parameter | What it controls |
|---|---|
| `FILE`, `SHEET`, `SEPARATOR` | Input file, Excel sheet and CSV separator. |
| `TARGET` | The multi-category Y. |
| `REFERENCE_CATEGORY` | Baseline category, or `"auto"` (most frequent). |
| `X_VARIABLES`, `ID_COLUMN`, `IGNORE_COLUMNS` | Explanatory variables and columns to exclude. |
| `CATEGORICAL_VARIABLES` | `"auto"` or a list of columns to treat as categories. |
| `STEPWISE`, `P_VALUE_THRESHOLD` | Stepwise selection and its threshold. |
| `CROSS_VALIDATION`, `SEED` | Folds for out-of-sample accuracy and random seed. |
| `MAX_CLASSES` | Safety limit against choosing a continuous column or an ID as target. |
| `PREDICT_FILE` | Optional file with new cases to score. |

## Outputs

Saved in `results/`:

| File | Content |
|---|---|
| `report.md` | Class counts, stepwise steps, model fit, coefficients/RRR per equation, confusion matrix and interpretation. |
| `coefficients_rrr.csv` | Coefficient, p-value and RRR with CI for every variable in every equation. |
| `statsmodels_summary.txt` | Full statsmodels summary. |
| `data_with_probabilities.csv` | Original data plus the probability of each class and the predicted class. |
| `new_data_predictions.csv` | Probabilities and predicted class for `PREDICT_FILE`. |
| `01_confusion_matrix.png` | Confusion matrix heatmap. |

## How to read the results

- Each equation reads as "**category k vs. reference**". All interpretations are relative to the reference category.
- **RRR = exp(coefficient)**: RRR > 1 means that +1 unit of X increases the odds of being in category k rather than in the reference. RRR < 1 means it decreases them. If the CI includes 1, the effect is not significant.
- **Likelihood-ratio chi-square p < 0.05**: the model explains the target better than having no explanatory variables.
- **Accuracy vs. majority class**: the model only adds value if its accuracy beats the share of the most frequent class.
- **Confusion matrix**: shows which categories the model confuses with each other.
- **Cross-validated accuracy** close to the fit accuracy indicates a stable model.
- The model assumes the independence of irrelevant alternatives: adding or removing a category should not change the relative odds between the others.

## How to run

```bash
pip install -r requirements.txt
```

1. Put your data file next to `multinomial_logistic_regression.py`.
2. Edit the `CONFIGURATION` cell (at least `FILE`, `TARGET` and `REFERENCE_CATEGORY`).
3. Run the script in Spyder (F5) or with `python multinomial_logistic_regression.py`.
4. Open `results/report.md`.

## Example dataset

`atrasado_multinomial.csv` lists students (`estudante`), whether they were late (`atrasado`: not late / late for the first class / late for the second class), the distance traveled to school (`dist`) and the number of traffic lights on the way (`sem`). "Not late" (`nao chegou atrasado`) is the reference category.

## Credits

Built from the "2 - Binary and Multinomial Logistic Models" course script by Prof. Dr. Luiz Paulo Fávero (MBA in Data Science and Analytics, USP/Esalq). Theory: Fávero & Belfiore, *Manual de Análise de Dados*, Ch. 13.
