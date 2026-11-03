# 08 · Multilevel Modeling (HLM2 and HLM3)

A ready-to-use tool for a continuous target when observations are **nested in groups**: students in schools, customers in stores, employees in branches, or repeated measures within individuals. It measures how much of the variation lies between groups, tests random slopes, fits the final mixed model and ranks the groups by their random effects.

> Part of the [Data-Science](https://github.com/PRBerlanda/Data-Science/tree/main) repository. Each branch holds one model.

---

## What this model does

Ordinary regression assumes independent observations. When observations share a group, they tend to be more similar to each other, and ignoring that gives wrong standard errors and misses group-level effects. A hierarchical linear model (HLM) splits the variation into levels:

- **Fixed effects**: the average effect of each explanatory variable, as in a regression.
- **Random intercepts**: each group has its own baseline level of Y.
- **Random slopes** (optional): the effect of a variable can differ from group to group.

**HLM2** has two levels (observation → group). **HLM3** has three levels (measure over time → individual → group) and is enabled by setting a middle group.

## When to use it

- Education (students → classes → schools), retail (customers → stores), HR (employees → teams → branches), health (patients → hospitals).
- Longitudinal data: repeated measures of the same individuals over time.
- Whenever you want to rank groups fairly, after accounting for the characteristics of their members.

**Use another model when** the observations are independent ([04](https://github.com/PRBerlanda/Data-Science/tree/04-Linear-Regression)) or when the target is not continuous.

## How it works

1. **Load the data**, define the top group (and optionally the middle group), the fixed effects and any interactions (for example, cross-level interactions).
2. **Null model** (no explanatory variables): computes the **ICC** (intraclass correlation) and tests the multilevel null model against an OLS null model with a likelihood-ratio test.
3. **Random slope tests**: for each numeric variable that varies within groups, a random slope is added and tested by likelihood ratio. The significant ones are kept (up to a limit), unless you set them yourself.
4. **Final model**: fixed effects with 95% CI, variance components, and a likelihood-ratio test against an OLS with the same fixed effects.
5. **Random effects per group**: how far each group is above or below the average, shown in a chart.
6. **Observed × predicted** comparison of the multilevel model and OLS, and optional **predictions for new data**.

## Inputs

**Data:** one row per observation (or per measure, in HLM3), with the target, the group identifiers and the explanatory variables.

**Configuration** (the `CONFIGURATION` cell):

| Parameter | What it controls |
|---|---|
| `FILE`, `SHEET`, `SEPARATOR` | Input file, Excel sheet and CSV separator. |
| `TARGET` | The continuous Y. |
| `TOP_GROUP` | Highest-level group column (required). |
| `MIDDLE_GROUP` | Middle-level group column (turns the model into HLM3), or `None`. |
| `X_VARIABLES`, `IGNORE_COLUMNS` | Fixed effects (level-1 and level-2 variables) and columns to exclude. |
| `CATEGORICAL_VARIABLES` | `"auto"` (text) or a list. |
| `INTERACTIONS` | Pairs of variables to interact, e.g. `[("hours", "teacher_experience")]`. |
| `RANDOM_SLOPES`, `MAX_SLOPES` | `"auto"` (test and keep the significant ones), a list, or `[]`. |
| `ESTIMATION_METHOD` | `"ML"` (allows likelihood-ratio comparisons) or `"REML"`. |
| `PREDICT_FILE` | Optional file with new observations (X and group columns). |

## Outputs

Saved in `results/`:

| File | Content |
|---|---|
| `report.md` | Structure of the data, ICC, slope tests, fixed effects, variance components and interpretation. |
| `fixed_effects.csv` | Coefficient, standard error, p-value and 95% CI of the fixed effects. |
| `variance_components.csv` | Residual variance and the variance of each random effect. |
| `random_effects_by_group.csv` | Random intercept (and slopes) estimated for each group. |
| `statsmodels_summary.txt` | Full summary of the final mixed model. |
| `data_with_predictions.csv` | Original data plus multilevel and OLS predictions. |
| `new_data_predictions.csv` | Predictions for `PREDICT_FILE` (fixed part, and fixed + group effect for known groups). |
| `01_random_intercepts.png`, `02_observed_predicted.png` | Random intercepts by group and observed × predicted. |

## How to read the results

- **ICC** (0 to 1): the share of the total variance of Y that is due to differences between groups. Even a moderate ICC means the group structure matters.
- **LR test multilevel × OLS** with p < 0.05: the multilevel structure is needed. Plain OLS would give misleading standard errors.
- **Fixed effects**: average effect of X on Y across all groups, with standard errors that are correct for nested data.
- **Variance components**: how much Y varies between groups (intercept variance) and how much the effect of X varies between groups (slope variance). A large slope variance means "one size fits all" does not hold.
- **Random effect of a group**: how much that group is above (positive) or below (negative) the average, **after** accounting for the variables in the model. This is useful for fair rankings of schools, stores or branches.
- New groups not seen in the data receive only the fixed part of the prediction.

## How to run

```bash
pip install -r requirements.txt
```

1. Put your data file next to `multilevel_modeling.py`.
2. Edit the `CONFIGURATION` cell (at least `FILE`, `TARGET`, `TOP_GROUP` and `X_VARIABLES`).
3. Run the script in Spyder (F5) or with `python multilevel_modeling.py`.
4. Open `results/report.md`.

## Example dataset

`desempenho_aluno_escola.csv` contains students (`estudante`) nested in schools (`escola`), each student's performance score (`desempenho`, the target), weekly study hours (`horas`, a student-level variable) and the teacher's years of experience (`texp`, a school-level variable). The default model includes a cross-level interaction between study hours and teacher experience.

## Credits

Built from the "04 - Multilevel Modeling" course script by Prof. Dr. Luiz Paulo Fávero (MBA in Data Science and Analytics, USP/Esalq). Theory: Fávero & Belfiore, *Manual de Análise de Dados*, Ch. 16.
