# 15 · Linear and Integer Programming (an Excel Solver replacement)

A ready-to-use tool that solves **optimization problems** (maximize profit, minimize cost) described in a **spreadsheet**, with continuous, integer or binary decision variables. It returns the optimal solution, the objective value, constraint slacks and, for continuous problems, shadow prices and reduced costs: the equivalent of the Solver sensitivity report.

> Part of the [Data-Science](https://github.com/PRBerlanda/Data-Science/tree/main) repository. Each branch holds one model.

---

## What this model does

Linear programming finds the values of the decision variables that optimize a linear objective while respecting linear constraints:

```
maximize (or minimize)   c1·x1 + c2·x2 + … + cn·xn
subject to               a11·x1 + … + a1n·xn  ≤ / ≥ / =  b1
                         …
                         lower bound ≤ xj ≤ upper bound
```

When some variables must be whole numbers (units, trucks, people) or yes/no decisions, the problem becomes **integer** or **binary** programming. The tool uses the HiGHS solver through SciPy (`milp` and `linprog`).

## When to use it

- Production mix, resource allocation, staffing and scheduling.
- **Transportation** problems: ship from origins to destinations at minimum cost.
- Cutting stock, blending, budget allocation, project selection (binary).
- Any model you would otherwise set up in the Excel Solver.

## How it works

1. **Read the model** from the spreadsheet in one of two formats (below). If the file does not exist, a filled-in example is created to use as a template.
2. **Build the matrices**: objective coefficients, constraint matrix, bounds and integrality.
3. **Solve** with `milp` (HiGHS), within a time limit for large integer problems.
4. **Report** the status, the optimal objective value, the variable values (or the origin × destination shipment matrix), and the use and slack of each constraint.
5. **Sensitivity** (continuous problems only): the LP is solved again with `linprog` to read the shadow prices of the constraints and the reduced costs of the variables.

## Inputs

### Format 1: `MODEL_TYPE = "general"`

A workbook with two sheets:

**`variables`**: one row per decision variable

| name | objective_coef | lower_bound | upper_bound | type |
|---|---|---|---|---|
| H1 | 1.875 | 0 | *(empty = no bound)* | continuous |

`type` = `continuous`, `integer` or `binary`.

**`constraints`**: one row per constraint, plus one column per variable with its coefficient

| name | sign | rhs | H1 | H2 |
|---|---|---|---|---|
| available_time | <= | 30 | 1 | 1 |

`sign` = `<=`, `>=` or `equal`. In Excel, typing `=` alone starts a formula, so write `equal` (`==` and `eq` are also accepted). An empty coefficient counts as 0.

### Format 2: `MODEL_TYPE = "transportation"`

A workbook with three sheets:
- `costs`: origins in rows, destinations in columns, unit cost in the cells;
- `supply`: origin | capacity;
- `demand`: destination | quantity.

The transportation model (minimize total cost, ship at most the supply, meet each demand) is built automatically.

**Configuration** (the `CONFIGURATION` cell):

| Parameter | What it controls |
|---|---|
| `FILE` | The model workbook. |
| `MODEL_TYPE` | `"general"` or `"transportation"`. |
| `SENSE` | `"max"` or `"min"` (transportation is always minimization). |
| `SUPPLY_EQUAL` | Transportation: each origin ships all of its supply (`True`) or up to its capacity (`False`). |
| `TIME_LIMIT_SECONDS` | Time limit for large integer problems. |

## Outputs

Saved in `results/`:

| File | Content |
|---|---|
| `report.md` | Model summary, status, optimal value, variable values, constraint table and interpretation guide. |
| `solution.csv` | Value of each variable, its contribution to the objective and its reduced cost (continuous problems). |
| `constraints.csv` | Right-hand side, value used, slack, binding flag and shadow price (continuous problems) for each constraint. |
| `shipment_matrix.csv` | Origin × destination quantities (transportation model). |
| `optimization_result.xlsx` | Solution and constraints in one workbook. |

## How to read the results

- **Status**: "Optimal" means the best solution was found. *Infeasible* means the constraints conflict. *Unbounded* means some capacity constraint is missing.
- **Optimal value**: the best achievable objective under the constraints.
- **Binding constraint** (slack 0): the resource is fully used, and it is what limits the result. Non-binding constraints have spare capacity (slack > 0).
- **Shadow price**: how much the objective improves with one extra unit of that constraint's right-hand side. It is valid for small changes, and its sign follows the direction of the constraint.
- **Reduced cost**: for a variable at zero, how much its objective coefficient would need to improve before it enters the solution.
- **Integer and binary problems** have no shadow prices, because the solution does not change continuously.

## How to run

```bash
pip install -r requirements.txt
```

1. Fill in a workbook in one of the formats above (start from `example_study_hours.xlsx` or the files in `examples/`).
2. Edit the `CONFIGURATION` cell.
3. Run the script in Spyder (F5) or with `python linear_programming_optimization.py`.
4. Open `results/report.md`.

## Example files

| File | Problem | Settings |
|---|---|---|
| `example_study_hours.xlsx` | Split study hours between two subjects to maximize the expected grade, given the time available and minimum grades. | `MODEL_TYPE = "general"`, `SENSE = "max"` |
| `examples/example_transportation.xlsx` | Ship from two distribution centers to three stores at minimum cost. | `MODEL_TYPE = "transportation"` |
| `examples/example_transportation_multiple7_integer.xlsx` | The same transportation problem written in the general format, with integer quantities and one route constrained to multiples of 7. | `MODEL_TYPE = "general"`, `SENSE = "min"` |

## Credits

Built from the Operations Research Solver spreadsheets (Classes 02 and 03) of the MBA in Data Science and Analytics (USP/Esalq).
