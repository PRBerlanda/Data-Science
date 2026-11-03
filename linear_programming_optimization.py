# -*- coding: utf-8 -*-
"""
LINEAR AND INTEGER PROGRAMMING (A REPLACEMENT FOR THE EXCEL SOLVER): WORKING TOOL
MBA in Data Science and Analytics (USP/Esalq). Working version generated with
Claude from the Operations Research Solver spreadsheets (Classes 02 and 03:
study hours, coil cutting, transportation, transshipment, new hospital wing,
book sales).

GOAL
- Solve optimization problems (maximize profit, minimize cost) with
  continuous, integer or binary variables, described in a SPREADSHEET,
  without programming the model.

TWO INPUT FORMATS
1. MODEL_TYPE = "general": spreadsheet with two sheets
   - "variables": name | objective_coef | lower_bound | upper_bound | type
                  (type = continuous, integer or binary)
   - "constraints": name | sign (<=, >= or equal) | rhs | <one column per
                    variable with the variable's coefficient in the constraint>
2. MODEL_TYPE = "transportation": spreadsheet with the "costs" sheet (origins
   in rows, destinations in columns), plus the "supply" (origin | capacity)
   and "demand" (destination | quantity) sheets. The model is built
   automatically.

If FILE does not exist, the script creates a filled-in example model (the
study-hours problem from class) to serve as a template.

OUTPUTS
- Optimal solution, objective function value, constraint slacks and, for
  continuous problems, shadow prices (how much the objective improves per
  extra unit of resource): the equivalent of the Solver sensitivity report.

Installation (once, in the Anaconda Prompt):
    pip install pandas numpy scipy openpyxl
"""

#%% CONFIGURATION (the only place to edit)

FILE = "example_study_hours.xlsx"   # relative to the script folder
MODEL_TYPE = "general"           # "general" or "transportation"
SENSE = "max"                    # "max" (maximize) or "min" (minimize)
SUPPLY_EQUAL = False             # transportation: True = each origin ships ALL of
                                 # its supply (=); False = up to capacity (<=)
TIME_LIMIT_SECONDS = 60          # limit for large integer problems

OUTPUT_FOLDER = "results"


#%% Packages

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


#%% Utility functions

SCRIPT_DIR = (Path(__file__).resolve().parent
              if "__file__" in globals() else Path.cwd())
OUTPUT = (SCRIPT_DIR / OUTPUT_FOLDER).resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
REPORT = []


def log(text=""):
    print(text)
    REPORT.append(str(text))


def resolve_path(relative):
    p = Path(relative)
    return p if p.is_absolute() else (SCRIPT_DIR / p).resolve()


def create_example(target):
    """Generate the template spreadsheet with the study-hours problem from class."""
    variables = pd.DataFrame({
        "name": ["H1", "H2"],
        "objective_coef": [15 / 8, 20 / 8],     # G = (15·H1 + 20·H2)/8
        "lower_bound": [0, 0],
        "upper_bound": [None, None],
        "type": ["continuous", "continuous"]})
    constraints = pd.DataFrame({
        "name": ["available_time", "min_grade_subject1", "min_grade_subject2"],
        "sign": ["<=", ">=", ">="],
        "rhs": [30, 10, 12.5],
        "H1": [1, 1, 0],
        "H2": [1, 0, 1]})
    read_me = pd.DataFrame({"instructions": [
        "Sheet 'variables': one row per decision variable.",
        "type: continuous, integer or binary. Empty bounds = no bound.",
        "Sheet 'constraints': one row per constraint; sign <=, >= or equal (do not use '=' alone);",
        "one column per variable (same name as in the 'variables' sheet) with the coefficient.",
        "Empty coefficient = 0. Set SENSE (max/min) in the script."]})
    with pd.ExcelWriter(target) as writer:
        variables.to_excel(writer, sheet_name="variables", index=False)
        constraints.to_excel(writer, sheet_name="constraints", index=False)
        read_me.to_excel(writer, sheet_name="read_me", index=False)


#%% 1. Read the model

file = resolve_path(FILE)
if not file.exists():
    create_example(file)
    log(f"- File not found: the example model `{file.name}` was created.")

if MODEL_TYPE == "transportation":
    costs = pd.read_excel(file, sheet_name="costs", index_col=0)
    supply = pd.read_excel(file, sheet_name="supply", index_col=0).iloc[:, 0]
    demand = pd.read_excel(file, sheet_name="demand", index_col=0).iloc[:, 0]
    origins, destinations = list(costs.index), list(costs.columns)
    names = [f"{o} -> {d}" for o in origins for d in destinations]
    c = costs.values.ravel().astype(float)
    rows, signs, rhs, constraint_names = [], [], [], []
    for i, o in enumerate(origins):
        a = np.zeros(len(names)); a[i * len(destinations):(i + 1) * len(destinations)] = 1
        rows.append(a); signs.append("=" if SUPPLY_EQUAL else "<=")
        rhs.append(float(supply[o])); constraint_names.append(f"supply_{o}")
    for j, d in enumerate(destinations):
        a = np.zeros(len(names)); a[j::len(destinations)] = 1
        rows.append(a); signs.append(">="); rhs.append(float(demand[d]))
        constraint_names.append(f"demand_{d}")
    A = np.array(rows)
    lb, ub = np.zeros(len(names)), np.full(len(names), np.inf)
    integrality = np.zeros(len(names))
    sense = "min"
    if supply.sum() < demand.sum():
        log(f"- Warning: total supply ({supply.sum()}) < total demand "
            f"({demand.sum()}): the problem is infeasible.")
else:
    tab_v = pd.read_excel(file, sheet_name="variables")
    tab_c = pd.read_excel(file, sheet_name="constraints")
    tab_v.columns = [str(x).strip().lower() for x in tab_v.columns]
    names = tab_v["name"].astype(str).str.strip().tolist()
    c = tab_v["objective_coef"].fillna(0).astype(float).values
    lb = tab_v.get("lower_bound", pd.Series(0, index=tab_v.index)).fillna(0).astype(float).values
    ub = tab_v.get("upper_bound", pd.Series(np.nan, index=tab_v.index)).astype(float) \
        .fillna(np.inf).values
    types = tab_v.get("type", pd.Series("continuous", index=tab_v.index)).fillna("continuous") \
        .astype(str).str.lower().str.strip()
    integrality = types.isin(["integer", "binary"]).astype(int).values
    ub = np.where(types.isin(["binary"]), 1, ub)
    tab_c.columns = [str(x).strip() for x in tab_c.columns]
    missing_cols = [v for v in names if v not in tab_c.columns]
    if missing_cols:
        raise KeyError(f"The 'constraints' sheet has no columns for the variables: {missing_cols}")
    A = tab_c[names].fillna(0).astype(float).values
    # In Excel, typing "=" starts a formula: also accept "equal"/"==" and "<="/"le" etc.
    sign_map = {"<=": "<=", "≤": "<=", "less_equal": "<=", "le": "<=",
                ">=": ">=", "≥": ">=", "greater_equal": ">=", "ge": ">=",
                "=": "=", "==": "=", "equal": "=", "eq": "="}
    signs = [sign_map.get(str(v).strip().lower()) for v in tab_c["sign"]]
    invalid = [n for n, v in zip(tab_c["name"], signs) if v is None]
    if invalid:
        raise ValueError("Invalid sign in constraints " + str(invalid) +
                         ". Use <=, >= or 'equal' (in Excel, '=' alone starts a formula).")
    rhs = tab_c["rhs"].astype(float).tolist()
    constraint_names = tab_c["name"].astype(str).tolist()
    sense = SENSE.lower()

log("## 1. Model")
log(f"- File: `{file.name}` | type: {MODEL_TYPE} | objective: "
    f"{'maximize' if sense == 'max' else 'minimize'}")
log(f"- Variables: {len(names)} ({int(integrality.sum())} integer/binary) | "
    f"constraints: {len(rhs)}")


#%% 2. Solve
# scipy minimizes; to maximize, it minimizes -objective.

obj_sign = -1 if sense == "max" else 1
low_r = np.array([r if s in (">=", "=") else -np.inf for s, r in zip(signs, rhs)])
high_r = np.array([r if s in ("<=", "=") else np.inf for s, r in zip(signs, rhs)])
result = milp(c=obj_sign * c, constraints=LinearConstraint(A, low_r, high_r),
              integrality=integrality, bounds=Bounds(lb, ub),
              options={"time_limit": TIME_LIMIT_SECONDS})

log("\n## 2. Result")
if result.x is None:
    log(f"- No solution: {result.message}")
    log("- Check for conflicting constraints (infeasible) or an unbounded "
        "objective (unbounded: some capacity constraint is missing).")
else:
    x = np.round(result.x, 8) + 0.0
    value = float(c @ x)
    log(f"- Status: {result.message}")
    log(f"- **Optimal objective function value: {value:,.4f}**")
    solution = pd.DataFrame({"variable": names, "value": x,
                             "objective_contribution": c * x})
    if MODEL_TYPE == "transportation":
        matrix = pd.DataFrame(x.reshape(len(origins), len(destinations)),
                              index=origins, columns=destinations)
        log("\n### Quantities shipped (origin x destination)")
        log(matrix.to_string())
        matrix.to_csv(OUTPUT / "shipment_matrix.csv", encoding="utf-8-sig")
    else:
        log("\n### Variable values")
        log(solution.to_string(index=False))

    usage = A @ x
    slacks = pd.DataFrame({"constraint": constraint_names, "sign": signs, "rhs": rhs,
                           "value_used": usage,
                           "slack": [r - u if s == "<=" else u - r if s == ">=" else 0
                                     for s, r, u in zip(signs, rhs, usage)]})
    slacks["binding"] = slacks["slack"].abs() < 1e-6

    # Shadow prices (continuous problems only): solve the LP with HiGHS and
    # read the constraint multipliers.
    if integrality.sum() == 0:
        idx_le = [i for i, s in enumerate(signs) if s in ("<=", ">=")]
        idx_eq = [i for i, s in enumerate(signs) if s == "="]
        A_ub = np.array([A[i] * (1 if signs[i] == "<=" else -1) for i in idx_le]) if idx_le else None
        b_ub = np.array([rhs[i] * (1 if signs[i] == "<=" else -1) for i in idx_le]) if idx_le else None
        lp = linprog(obj_sign * c, A_ub=A_ub, b_ub=b_ub,
                     A_eq=A[idx_eq] if idx_eq else None,
                     b_eq=np.array(rhs)[idx_eq] if idx_eq else None,
                     bounds=list(zip(lb, [None if np.isinf(u) else u for u in ub])),
                     method="highs")
        shadow = np.zeros(len(rhs))
        if lp.status == 0:
            for k, i in enumerate(idx_le):
                shadow[i] = obj_sign * lp.ineqlin.marginals[k] * (1 if signs[i] == "<=" else -1)
            for k, i in enumerate(idx_eq):
                shadow[i] = obj_sign * lp.eqlin.marginals[k]
        slacks["shadow_price"] = np.round(shadow, 6) + 0.0
        reduced_cost = obj_sign * lp.lower.marginals if lp.status == 0 else np.nan
        solution["reduced_cost"] = reduced_cost

    log("\n### Constraints (slack and shadow price)")
    log(slacks.to_string(index=False))
    solution.to_csv(OUTPUT / "solution.csv", index=False, encoding="utf-8-sig")
    slacks.to_csv(OUTPUT / "constraints.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUTPUT / "optimization_result.xlsx") as writer:
        solution.to_excel(writer, sheet_name="solution", index=False)
        slacks.to_excel(writer, sheet_name="constraints", index=False)

log("\n## 3. How to interpret")
log("- BINDING constraint (slack 0) = resource used up: it is what limits the result.")
log("- Shadow price = how much the objective improves with +1 unit of that "
    "constraint's right-hand side (valid for small changes).")
log("- Reduced cost = how much the coefficient of a zero-valued variable would "
    "have to improve for it to enter the solution.")
log("- Integer problems have no shadow prices (the solution does not change "
    "continuously).")

(OUTPUT / "report.md").write_text("# Report: Optimization\n\n" + "\n".join(REPORT)
                                  + "\n", encoding="utf-8")
