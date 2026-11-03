# Data Science Toolkit – Paulo Berlanda

A collection of **ready-to-use data science tools in Python**, one per technique: from clustering and regression to time series, tree ensembles, neural networks, text mining, optimization and network analysis.

Each tool is a single, well-documented script. You edit one **CONFIGURATION** block (file, columns, options), run it, and get a `results/` folder with tables, charts and a written report (`report.md`) that explains the findings. Where a decision has a standard criterion (number of clusters, number of factors, best forecasting model, cutoff, overdispersion…), the script makes it automatically and tells you why. You can always override it.

The tools were developed from the course material of the **MBA in Data Science and Analytics (USP/Esalq)** and generalized into reusable working tools.

---

## 📂 How the repository is organized

Each model lives in **its own branch**. Switch branches with the selector at the top of the file list, or use the links below. Every branch contains:

- the **Python script** (`.py`), fully in English;
- an **example dataset**, so you can run it immediately;
- a **README.md** user guide: what the model is, where it applies, the workflow, inputs, outputs and how to interpret the results;
- a `requirements.txt`.

| # | Branch | Technique | Use it to |
|---|---|---|---|
| 01 | [`01-Cluster-Analysis`](https://github.com/PRBerlanda/Data-Science/tree/01-Cluster-Analysis) | Hierarchical clustering + k-means | Segment observations described by numeric variables |
| 02 | [`02-PCA-Factor-Analysis`](https://github.com/PRBerlanda/Data-Science/tree/02-PCA-Factor-Analysis) | Principal component factor analysis | Reduce many correlated variables into factors, build indices and rankings |
| 03 | [`03-Correspondence-Analysis`](https://github.com/PRBerlanda/Data-Science/tree/03-Correspondence-Analysis) | Simple (ANACOR) and multiple (MCA) correspondence analysis | Find associations between categorical variables on a perceptual map |
| 04 | [`04-Linear-Regression`](https://github.com/PRBerlanda/Data-Science/tree/04-Linear-Regression) | OLS + stepwise + Box-Cox + diagnostics | Explain and predict a continuous target |
| 05 | [`05-Binary-Logistic-Regression`](https://github.com/PRBerlanda/Data-Science/tree/05-Binary-Logistic-Regression) | Binary logit | Estimate the probability of an event (2 classes) |
| 06 | [`06-Multinomial-Logistic-Regression`](https://github.com/PRBerlanda/Data-Science/tree/06-Multinomial-Logistic-Regression) | Multinomial logit | Estimate probabilities of 3+ unordered classes |
| 07 | [`07-Count-Data-Models`](https://github.com/PRBerlanda/Data-Science/tree/07-Count-Data-Models) | Poisson, Negative Binomial, ZIP, ZINB | Model counts (0, 1, 2, …) |
| 08 | [`08-Multilevel-Modeling`](https://github.com/PRBerlanda/Data-Science/tree/08-Multilevel-Modeling) | Hierarchical linear models (HLM2 / HLM3) | Model data nested in groups or repeated measures |
| 09 | [`09-Time-Series-Forecasting`](https://github.com/PRBerlanda/Data-Science/tree/09-Time-Series-Forecasting) | Benchmarks, ETS / Holt-Winters, SARIMA | Forecast a univariate time series |
| 10 | [`10-Trees-and-Ensembles`](https://github.com/PRBerlanda/Data-Science/tree/10-Trees-and-Ensembles) | Decision tree, Random Forest, XGBoost, LightGBM + SHAP | Get the best prediction (class or value) and explain it |
| 11 | [`11-Neural-Networks-Tabular`](https://github.com/PRBerlanda/Data-Science/tree/11-Neural-Networks-Tabular) | Multilayer perceptron (scikit-learn or Keras) | Predict with a neural network on tabular data |
| 12 | [`12-LSTM-Time-Series`](https://github.com/PRBerlanda/Data-Science/tree/12-LSTM-Time-Series) | LSTM recurrent network | Forecast a series with a recurrent neural network |
| 13 | [`13-Text-Mining-NLP`](https://github.com/PRBerlanda/Data-Science/tree/13-Text-Mining-NLP) | Frequencies, LDA topics, TF-IDF + Naive Bayes / logistic | Explore and classify texts, sentiment |
| 14 | [`14-Web-Scraping`](https://github.com/PRBerlanda/Data-Science/tree/14-Web-Scraping) | requests + BeautifulSoup + read_html | Collect tables and items from web pages |
| 15 | [`15-Linear-Programming`](https://github.com/PRBerlanda/Data-Science/tree/15-Linear-Programming) | Linear / integer programming (HiGHS) | Solve optimization problems from a spreadsheet (Solver replacement) |
| 16 | [`16-Social-Network-Analysis`](https://github.com/PRBerlanda/Data-Science/tree/16-Social-Network-Analysis) | Centrality + community detection | Analyze relationship networks |

---

## 🧭 Which tool should I use?

```
Do you have a target variable (Y)?
├─ No ─► numeric variables? ─ yes ─► 01 Cluster (groups) or 02 PCA (indices)
│                           └ no ──► 03 Correspondence Analysis
│        texts? ─► 13 · relationships between actors? ─► 16 · optimal decision? ─► 15
│        need to collect data from the web first? ─► 14
└─ Yes ─► Y over time (a series)? ─► 09 (classical) or 12 (LSTM)
          continuous Y ─► 04 (explain) or 10 / 11 (predict)
          binary Y ─► 05 (explain, odds ratios) or 10 / 11 (predict)
          Y with 3+ classes ─► 06 or 10
          count Y ─► 07
          observations nested in groups ─► 08
```

Rule of thumb: tools **04 to 08** focus on **explaining** (coefficients, tests, assumptions), and tools **10 to 12** focus on **predicting** (cross-validation, out-of-sample performance).

---

## ⚙️ Common features

- Reads `.csv` (separator and encoding detected), `.xlsx/.xls`, `.parquet`, `.pkl` and `.json`.
- Paths are relative to the script's folder, so data and script travel together.
- Automatic dummies for categorical variables, fixed random seeds, no data leakage in the predictive pipelines.
- Charts can be shown on screen or only saved (`SHOW_PLOTS = False` for batch runs).
- Predictive tools can score a new file (`PREDICT_FILE`) and save the trained model.

## 🚀 Getting started

```bash
git clone https://github.com/PRBerlanda/Data-Science.git
cd Data-Science
git checkout 04-Linear-Regression        # or any other branch
pip install -r requirements.txt
python linear_regression.py              # or open it in Spyder / VS Code
```

TensorFlow is optional and only needed for tool 11 with `ENGINE = "keras"` and for tool 12 (LSTM).

## 🛠️ Tech stack

Python · pandas · NumPy · SciPy · statsmodels · scikit-learn · XGBoost · LightGBM · SHAP · TensorFlow/Keras · factor_analyzer · prince · NetworkX · BeautifulSoup · Matplotlib · seaborn

## 📬 About

Paulo Berlanda, Data Analyst. See also my [Power BI portfolio](https://github.com/PRBerlanda/PowerBI).
