# 11 · Neural Networks (MLP) for Tabular Data

A ready-to-use tool that predicts a class or a value with a **multilayer perceptron**, a neural network made of dense layers. It searches the architecture by cross-validation with early stopping, compares the network with a linear reference model, evaluates it on a test set, shows the learning curve and saves the trained model. It runs with scikit-learn by default, or with TensorFlow/Keras.

> Part of the [Data-Science](https://github.com/PRBerlanda/Data-Science/tree/main) repository. Each branch holds one model.

---

## What this model does

A multilayer perceptron passes the input variables through one or more **hidden layers** of neurons. Each neuron computes a weighted sum of its inputs and applies a non-linear activation (ReLU). Stacking layers lets the network learn complex, non-linear relationships. The weights are learned by gradient descent, minimizing a loss function (cross-entropy for classification, squared error for regression).

Because neural networks can easily overfit small tables, the tool always benchmarks the network against a **linear reference** (logistic or linear regression) and uses **early stopping** and **L2 regularization**.

## When to use it

- Tabular prediction problems with enough data and potentially non-linear relationships.
- As a contender alongside tree ensembles ([10](https://github.com/PRBerlanda/Data-Science/tree/10-Trees-and-Ensembles)) when you want to see whether a neural network adds value.

**Use another model when** the data is small or you need interpretable coefficients ([04](https://github.com/PRBerlanda/Data-Science/tree/04-Linear-Regression)–[06](https://github.com/PRBerlanda/Data-Science/tree/06-Multinomial-Logistic-Regression)), or when the data is a sequence over time ([12 · LSTM](https://github.com/PRBerlanda/Data-Science/tree/12-LSTM-Time-Series)).

## How it works

1. **Load the data** and detect classification vs. regression (or set it). Columns that look like identifiers are removed.
2. **Train/test split** (stratified for classification).
3. **Preprocessing inside a pipeline**: imputation, **standardization** (networks are sensitive to scale) and one-hot encoding, fitted only on the training data.
4. **Linear reference** scored by cross-validation.
5. **Architecture search** with randomized cross-validation over:
   - hidden layer sizes (e.g. `(16,)`, `(32, 16)`);
   - L2 regularization strength;
   - learning rate;
   with early stopping on a validation slice.
6. **Comparison** of the network with the linear reference, and **evaluation on the test set**.
7. **Learning curve** (loss per epoch) and **permutation importance** on the test set.
8. **Save** the model and, optionally, predict new data.

## Inputs

**Data:** one row per case with the target and the explanatory variables.

**Configuration** (the `CONFIGURATION` cell):

| Parameter | What it controls |
|---|---|
| `FILE`, `SHEET`, `SEPARATOR` | Input file, Excel sheet and CSV separator. |
| `TARGET`, `X_VARIABLES`, `ID_COLUMN`, `IGNORE_COLUMNS` | Target, explanatory variables and columns to exclude. |
| `PROBLEM_TYPE`, `POSITIVE_CLASS` | `"auto"`, `"classification"` or `"regression"`, and the class of interest in binary problems. |
| `ENGINE` | `"sklearn"` (no TensorFlow needed) or `"keras"`. |
| `ARCHITECTURES` | Candidate hidden-layer structures, each a tuple of neurons per layer. |
| `L2_REGULARIZATION`, `LEARNING_RATES` | Candidate values for the search. |
| `N_SEARCH_ITER`, `FOLDS` | Number of combinations tested and cross-validation folds. |
| `MAX_EPOCHS`, `PATIENCE` | Training length and epochs without improvement before early stopping. |
| `TEST_PROPORTION`, `METRIC`, `SEED` | Test share, selection metric and seed. |
| `PREDICT_FILE` | Optional file with new cases to predict. |

## Outputs

Saved in `results/`:

| File | Content |
|---|---|
| `report.md` | Data summary, reference vs. network comparison, best architecture, test metrics, epochs, importances and interpretation. |
| `architecture_search.csv` | Every combination tested with its cross-validated scores. |
| `permutation_importance.csv` | Loss in the metric when each variable is shuffled. |
| `final_model.joblib` | Trained pipeline (scikit-learn engine). |
| `keras_network.keras` + `preprocessor.joblib` | Trained network and preprocessing (Keras engine). |
| `new_data_predictions.csv` | Predictions for `PREDICT_FILE`. |
| `01_observed_predicted.png` / `01_confusion_matrix.png`, `02_learning_curve.png` | Charts. |

## How to read the results

- **Gain over the reference**: if the network does not beat the linear model, prefer the linear model. It is simpler, faster and easier to explain.
- **Cross-validated score ± standard deviation**: expected performance and stability. **Training score much better than validation** means overfitting.
- **Best architecture**: more neurons and layers add capacity but also risk of overfitting. Higher L2 means stronger regularization.
- **Learning curve**: training loss should go down and level off. If the validation loss rises while the training loss keeps falling, the network is overfitting: increase regularization or shrink the network.
- **Epochs trained**: early stopping halts training when the validation score stops improving.
- **Test metrics**: accuracy, macro F1 and AUC (classification) or RMSE, MAE and R² (regression) on data never used in training.
- **Permutation importance**: which variables the network relies on most (not causality).
- Neural networks shine with large datasets. On small tables, linear models or tree ensembles often tie or win.

## How to run

```bash
pip install -r requirements.txt
# optional, only for ENGINE = "keras":
pip install tensorflow
```

1. Put your data file next to `neural_networks_tabular.py`.
2. Edit the `CONFIGURATION` cell (at least `FILE`, `TARGET` and `PROBLEM_TYPE`).
3. Run the script in Spyder (F5) or with `python neural_networks_tabular.py`.
4. Open `results/report.md`.

## Example dataset

`winequality-red.csv` is the UCI red wine quality dataset: eleven physicochemical measurements (acidity, residual sugar, chlorides, sulfur dioxide, density, pH, sulphates, alcohol…) and the sensory quality score (`quality`, the target), modeled as a regression in the default configuration.

## Credits

Built from the Deep Learning course (Classes 1–3) and the Keras Titanic script of "Trees, Networks and Ensembles" (MBA in Data Science and Analytics, USP/Esalq).
