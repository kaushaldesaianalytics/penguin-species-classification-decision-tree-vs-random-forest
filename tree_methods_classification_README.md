# Penguin Species Classification
## Decision Tree and Random Forest Classifiers

Which physical measurements best predict penguin species, and how much does an ensemble improve on a single tree? This project builds both a Decision Tree and a Random Forest classifier on the Palmer Penguins dataset, using the shared dataset to make a direct performance and stability comparison between the two approaches. A second dataset (UCI Banknote Authentication) extends the Random Forest section into GridSearchCV tuning and OOB scoring.

---

## Overview

Decision Trees and Random Forests represent two points on the same spectrum: a single tree is fully interpretable but prone to high variance, while a forest of many trees trades some interpretability for substantially better generalization. By applying both to identical data, this project makes that tradeoff concrete and measurable.

---

## Datasets

### Palmer Penguins

344 penguin records across three species from the Palmer Archipelago, Antarctica. After removing 10 records with missing values, 334 clean records remain.

| Feature | Description |
|---|---|
| culmen_length_mm | Length of the culmen in mm |
| culmen_depth_mm | Depth of the culmen in mm |
| flipper_length_mm | Flipper length in mm |
| body_mass_g | Body mass in grams |
| island | Island of observation (one-hot encoded) |
| sex | Biological sex (one-hot encoded) |
| **species** | **Target: Adelie, Chinstrap, or Gentoo** |

### UCI Banknote Authentication

1,372 records of genuine and counterfeit banknotes, derived from wavelet transforms of scanned images. Used for GridSearchCV tuning and error rate analysis.

| Feature | Description |
|---|---|
| variance | Variance of wavelet-transformed image |
| skewness | Skewness of wavelet-transformed image |
| curtosis | Kurtosis of wavelet-transformed image |
| entropy | Image entropy |
| **Class** | **Binary target: 0 = genuine, 1 = counterfeit** |

---

## Workflow

### Part 1: EDA
Missing value audit, species/island/sex distributions, scatter plots, and pairplot confirm strong visual separation between species in multiple feature combinations. This signals that tree-based classifiers should achieve high accuracy with relatively shallow splits.

### Part 2: Feature Engineering
Island and sex are one-hot encoded. A 70/30 train/test split is shared across all models in Parts 3 through 6 for consistent comparison.

### Part 3: Decision Tree Baseline
An unconstrained `DecisionTreeClassifier` grows until all leaves are pure. The full learned tree is rendered with `plot_tree`, making every split condition, Gini impurity, and class distribution visible. Gini-based feature importances identify body mass as the strongest single predictor.

### Part 4: Decision Tree Hyperparameter Exploration
Three pruning strategies are compared using a shared `report_model` evaluation function:

- `max_depth=2`: limits tree depth to the two most discriminative splits
- `max_leaf_nodes=3`: caps terminal nodes while allowing variable depth
- `criterion='entropy'`: switches from Gini impurity to information gain

### Part 5: Random Forest Baseline
A 10-tree Random Forest is trained on the same split used for the Decision Tree. Predictions are aggregated by majority vote across trees. Feature importances are compared against the single tree to show how ensemble aggregation produces more stable importance estimates.

### Part 6: Tree Count Sweep
n_estimators is swept from 1 to 40 and test error is recorded at each step. The error curve reveals where the ensemble stabilizes on the penguin dataset. On this small dataset, performance plateaus around 10 trees.

### Part 7: Banknote Authentication
Four Random Forest hyperparameters are tuned simultaneously via 5-fold GridSearchCV: n_estimators, max_features, bootstrap, and oob_score. The best estimator is evaluated on a held-out test set. OOB score is reported as an independent validation metric. The tree count sweep is repeated on the larger dataset to show that bigger datasets benefit from more trees before reaching diminishing returns.

---

## Results

### Penguin Dataset

| Model | Accuracy |
|---|---|
| Decision Tree (unconstrained) | ~97% |
| Decision Tree (max_depth=2) | ~97% |
| Random Forest (10 trees) | ~97% |

Both models achieve similar accuracy on this dataset. The Random Forest advantage becomes more visible in stability: the single tree produces one specific boundary, while the forest generalizes the decision across 10 different bootstrapped views of the data.

### Banknote Dataset

| Metric | Value |
|---|---|
| GridSearchCV best accuracy | ~100% |
| OOB score | ~99.9% |
| Error stabilizes after | ~30 trees |

The banknote dataset achieves near-perfect classification due to strong linear separation in wavelet features.

---

## Key Concepts

**Single Tree vs. Ensemble:** A single decision tree can overfit by learning splits specific to training noise. Random Forest decorrelates trees by training each on a bootstrapped sample with a random feature subset, reducing variance without increasing bias.

**Feature Importance Stability:** Single tree importances can change significantly with small data changes. Random Forest importance averages across many trees, producing more reliable feature rankings.

**Bootstrap Aggregation:** Each tree sees a random 63% of training data (with replacement). The remaining 37% (out-of-bag samples) can be used to estimate generalization error without a separate validation split.

**Diminishing Returns in Tree Count:** Additional trees improve performance until the ensemble has seen sufficient diversity. The optimal count depends on dataset size and complexity, as shown by comparing the penguin and banknote error curves.

**Gini vs. Entropy:** Both measure split quality and typically produce similar trees. Gini is computationally cheaper; entropy can occasionally identify different optimal thresholds on borderline splits.

---

## Stack

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn (DecisionTreeClassifier, RandomForestClassifier, GridSearchCV, plot_tree, ConfusionMatrixDisplay)

---

## File Structure

```
tree-methods-classification/
├── tree_methods_classification.ipynb   # Main project notebook
├── penguins_size.csv                   # Palmer Penguins dataset
├── data_banknote_authentication.csv    # UCI Banknote Authentication dataset
└── README.md
```

---

## How to Run

1. Clone the repository
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Open `tree_methods_classification.ipynb` in Jupyter and run all cells
