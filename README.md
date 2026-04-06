# LogisticLearn â€” Exact Build Specification

## What This Is

A from-scratch implementation of Binary Logistic Regression using only NumPy.
No sklearn for modeling. No autograd. No shortcuts.
The goal is to own every line mathematically â€” not just have code that produces a number.

---

## Project Structure

```
LogisticLearn/
â”œâ”€â”€ logistic_regression.py       # Core model class
â”œâ”€â”€ cross_validation.py          # K-Fold, Stratified K-Fold, GridSearchCV
â”œâ”€â”€ utils.py                     # Data generation, cloning, pure loss helper
â”œâ”€â”€ data/
â”‚   â””â”€â”€ heart_disease_uci.csv    # Real dataset for benchmarking
â”œâ”€â”€ notebooks/
â”‚   â””â”€â”€ LogisticLearn.ipynb      # Step-by-step walkthrough with explanations
â”œâ”€â”€ docs/
â”‚   â””â”€â”€ theory/                  # Your handwritten or typed math derivations
â”œâ”€â”€ results/                     # Plots, metric tables, convergence gifs
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

---

## What To Build â€” In Order

### Step 1: The Core Model Class

Build a class called `LogisticRegression` with the following:

**Constructor parameters:**
- `lr` â€” learning rate, default 0.001
- `n_itr` â€” number of iterations, default 2000
- `batch_size` â€” if None use full-batch gradient descent, otherwise mini-batch
- `penalty` â€” None, "l1", or "l2"
- `C` â€” inverse regularization strength, default 1.0 (higher C = less regularization)
- `tol` â€” convergence tolerance, default 1e-6

**Input validation on construction â€” raise errors for:**
- `lr`, `n_itr`, `tol`, `C` must be positive numbers
- `batch_size` must be a positive integer or None
- `penalty` must be "l1", "l2", or None â€” case-insensitive, strip whitespace

**Methods to implement:**

`_sigmoid_function(X)` â€” the numerically stable version.
Do NOT implement it as `1 / (1 + np.exp(-logit))` blindly.
When logit is very negative, `np.exp(-logit)` overflows.
The correct implementation splits on the sign of logit:
- When logit >= 0: use `1 / (1 + exp(-logit))`
- When logit < 0: use `exp(logit) / (1 + exp(logit))`
Both are mathematically equivalent but numerically safe.
Also clip logit to [-500, 500] before computing to prevent any remaining overflow.

`cross_entropy_loss(Y, X)` â€” the negative log-likelihood:
```
loss = -mean( Y * log(P) + (1 - Y) * log(1 - P) )
```
Clip P to [1e-15, 1-1e-15] before taking the log â€” log(0) is -inf and will break training.
Add regularization term to the loss if penalty is set:
- L2: `(1 / (2 * C)) * sum(beta^2)`
- L1: `(1 / C) * sum(|beta|)`

`fit(X, Y)` â€” the training loop:
- Initialize `beta` as zeros of shape (p, 1), `intercept` as 0
- Each iteration:
  - If batch_size is None: use all data
  - If batch_size is set: randomly sample `batch_size` rows WITHOUT replacement each iteration
  - Compute gradient of cross-entropy loss w.r.t. beta: `(X.T @ (P - Y)) / n`
  - Compute gradient w.r.t. intercept: `sum(P - Y) / n`
  - Apply the update depending on penalty:
    - **No penalty**: `beta = beta - lr * gradient`
    - **L2**: add `(1/C) * beta` to the gradient before updating. Also clip gradient norm to 1e3 to prevent exploding gradients.
    - **L1**: use the proximal operator (soft-thresholding) â€” do NOT just subtract the subgradient:
      ```
      beta_temp = beta - lr * gradient
      threshold = lr * (1/C)
      beta = sign(beta_temp) * max(|beta_temp| - threshold, 0)
      ```
      This is the correct way to apply L1 regularization in gradient descent. The naive subgradient approach does not produce sparse solutions properly.
  - Update intercept: `intercept = intercept - lr * gradient_intercept`
  - Compute and store the loss each iteration
  - Early stopping: if `abs(prev_loss - current_loss) < tol`, stop training
- Return `beta` and the list of losses

`predict(X, threshold)` â€” returns:
- `predicted_class`: 1 if probability >= threshold, else 0
- `predicted_probabilities`: the raw sigmoid output

`accuracy(Y, Y_pred)` â€” `mean(Y_pred == Y)`

`get_params()` â€” return a dict of all constructor parameters

`set_params(**params)` â€” set attributes from a dict, return self

---

### Step 2: Model Cloning

Implement a standalone `clone(estimator)` function outside the class:
- Get the class of the estimator
- Get its params via `get_params()`
- Return a new instance of the same class initialized with those params

This is required for cross-validation â€” you need a fresh unfitted copy of the model for each fold.

---

### Step 3: Cross-Validation

Build these as standalone functions (not a class, keep it simple):

**`K_folds(n, K, shuffle, seed)`**
- Takes total sample count `n`, number of folds `K`
- Returns a list of `(train_idx, test_idx)` tuples
- The last fold should absorb the remainder when `n % K != 0`
- Shuffle indices before splitting if `shuffle=True`

**`stratified_K_folds(K, y, shuffle, seed)`**
- Splits each class separately into K equal parts, then assembles folds
- Each fold's test set has the same class ratio as the full dataset
- This is critical for imbalanced data â€” plain K-Fold can put all of one class in one fold
- Returns a list of `(train_idx, test_idx)` tuples

**`cross_validate(model, X, Y, seed, shuffle, cv)`**
- Clones the model for each fold
- Fits on train, evaluates on test
- For penalized models: compute the pure cross-entropy loss (no regularization term) on the test fold â€” otherwise you are penalizing test performance which makes no sense
- Returns list of per-fold losses and the mean loss

---

### Step 4: Grid Search

**`grid_searchCV(model, param_grid, cv, X, Y)`**
- param_grid is a dict: `{"lr_values": [...], "C_values": [...]}`
- For every combination of `(lr, C)`: clone the model, set the params, cross-validate
- Track the combination with the lowest average CV loss
- After finding the best params: refit the best model on the FULL dataset
- Return a dict with `best_score`, `best_model`, `best_params`, `score_cv`

---

### Step 5: Data

Use two datasets:

**Breast Cancer (from sklearn.datasets)**
- Load with `datasets.load_breast_cancer()`
- Standardize manually: `(X - mean) / std` â€” do not use sklearn's StandardScaler
- 30 features, binary target

**Heart Disease UCI**
- Download the CSV: `https://archive.ics.uci.edu/ml/datasets/Heart+Disease`
- Store in `data/heart_disease_uci.csv`
- Preprocess: handle missing values, encode categorical features, standardize

Also implement two synthetic data generators for debugging:

**`generate_dummy_data(n, p, seed)`** â€” easy, low-noise data to verify your model learns

**`generate_challenging_data(n, p, seed, noise_level)`** â€” correlated features, only first 5 matter (sparse true coefficients), higher noise â€” use this to verify L1 actually zeroes out irrelevant features

---

### Step 6: Benchmark Against sklearn

For each model variant (no penalty, L2, L1, grid search best), compare:
- Your accuracy vs `sklearn.linear_model.LogisticRegression`
- Your CV loss vs sklearn's
- Your coefficients vs sklearn's

If your numbers are not close, you have a bug. Find it. This is not optional â€” the benchmark is the proof that your implementation is correct.

---

### Step 7: The Notebook

The notebook is not an afterthought â€” it is the explanation layer. Structure it as:

1. Math derivation: sigmoid, log-likelihood, gradient derivation (write it out)
2. Why L1 needs the proximal operator (not just subgradient) â€” derive it
3. Why stratified K-fold matters â€” show what happens with plain K-fold on imbalanced data
4. Build the model step by step with intermediate outputs
5. Plot the loss curve across iterations for each model variant
6. Show the coefficient values for L1 vs L2 â€” L1 should have sparse coefficients
7. GridSearch results: heatmap of CV loss across (lr, C) grid
8. Benchmark table against sklearn

---

## Metrics To Report For Every Model

- Accuracy
- Precision, Recall, F1 (use sklearn.metrics only for this â€” computing predictions is fine)
- AUC-ROC
- Average CV loss across folds
- Training time

Report all of these. Not just the one that looks best.

---

## The Non-Negotiables

These are the things that separate this project from a basic implementation:

1. **Numerically stable sigmoid** â€” the split implementation. If you just write `1 / (1 + exp(-x))` you have not done this correctly.

2. **Proximal operator for L1** â€” soft-thresholding. If you subtract the subgradient of the absolute value you have not done this correctly.

3. **Stratified K-fold** â€” not plain K-fold. Preserves class ratios across folds.

4. **Pure loss for penalized model evaluation** â€” when evaluating a regularized model on the test fold, use the unregularized loss. You are measuring generalization, not the training objective.

5. **Early stopping** â€” convergence check on loss delta, not just running all iterations.

6. **sklearn benchmark** â€” your coefficients and accuracy must be close to sklearn's output. If they are not, you debug until they are.

---

## Requirements

```
numpy
scikit-learn   # for datasets and benchmarking only
matplotlib
jupyter
pandas
```

---

## Definition of Done

This project is done when:
- All three model variants (no penalty, L2, L1) produce results within acceptable margin of sklearn
- L1 model produces visibly sparse coefficients on the challenging synthetic dataset
- GridSearchCV selects params that improve over the default model
- The notebook explains the math behind every implementation decision
- Someone reading the code and notebook can understand exactly what is happening mathematically at every step

If it only works on the breast cancer dataset and breaks on new data, it is not done.
