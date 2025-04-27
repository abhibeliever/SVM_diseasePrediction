# SVM_diseasePrediction
It will help in the prediction of the cancer cell prediction


## Support Vector Machine (SVM) for Cancer Cell Classification

**Overview**  
This project uses a **Support Vector Machine (SVM)** model to predict whether a cancer cell sample is **malignant** or **benign**. We train on features such as radius, texture, perimeter, area, smoothness, etc., from the Breast Cancer Wisconsin dataset.

---

### What Is SVM?

- **Definition:**  
  SVM is a **max-margin** classifier that finds the hyperplane which best separates classes in feature space.
- **Key Concepts:**  
  - **Support vectors:** The data points closest to the decision boundary that “support” the optimal hyperplane.  
  - **Margin:** Distance between the support vectors of each class and the hyperplane; SVM maximizes this margin.  
  - **Kernel trick:** Transforms data into higher-dimensional space for non-linearly separable problems (common kernels: linear, polynomial, RBF).

---

### Why Use SVM for Cancer Prediction?

- **Effective in High Dimensions:** Works well when the number of features is large relative to samples.  
- **Robust to Overfitting:** The margin maximization principle provides good generalization, especially with proper regularization (`C` parameter).  
- **Flexibility via Kernels:** Can capture non-linear relationships in the data without explicitly mapping to high dimensions.  
- **Clear Decision Boundary:** Produces a deterministic boundary that’s easy to inspect in feature space (or understood via support vectors).

---

### Hyperparameter Tuning with Grid Search

To get the best performance, we tune key SVM hyperparameters using **Grid Search with Cross-Validation**:

- **parameter grid is simply a mapping of hyperparameter names to the list of values you want to try for each. Grid search will then exhaustively train and evaluate your model on every possible combination of those values.**

1. **Parameters to tune:**  
   - `C` (regularization strength)  
   - `kernel` (e.g. `linear`, `rbf`, `poly`)  
   - `gamma` (kernel coefficient for `rbf`/`poly`)  
   - `degree` (for `poly` kernel)  

2. **Example code:**
   ```python
   from sklearn.svm import SVC
   from sklearn.model_selection import GridSearchCV

   # define parameter grid
   param_grid = {
     'C': [0.1, 1, 10],
     'kernel': ['linear', 'rbf'],
     'gamma': ['scale', 0.01, 0.1],
   }

   # set up grid search with 5-fold CV
   grid = GridSearchCV(
     estimator=SVC(),
     param_grid=param_grid,
     cv=5,
     scoring='accuracy',
     n_jobs=-1
   )

   # fit and find best params
   grid.fit(X_train, y_train)
   print("Best parameters:", grid.best_params_)
   print("CV accuracy:", grid.best_score_)
