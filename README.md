# RegressionTools

This repository contains Python functions for performing regression analysis and related tasks, including handling multicollinearity. These functions utilize the `statsmodels` library for regression modeling.

## Functions Included:

1. **regresion(X, y, const=1)**:
   - Perform Ordinary Least Squares (OLS) regression.
   
2. **mc2e_tree(df, y, funcSeleccionVars, minUmbral)**:
   - Construct a regression tree using instrumental variables and handle endogeneity.
   
3. **regTree(df, tree, y)**:
   - Perform regression based on a regression tree.
   
4. **regTreeSinMultiSinX(df, y, funcSeleccionVars, minUmbral, summary=True)**:
   - Perform regression tree without multicollinearity and without using X.
   
5. **reducir_variables_aleatoriamente(X, y, umbral_condicional)**:
   - Reduce variables randomly to manage multicollinearity.
   
6. **regTreeMulti(df, tree, y)**:
   - Perform regression with handling multicollinearity.
   
7. **regTreeMultiSinX(df, y, minUmbral, funcSeleccionVars=varsMasCorrelacionadas)**:
   - Perform regression with handling multicollinearity without using X.

## How to Use:

- Clone the repository: `git clone https://github.com/your-username/RegressionTools.git`
- Import the necessary functions into your Python environment.
- Ensure you have `statsmodels`, `pandas`, and `numpy` installed.

## Example Usage:

```python
import pandas as pd
from RegressionTools import regresion, mc2e_tree, regTree

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Example usage of functions
X = df[['independent_var1', 'independent_var2']]
y = df['dependent_var']

# Perform OLS regression
results = regresion(X, y)

# Construct a regression tree and perform regression
tree = mc2e_tree(df, y)
regression_results = regTree(df, tree, y)
print(regression_results.summary())

