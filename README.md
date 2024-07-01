# Regression-Analysis-Toolkit

This repository contains scripts and utilities for data analysis and regression modeling tasks. The scripts are organized into several modules for different aspects of the analysis process.

## Modules

### 1. `data_utils.py`

This module provides utility functions for data manipulation and preprocessing.

- **Functions:**
  - `columnasCategoricas(df)`: Identifies categorical columns in the dataframe.
  - `vars_interaccion(df)`: Creates interaction variables based on pairwise multiplication of columns.

### 2. `instrument_selection.py`

This module contains functions for selecting exogenous and endogenous instruments.

- **Functions:**
  - `instrumentos_exogenos(df, var_ind, var_dep, minUmbral, funcSeleccionVars)`: Identifies exogenous instruments.
  - `vars_endogena(df, var_ind, var_dep, minUmbral, funcSeleccionVars)`: Identifies endogenous variables and their instruments.

### 3. `regression_analysis.py`

This module includes functions for regression analysis and handling multicollinearity.

- **Functions:**
  - `estimacion (varsIndependientes, varDependiente, df,intercept = False)` :Estimates the dependent variable using a regression model.
  - `regresion(X, y, const=1)` : Perform OLS regression.
  - `residuo(varsIndependientes, varDependiente, df, intercept=False)`: Calculates residuals of a regression model.
  - `multicolinealidad(X)`: Detects multicollinearity in independent variables.
  - `mc2e_tree(df, y, funcSeleccionVars = varsMasCorrelacionadas, minUmbral = 0.4)` : Construct a regression tree using instrumental variables. 
  - `regTree (df, tree, y)` : Perform regression based on a regression tree.
  - `regTreeSinMultiSinX (df, y, funcSeleccionVars, minUmbral, summary = True)` : Perform regression tree without multicollinearity and without using matrix X (this matrix is identified automatically).
  - `regTreeMulti (df, tree, y)` : Perform regression with handling multicollinearity.
  - `regTreeMultiSinX (df, y, minUmbral, funcSeleccionVars = varsMasCorrelacionadas, summary = True)`: Perform regression with handling multicollinearity without firstly using X (it detects best X)

### 4. `variable_fussion.py`

This module deals with merging or fusing variables from different sources.

- **Functions:**
  - `enalzarVariables(dataframe, vars_enlazar, str_nueva_var, porc_varExplicada)`:This function performs variable linking using Principal Component Analysis (PCA).
    It creates a new composite variable that captures a significant amount of explained variance
    from the original set of variables.
  - `busquedaEnlazamiento(dataframe, porcMinCorrelacion)`: This function searches for pairs of variables that have high correlation to be linked together.


### 5. `variable_selection.py`

This module contains functions for selecting important variables in regression models.

- **Functions:**
  - `varsMasCorrelacionadas(df, varDependiente, minCorr)`: Identifies variables most correlated with the dependent variable.
  - `varsMasImportantesArbol(df, varDependiente, minUmbral)`: Identifies important variables using decision tree models.
  - `reducir_variables_aleatoriamente(X, y, umbral_condicional, model_reg=True)` : Reduce variables randomly to manage multicollinearity.

## Usage

To use these modules, import them into your Python scripts or notebooks as needed. Here's an example of how you might import and use functions from these modules:

```python
from data_utils import columnasCategoricas, vars_interaccion
from instrument_selection import instrumentos_exogenos, vars_endogena
from regression_analysis import residuo, multicolinealidad
from variable_selection import varsMasCorrelacionadas, varsMasImportantesArbol

# Example usage
# Use functions from each module as per your analysis requirements
