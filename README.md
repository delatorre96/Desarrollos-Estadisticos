# Regression-Analysis-Toolkit

This repository contains scripts and utilities for data analysis and regression modeling tasks. The scripts are organized into several modules for different aspects of the analysis process.

## Modules

### 1. `data_utils.py`

This module provides utility functions for data manipulation and preprocessing.

- **Functions:**
  - `columnasCategoricas(df)`: Identifies categorical columns in the dataframe.
  - `vars_interaccion(df)`: Creates interaction variables based on pairwise multiplication of columns.
  - Add other data manipulation functions as needed.

### 2. `instrument_selection.py`

This module contains functions for selecting exogenous and endogenous instruments.

- **Functions:**
  - `instrumentos_exogenos(df, var_ind, var_dep, minUmbral, funcSeleccionVars)`: Identifies exogenous instruments.
  - `vars_endogena(df, var_ind, var_dep, minUmbral, funcSeleccionVars)`: Identifies endogenous variables and their instruments.
  - Add other instrument selection functions as needed.

### 3. `regression_analysis.py`

This module includes functions for regression analysis and handling multicollinearity.

- **Functions:**
  - `residuo(varsIndependientes, varDependiente, df, intercept=False)`: Calculates residuals of a regression model.
  - `multicolinealidad(X)`: Detects multicollinearity in independent variables.
  - Add other regression analysis functions as needed.

### 4. `variable_fussion.py`

This module deals with merging or fusing variables from different sources.

- **Functions:**
  - Include functions related to merging or fusing variables.
  - Ensure functions are well-documented and modular for reusability.

### 5. `variable_selection.py`

This module contains functions for selecting important variables in regression models.

- **Functions:**
  - `varsMasCorrelacionadas(df, varDependiente, minCorr)`: Identifies variables most correlated with the dependent variable.
  - `varsMasImportantesArbol(df, varDependiente, minUmbral)`: Identifies important variables using decision tree models.
  - Add other variable selection functions as needed.

## Usage

To use these modules, import them into your Python scripts or notebooks as needed. Here's an example of how you might import and use functions from these modules:

```python
from data_utils import columnasCategoricas, vars_interaccion
from instrument_selection import instrumentos_exogenos, vars_endogena
from regression_analysis import residuo, multicolinealidad
from variable_selection import varsMasCorrelacionadas, varsMasImportantesArbol

# Example usage
# Use functions from each module as per your analysis requirements
