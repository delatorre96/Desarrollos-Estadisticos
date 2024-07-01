import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from regression_analysis import regresion
import numpy as np

def varsMasCorrelacionadas(df, varDependiente, minCorr=0.5):
    """
    Identifies variables most correlated with the dependent variable.
    
    Parameters:
    df (pandas.DataFrame): The input dataframe.
    varDependiente (str): The dependent variable.
    minCorr (float): Minimum correlation threshold.
    
    Returns:
    list: A list of variable names with correlation values above the threshold.
    """
    y = df[varDependiente]
    correlaciones_dict = {columna: y.corr(df[columna]) for columna in df.columns}
    correlaciones_lista = [key for key, value in correlaciones_dict.items() if value >= minCorr and value < 0.99]
    return correlaciones_lista

def varsMasImportantesArbol(df, varDependiente, minUmbral=0.05):
    """
    Identifies important variables using a decision tree model.
    
    Parameters:
    df (pandas.DataFrame): The input dataframe.
    varDependiente (str): The dependent variable.
    minUmbral (float): Minimum importance threshold.
    
    Returns:
    list: A list of important variable names.
    """
    X = df.copy().drop(varDependiente, axis=1)
    y = df[varDependiente]
    if len(list(y.unique())) == 2:
        modelo_arbol = DecisionTreeClassifier()
    else:
        modelo_arbol = DecisionTreeRegressor()
    modelo_arbol.fit(X, y)
    importancias_variables = modelo_arbol.feature_importances_
    importancias_df = pd.DataFrame({'Variable': X.columns, 'Importancia': importancias_variables})
    importancias_df = importancias_df.sort_values(by='Importancia', ascending=False)
    variables_importantes = importancias_df[importancias_df['Importancia'] > minUmbral]['Variable'].tolist()
    return variables_importantes

def reducir_variables_aleatoriamente(X, y, umbral_condicional, model_reg=True):
    """
    Reduce variables randomly to manage multicollinearity.

    Parameters:
    X : DataFrame
        Independent variables.
    y : Series or ndarray
        Dependent variable.
    umbral_condicional : float
        Threshold for the condition number to manage multicollinearity.
    model_reg : bool, optional
        If True (default), returns RegressionResults after reducing variables.
        If False, returns a list of variables removed to meet the threshold.

    Returns:
    Union[RegressionResults, list]:
        If model_reg=True, returns RegressionResults of the regression after variable reduction.
        If model_reg=False, returns a list of variables removed to manage multicollinearity.
    """
    num_variables_a_eliminar = 1
    vars_used = []
    
    while num_variables_a_eliminar <= len(X.columns):
        X_temp = X.copy()
        
        # Randomly select variables to remove
        variables_to_remove = list(np.random.choice(X.columns, num_variables_a_eliminar, replace=False))
        
        # Remove selected variables
        X_temp = X_temp.drop(variables_to_remove, axis=1)
        
        # Perform regression with reduced variables
        model_reg = regresion(X_temp, y)
        condition_number = model_reg.condition_number
        
        if condition_number <= umbral_condicional:
            break  # Exit loop if condition number is acceptable
        else:
            vars_used.append(variables_to_remove[0])
            
            # If all variables have been tried and condition number still not acceptable
            if len(set(vars_used)) == len(X.columns):
                print(f'With a conditional number of {condition_number}, {num_variables_a_eliminar} variables were randomly removed.')
                num_variables_a_eliminar += 1
                vars_used.clear()
    else:
        print(f'{condition_number} is the smallest conditional number achieved to manage multicollinearity.')
    
    if model_reg:
        return model_reg  # Return RegressionResults if model_reg=True
    else:
        return vars_used  # Return list of removed variables if model_reg=False

