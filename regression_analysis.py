import pandas as pd
import numpy as np
import statsmodels.api as sm
from variable_selection import varsMasCorrelacionadas, varsMasImportantesArbol,reducir_variables_aleatoriamente
from sklearn.linear_model import LinearRegression


def estimacion (varsIndependientes, varDependiente, df,intercept = False):
    """
    Estimates the dependent variable using a regression model.
    
    Parameters:
    varsIndependientes (list): List of independent variables.
    varDependiente (str): Dependent variable.
    df (pandas.DataFrame): The input dataframe.
    intercept (bool): Whether to include an intercept in the regression.
    
    Returns:
    numpy.ndarray: The estimated values of the dependent variable.
    """
    #if list(df[varDependiente].unique()) == [0, 1]:
    #    modelo_regresion = LinearRegression()
    #    modelo_regresion.fit(df[varsIndependientes], df[varDependiente])
    #    varDependiente_hat = modelo_regresion.predict(df[varsIndependientes])
    #    return varDependiente_hat
    #else:
    modelo_regresion = LinearRegression(fit_intercept=intercept)
    modelo_regresion.fit(df[varsIndependientes], df[varDependiente])
    varDependiente_hat = modelo_regresion.predict(df[varsIndependientes])
    return varDependiente_hat
        

def regresion(X, y, const=1):
    """
    Perform OLS regression.

    Parameters:
    X : DataFrame or ndarray
        Independent variables.
    y : Series or ndarray
        Dependent variable.
    const : int, optional
        Whether to include a constant (default is 1, include constant).

    Returns:
    results : RegressionResults
        Results of the regression.
    """
    if const == 1:
        X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    return results

def residuo(varsIndependientes, varDependiente, df, intercept=False):
    """
    Calculates the residuals of a regression model.
    
    Parameters:
    varsIndependientes (list): List of independent variables.
    varDependiente (str): Dependent variable.
    df (pandas.DataFrame): The input dataframe.
    intercept (bool): Whether to include an intercept in the regression.
    
    Returns:
    numpy.ndarray: The residuals of the regression.
    """
    if list(df[varDependiente].unique()) == [0, 1]:
        regresion = sm.Logit(df[varDependiente], df[varsIndependientes])
    else:
        regresion = sm.OLS(df[varDependiente], df[varsIndependientes])
    resultados = regresion.fit()
    v = resultados.resid
    return v

def multicolinealidad(X):
    """
    Detects multicollinearity in the independent variables.
    
    Parameters:
    X : DataFrame or ndarray
        Independent variables.
        
    Returns:
    bool: True if multicollinearity is detected, False otherwise.
    """
    matrix_correlacion = X.corr().abs()
    matrix_correlacion_upper = matrix_correlacion.where(np.triu(np.ones(matrix_correlacion.shape), k=1).astype(np.bool))
    return any(matrix_correlacion_upper > 0.95)

def instrumentos_exogenos (df, var_ind, var_dep, minUmbral, funcSeleccionVars = varsMasCorrelacionadas):
    """
    Identifies exogenous instruments for a set of independent variables.
    
    Parameters:
    df (pandas.DataFrame): The input dataframe.
    var_ind (list): List of independent variables.
    var_dep (str): Dependent variable.
    minUmbral (float): Minimum importance threshold.
    funcSeleccionVars (function): Function to select variables.
    
    Returns:
    dict: A dictionary indicating whether each variable is exogenous.
    """
    alfa = 0.05
    instrumentos_exogenos = {}
    for x_i in var_ind:
        print('Evaluando',x_i,'...')
        #Calculamos las vars más correlacionadas
        minCorr = 1
        while minCorr > 0:
            if funcSeleccionVars == varsMasCorrelacionadas:
                instrumentos_lista = [i for i in varsMasCorrelacionadas (varDependiente = x_i, df = df, minCorr = minUmbral) if i not in var_ind and i not in var_dep]
            if funcSeleccionVars == varsMasImportantesArbol:
                instrumentos_lista = [i for i in varsMasImportantesArbol (df = df, varDependiente = x_i, minUmbral = minUmbral) if i not in var_ind and i not in var_dep]
            if len(instrumentos_lista) >= len(var_ind):
                v = residuo (varsIndependientes = instrumentos_lista, varDependiente = x_i, df = df)
                x_df = df.copy()
                x_df = x_df[var_ind]
                x_df['v'] = v
                x_df = sm.add_constant(x_df)
                model = sm.OLS(df[var_dep], x_df).fit()
                p_values = dict(model.pvalues)            
                if alfa < p_values['v']:
                    instrumentos_exogenos.update({x_i : True})
                    break
                else:
                    minCorr -= 0.01
                    
            else:
                minCorr -= 0.01
        instrumentos_exogenos.update({x_i : False})
    return instrumentos_exogenos
    
def vars_endogena (df, var_ind, var_dep, minUmbral ,funcSeleccionVars = varsMasCorrelacionadas):
    """
    Identifies endogenous variables in a set of independent variables.
    
    Parameters:
    df (pandas.DataFrame): The input dataframe.
    var_ind (list): List of independent variables.
    var_dep (str): Dependent variable.
    minUmbral (float): Minimum importance threshold.
    funcSeleccionVars (function): Function to select variables.
    
    Returns:
    dict: A dictionary of endogenous variables and their instruments.
    """
    alfa = 0.05
    instrumentos = {}
    for x_i in var_ind:
        #Calculamos las vars más correlacionadas
        minCorr = 1
        while minCorr > 0:
            if funcSeleccionVars == varsMasCorrelacionadas:
                instrumentos_lista = [i for i in varsMasCorrelacionadas (varDependiente = x_i, df = df, minCorr = minUmbral) if i not in var_ind and i not in var_dep]
            if funcSeleccionVars == varsMasImportantesArbol:
                instrumentos_lista = [i for i in varsMasImportantesArbol (varDependiente = x_i, df = df, minUmbral = minUmbral) if i not in var_ind and i not in var_dep]               
            if len(instrumentos_lista) >= len(var_ind):
                break
            else:
                minCorr -= 0.01     
        v = residuo (varsIndependientes = instrumentos_lista, varDependiente = x_i, df = df)
        x_df = df.copy()
        x_df = x_df[var_ind]
        x_df['v'] = v
        x_df = sm.add_constant(x_df)
        model = sm.OLS(df[var_dep], x_df).fit()
        p_values = dict(model.pvalues)
        if alfa > p_values['v']:
            instrumentos.update({x_i : instrumentos_lista})
    #Evaluamos los instrumentos sean exógenos dentro de la misma regresion VI
    #for i in instrumentos:
    #    if instrumentos[i] == None:
    #        pass
     #   else:
     #       instrumentos_exogenos (df = df, var_ind = instrumentos[i], var_dep = i)
    
            print('Encontramos que',x_i,'es endógena')
    return instrumentos

def mc2e_tree(df, y, funcSeleccionVars = varsMasCorrelacionadas, minUmbral = 0.4):
    """
    Construct a regression tree using instrumental variables.

    Parameters:
    df : DataFrame
        Dataset containing all variables.
    y : str
        Dependent variable name.
    funcSeleccionVars : function, optional
        Function to select independent variables (default is varsMasCorrelacionadas).
    minUmbral : float, optional
        Minimum correlation threshold for variable selection (default is 0.4).

    Returns:
    tree : dict
        Tree structure representing the regression model.
    """
    if funcSeleccionVars == varsMasCorrelacionadas:
        x = funcSeleccionVars (varDependiente = y, df = df, minCorr = minUmbral)
    if funcSeleccionVars == varsMasImportantesArbol:
        x = funcSeleccionVars (varDependiente = y, df = df, minUmbral = minUmbral)
    tree = {y : x}
    vars_endogenas =  vars_endogena (df = df, var_ind = x, var_dep = y, minUmbral = minUmbral ,funcSeleccionVars = varsMasCorrelacionadas)
    tree_endgogenas = {}
    for i in x:
        if i in vars_endogenas:
            tree_endgogenas.update({i : vars_endogenas[i]})
        else:
            tree_endgogenas.update({i : None})
    tree[y] = tree_endgogenas
    if len(list(tree.values())[0]) == 0:
        print('No existen variables explicativas con esa correlación')
    else:
        print('Se hace estimación por variables instrumentales')
        return tree
        

def regTree (df, tree, y):
    """
    Perform regression based on a regression tree.

    Parameters:
    df : DataFrame
        Dataset containing all variables.
    tree : dict
        Tree structure representing the regression model.
    y : str
        Dependent variable name.

    Returns:
    model_reg : RegressionResults
        Results of the regression.
    """
    df_x = {}
    for i in tree[y]:
        if tree[y][i] != None:
            df_x.update({str(i) + '_hat' : estimacion (varsIndependientes = tree[y][i], varDependiente = i, df = df)})
        else:
            df_x.update({i : df[i]})
    df_x = pd.DataFrame(df_x)
    model_reg = regresion(X = df_x, y = df[y])
    #Evaluamos p_values
    p_values = model_reg.pvalues
    variables_con_p_valor_alto = p_values[p_values > 0.1].index.tolist()
    if len(variables_con_p_valor_alto) > 0:
        print("Eliminando variables con p-valor mayor a 0.1:", variables_con_p_valor_alto)
        if any("const" in variable or "_hat" in variable for variable in variables_con_p_valor_alto):
            variables_con_p_valor_alto = [variable for variable in variables_con_p_valor_alto if "const" not in variable]
            if variables_con_p_valor_alto == []:
                model_reg = regresion(X = df_x, y = df[y], const = 0) 
            else:
                df_x = df_x.drop(variables_con_p_valor_alto, axis = 1)
                model_reg = regresion(X = df_x, y = df[y], const = 0)    
        else:
            df_x = df_x.drop(variables_con_p_valor_alto, axis = 1)
            model_reg = regresion(X = df_x, y = df[y], const = 1)
    return model_reg


def regTreeSinMultiSinX (df, y, funcSeleccionVars, minUmbral, summary = True):
    """
    Perform regression tree without multicollinearity and without using matrix X (this matrix is identified automatically).

    Parameters:
    df : DataFrame
        Dataset containing all variables.
    y : str
        Dependent variable name.
    funcSeleccionVars : function
        Function to select independent variables.
    minUmbral : float
        Minimum correlation threshold for variable selection.
    summary : bool, optional
        Whether to return summary of regression results (default is True).

    Returns:
    tree : dict
        Tree structure representing the regression model.
    summary : str or RegressionResults
        Summary of regression results if summary=True, otherwise None.
    """
    tree = mc2e_tree(df = df, y = y, funcSeleccionVars = varsMasImportantesArbol, minUmbral = 0.005)
    regTree (df = df, tree = tree, y = y)
    if summary == True:
        return tree, regTree.summary()
    else:
        return tree, regTree    




def regTreeMulti (df, tree, y):
    """
    Perform regression with handling multicollinearity.

    Parameters:
    df : DataFrame
        Dataset containing all variables.
    tree : dict
        Tree structure representing the regression model.
    y : str
        Dependent variable name.

    Returns:
    model_reg2 : RegressionResults
        Results of the regression with multicollinearity handling.
    """
    df_x = {}
    for i in tree[y]:
        if tree[y][i] != None:
            df_x.update({str(i) + '_hat' : estimacion (varsIndependientes = tree[y][i], varDependiente = i, df = df)})
        else:
            df_x.update({i : df[i]})
    df_x = pd.DataFrame(df_x)
    model_reg = regresion(X = df_x, y = df[y])
    #Evaluamos p_values
    p_values = model_reg.pvalues
    variables_con_p_valor_alto = p_values[p_values > 0.1].index.tolist()
    if len(variables_con_p_valor_alto) > 0:
        print("Eliminando variables con p-valor mayor a 0.1:", variables_con_p_valor_alto)
        if any("const" in variable or "_hat" in variable for variable in variables_con_p_valor_alto):
            variables_con_p_valor_alto = [variable for variable in variables_con_p_valor_alto if "const" not in variable]
            if variables_con_p_valor_alto == []:
                model_reg = regresion(X = df_x, y = df[y], const = 0) 
            else:
                df_x = df_x.drop(variables_con_p_valor_alto, axis = 1)
                model_reg = regresion(X = df_x, y = df[y], const = 0)    
        else:
            df_x = df_x.drop(variables_con_p_valor_alto, axis = 1)
            model_reg = regresion(X = df_x, y = df[y], const = 1)
    #Evaluamos multicolinealidad
    model_reg2 = reducir_variables_aleatoriamente(X = df_x, y = df[y], umbral_condicional = 100)
    return model_reg2
    

def regTreeMultiSinX (df, y, minUmbral, funcSeleccionVars = varsMasCorrelacionadas, summary = True):
    """
    Perform regression with handling multicollinearity without using X.

    Parameters:
    df : DataFrame
        Dataset containing all variables.
    y : str
        Dependent variable name.
    minUmbral : float
        Minimum correlation threshold for variable selection.
    funcSeleccionVars : function, optional
        Function to select independent variables (default is varsMasCorrelacionadas).

    Returns:
    tree : dict
        Tree structure representing the regression model.
    model_reg : RegressionResults
        Results of the regression with multicollinearity handling.
    """
    while minUmbral > 0:
        tree = mc2e_tree(df, y, funcSeleccionVars = varsMasCorrelacionadas, minUmbral = minUmbral)
        df_x = {}
        for i in tree[y]:
            if tree[y][i] != None:
                df_x.update({str(i) + '_hat' : estimacion (varsIndependientes = tree[y][i], varDependiente = i, df = df)})
            else:
                df_x.update({i : df[i]})
        df_x = pd.DataFrame(df_x)
        print('Vars explicativas:',list(df_x.columns))
        ##Evaluamos multicolinealidad. 
        umbral_condicional = 100
        model_reg = regresion(X = df_x, y = df[y])
        numero_condicional = model_reg.condition_number
        print ('Número condicional asociado a esta multicolinealidad es:',numero_condicional)
        if numero_condicional <= umbral_condicional:   #Si tiene un numero condicional mayor, entonces bajamos el umbral que selecciona variables
            print('Número condicional aceptable para considerar que no hay multicolinealidad')
            break
        else:
            print('Probamos a reducir el número de variables')
            model_reg = reducir_variables_aleatoriamente(X = df_x, y = df[y], umbral_condicional = umbral_condicional)
            if funcSeleccionVars == varsMasCorrelacionadas:
                minUmbral -= 0.01
            else:
                minUmbral -= 0.0001
    else:
        print('No se ha podido evitar Multicolinealidad',f'Se usa {numero_condicional}')
    #Evaluamos el primer modelo
    p_values = model_reg.pvalues
    variables_con_p_valor_alto = p_values[p_values > 0.1].index.tolist()
    if len(variables_con_p_valor_alto) > 0:
        print("Eliminando variables con p-valor mayor a 0.1:", variables_con_p_valor_alto)
        if any("const" in variable or "_hat" in variable for variable in variables_con_p_valor_alto):
            variables_con_p_valor_alto = [variable for variable in variables_con_p_valor_alto if "const" not in variable]
            if variables_con_p_valor_alto == []:
                model_reg = regresion(X = df_x, y = df[y], const = 0) 
            else:
                df_x = df_x.drop(variables_con_p_valor_alto, axis = 1)
                model_reg = regresion(X = df_x, y = df[y], const = 0)    
        else:
            df_x = df_x.drop(variables_con_p_valor_alto, axis = 1)
            model_reg = regresion(X = df_x, y = df[y], const = 1)
    if summary == True:
        return tree, model_reg.summary()
    else:
        return tree, model_reg    


