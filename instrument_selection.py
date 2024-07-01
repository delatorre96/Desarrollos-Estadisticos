import statsmodels.api as sm
from variable_selection import varsMasCorrelacionadas, varsMasImportantesArbol
from regression_analysis import residuo


def instrumentos_exogenos(df, var_ind, var_dep, minUmbral, funcSeleccionVars):
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
        minCorr = 1
        while minCorr > 0:
            if funcSeleccionVars == varsMasCorrelacionadas:
                instrumentos_lista = [i for i in varsMasCorrelacionadas(df=df, varDependiente=x_i, minCorr=minUmbral) if i not in var_ind and i not in var_dep]
            if funcSeleccionVars == varsMasImportantesArbol:
                instrumentos_lista = [i for i in varsMasImportantesArbol(df=df, varDependiente=x_i, minUmbral=minUmbral) if i not in var_ind and i not in var_dep]
            if len(instrumentos_lista) >= len(var_ind):
                v = residuo(varsIndependientes=instrumentos_lista, varDependiente=x_i, df=df)
                x_df = df.copy()
                x_df = x_df[var_ind]
                x_df['v'] = v
                x_df = sm.add_constant(x_df)
                model = sm.OLS(df[var_dep], x_df).fit()
                p_values = dict(model.pvalues)
                if alfa > p_values['v']:
                    instrumentos_exogenos.update({x_i: True})
                    break
                else:
                    minCorr -= 0.01
            else:
                minCorr -= 0.01
        instrumentos_exogenos.update({x_i: False})
    return instrumentos_exogenos

def vars_endogena(df, var_ind, var_dep, minUmbral, funcSeleccionVars):
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
        minCorr = 1
        while minCorr > 0:
            if funcSeleccionVars == varsMasCorrelacionadas:
                instrumentos_lista = [i for i in varsMasCorrelacionadas(df=df, varDependiente=x_i, minCorr=minUmbral) if i not in var_ind and i not in var_dep]
            if funcSeleccionVars == varsMasImportantesArbol:
                instrumentos_lista = [i for i in varsMasImportantesArbol(df=df, varDependiente=x_i, minUmbral=minUmbral) if i not in var_ind and i not in var_dep]
            if len(instrumentos_lista) >= len(var_ind):
                break
            else:
                minCorr -= 0.01
        v = residuo(varsIndependientes=instrumentos_lista, varDependiente=x_i, df=df)
        x_df = df.copy()
        x_df = x_df[var_ind]
        x_df['v'] = v
        x_df = sm.add_constant(x_df)
        model = sm.OLS(df[var_dep], x_df).fit()
        p_values = dict(model.pvalues)
        if alfa > p_values['v']:
            instrumentos.update({x_i: instrumentos_lista})
    return instrumentos

