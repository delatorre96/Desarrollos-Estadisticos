import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def enalzarVariables(dataframe, vars_enlazar, str_nueva_var, porc_varExplicada):
    """
    This function performs variable linking using Principal Component Analysis (PCA).
    It creates a new composite variable that captures a significant amount of explained variance
    from the original set of variables.

    Parameters:
    - dataframe (pandas dataframe): The pandas dataframe containing all variables.
    - vars_enlazar (list): List of variables from the dataframe to be linked.
    - str_nueva_var (string): Name of the new linked variable.
    - porc_varExplicada (float): Minimum percentage of explained variance desired.

    Returns:
    pandas dataframe: The dataframe with the linked variable in a new column.
    """
    
    # Select the variables to be linked
    df_vars_enlazar = dataframe[vars_enlazar]
    
    # Standardize the data
    scaler = StandardScaler()
    df_vars_enlazar = scaler.fit_transform(df_vars_enlazar)
    
    # Perform PCA to capture the desired percentage of explained variance
    pca = PCA(porc_varExplicada)
    componentes_principales = pca.fit_transform(df_vars_enlazar)
    
    # Inverse transform to get the original scale of the components
    componentes_principales = scaler.inverse_transform(componentes_principales)
    pca_vars_enlazadas = pd.DataFrame(componentes_principales)
    
    # Assign the linked variable to the dataframe and drop the original variables
    dataframe[str_nueva_var] = pca_vars_enlazadas
    dataframe = dataframe.drop(vars_enlazar, axis=1)
    
    return dataframe

def busquedaEnlazamiento(dataframe, porcMinCorrelacion):
    """
    This function searches for pairs of variables that have high correlation to be linked together.

    Parameters:
    - dataframe (pandas dataframe): The pandas dataframe containing all variables.
    - porcMinCorrelacion (float): Minimum percentage of correlation.

    Returns:
    list: List of variable pairs that have high correlation.
    """
    # Compute the correlation matrix
    matriz_correlacion = dataframe.corr()
    
    # Identify pairs of variables with correlation above porcMinCorrelacion
    variables_correlacionadas = (matriz_correlacion.abs() > porcMinCorrelacion) & (matriz_correlacion.abs() < 1)
    vars = []
    
    # Collect variable pairs into a list
    for variable in variables_correlacionadas.columns:
        correlacionadas = variables_correlacionadas[variable][variables_correlacionadas[variable]].index.tolist()
        if len(correlacionadas) > 0:
            vars.append([variable, correlacionadas[0]])
    
    # Remove duplicates and return unique pairs
    parejas_unicas_set = set()
    for pareja in vars:
        pareja_ordenada = tuple(sorted(pareja))  # Ensure order for uniqueness
        parejas_unicas_set.add(pareja_ordenada)
    
    parejas_unicas = [list(pareja) for pareja in parejas_unicas_set]

    return parejas_unicas
