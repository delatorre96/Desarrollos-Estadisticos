import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def enalzarVariables(dataframe, vars_enlazar, str_nueva_var, porc_varExplicada):
    """
    Esta función enlaza un conjunto de variables en una misma que nos da una información muy similar. 
    Para ello se hace un PCA a un porcentaje dado de varianza explicada a través del cuál conjugaremos nuestra nueva variable

    Parameters:
    - dataframe (pandas dataframe): Es el data frame en pandas en donde están todas las variables.
    - vars_enlazar (padnas series): LAs variables de nuestro data frame que queremos enlazar.
    - str_nueva_var (string): Nombre de la nueva variable enlazada.
    - porc_varExplicada (float): Porcentaje mínimo de varianza explicada al que queremos 
    Returns:
    dataframe: EL data frame con la variable enlazada en una columna nueva
    """
    
    df_vars_enlazar = dataframe[vars_enlazar]
    
    scaler = StandardScaler()
    df_vars_enlazar = scaler.fit_transform(df_vars_enlazar)
    pca = PCA(porc_varExplicada)
    
    componentes_principales = pca.fit_transform(df_vars_enlazar)
    componentes_principales = scaler.inverse_transform(componentes_principales)
    pca_vars_enlazadas= pd.DataFrame(componentes_principales)
    
    dataframe[str_var]=pca_vars_enlazadas
    dataframe = dataframe.drop(df_vars_enlazar, axis = 1)
    return df_financ_deuda

def busquedaEnlazamiento(dataframe, porcMinCorrelacion):
    """
    Esta función busca parejas de variables que tengan una correlación elevada para poder ser enlazadas

    Parameters:
    - dataframe (pandas dataframe): Es el data frame en pandas en donde están todas las variables.
    - porcMinCorrelacion (float): porcentaje mínimo de correlación
    Returns:
    lista: lista de parejas de variables
    """
    matriz_correlacion = socioEco_financDeuda_sinCat.corr()
    variables_correlacionadas = (matriz_correlacion.abs() > porcMinCorrelacion) & (matriz_correlacion.abs() < 1)
    vars =  []
    for variable in variables_correlacionadas.columns:
        correlacionadas = variables_correlacionadas[variable][variables_correlacionadas[variable]].index.tolist()
        if len(correlacionadas) > 0:
            vars.append([variable, correlacionadas[0]])
    parejas_unicas_set = set()
    for pareja in vars:
        pareja_ordenada = tuple(sorted(pareja)) 
        parejas_unicas_set.add(pareja_ordenada)
    
    parejas_unicas = [list(pareja) for pareja in parejas_unicas_set]

    return parejas_unicas        
        