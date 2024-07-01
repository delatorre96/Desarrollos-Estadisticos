import pandas as pd

def columnasCategoricas(df):
    """
    Identifies categorical columns in the dataframe.
    
    Parameters:
    df (pandas.DataFrame): The input dataframe.
    
    Returns:
    list: A list of column names that are of object type.
    """
    columnas_categoricas = []
    for columna in df.columns:
        if df[columna].dtype == 'O':
            columnas_categoricas.append(columna)
    return columnas_categoricas

def vars_interaccion(df):
    """
    Creates interaction variables based on pairwise multiplication of columns.
    
    Parameters:
    df (pandas.DataFrame): The input dataframe.
    
    Returns:
    pandas.DataFrame: Dataframe with interaction variables appended.
    """
    vars_interaccion = {}
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:
                result_col_name = f"{col1}#{col2}"
                vars_interaccion.update({result_col_name: df[col1] * df[col2]})
            else:
                result_col_name = f"{col1}^2"
                vars_interaccion.update({result_col_name: df[col1] * df[col2]})
    df_vars_interaccion = pd.DataFrame(vars_interaccion)
    result_df = pd.concat([df, df_vars_interaccion], axis=1)
    return result_df
