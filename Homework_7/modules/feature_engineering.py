import pandas as pd


def get_statistical_measures_df(windowed_data_df, functions, data_df_columns, result_df_columns):
    """Function for creating a dataframe X_df, the columns of which correspond to the required statistical measures for
    the windows of the windowed_data_df dataframe
    :param windowed_data_df: a dataframe whose rows contain arrays of data formed as a result of windowing
    :param functions: a list of references to lambda functions that will calculate the required statistical measures
    :param data_df_columns: a list of column names of the windowed_data_df dataframe for which to find statistical
    measures
    :param result_df_columns: a list of names of the searched statistical measures of the output dataframe
    :return: X_train dataframe
    An example of using the function:
        Let's imagine we have a windowed_data_df dataframe (a dataframe formed as a result of windowing - that is, each record in a row can be an array of records for a given window), which contains the columns 'accX', 'accY', 'accZ' (the results of measuring the readings of the accelerometer on the corresponding axes).
        Inside the function, we create an X_df dataframe to which we want to add new columns with statistical measure values.
        For example, we want to calculate the statistical mean and avg absolute diff for the accelerometer readings on all three axes (the columns 'accX', 'accY', 'accZ' of data_df), so the columns of the X_df dataframe will be named, for example, 'accX_mean', 'accY_mean', ..., 'accZ_aad'.
        These values can be calculated using the lambda functions: [
            lambda x: x.mean(),
            lambda x: np.mean(np.absolute(x - np.mean(x)))
        ]
        So, the function call will look like this:
        get_statistical_measures_df(df=data_df,
                                    functions=[
                                        lambda x: x.mean(),
                                        lambda x: np.mean(np.absolute(x - np.mean(x)))
                                    ],
                                    data_df_columns=['accX', 'accY', 'accZ'],
                                    result_df_columns=['mean', 'aad'])
    """
    X_df = pd.DataFrame()
    for [function, res_column] in zip(functions, result_df_columns):
        for data_column in data_df_columns:
            X_df[f'{data_column}_{res_column}'] = windowed_data_df[data_column].apply(function)
    return X_df