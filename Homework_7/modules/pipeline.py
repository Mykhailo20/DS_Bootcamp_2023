import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from modules import feature_engineering, model_training


def perform_feature_engineering(df_arg):
    y_train = df_arg['activity'].values
    functions_list = [
        lambda x: x.mean(),  # mean
        lambda x: x.std(),  # std deviation
        lambda x: np.mean(np.absolute(x - np.mean(x))),  # avg absolute diff
        lambda x: x.min(),  # min
        lambda x: x.max(),  # max
        lambda x: x.max() - x.min(),  # range = max-min diff
        lambda x: np.median(x),  # median
        lambda x: np.percentile(x, 75) - np.percentile(x, 25),  # interquartile range
        lambda x: np.sum(x < 0),  # negative count
        lambda x: np.sum(x > 0),  # positive count
        lambda x: stats.skew(x),  # skewness = assymetry
        lambda x: stats.kurtosis(x)  # kurtosis
    ]
    result_columns = ['mean', 'std', 'aad', 'min', 'max', 'range', 'median', 'iqr', 'neg_count', 'pos_count',
                      'assymetry', 'kurtosis']
    df_arg = feature_engineering.get_statistical_measures_df(windowed_data_df=df_arg,
                                                             functions=functions_list,
                                                             data_df_columns=['accX', 'accY', 'accZ', 'gyrZ'],
                                                             result_df_columns=result_columns)
    df_arg['activity'] = y_train
    return df_arg


def model_training_data_preparation(df_arg):
    # Convert string labels to int
    activity_dict = {'Squat': 0, 'Leg land': 1, 'Walk': 2, 'Lateral squat slide': 3, 'Jogging': 4}
    df_arg['activity_number'] = df_arg['activity'].apply(lambda x: activity_dict[x])

    X_train, y_train, X_valid, y_valid = model_training.split_train_data(train_df_arg=df_arg)
    y_train = model_training.prepare_target_features(y_arg=y_train, one_hot_encoding=True)
    y_valid = model_training.prepare_target_features(y_arg=y_valid, one_hot_encoding=True)
    # Scale feature vectors
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    return X_train, y_train, X_valid, y_valid