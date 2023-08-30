import numpy as np
import pandas as pd

from keras.utils import to_categorical


def split_train_data(train_df_arg, training_part=0.8):
    X_train = pd.DataFrame()
    y_train = []

    X_valid = pd.DataFrame()
    y_valid = []

    counter = 0

    for activity in train_df_arg['activity'].unique():
        activity_data = train_df_arg[train_df_arg['activity'] == activity].copy()
        activity_data.reset_index(inplace=True)
        activity_data.drop('index', axis=1, inplace=True)
        split_index = int(training_part * len(activity_data))
        if counter != 0:
            X_train = pd.concat([X_train, activity_data[activity_data.columns[:-2]][:split_index]])
            X_valid = pd.concat([X_valid, activity_data[activity_data.columns[:-2]][split_index:]])
        else:
            X_train = activity_data[activity_data.columns[:-2]][:split_index]
            X_valid = activity_data[activity_data.columns[:-2]][split_index:]

        y_train.extend(list(activity_data['activity_number'].values[:split_index]))
        y_valid.extend(list(activity_data['activity_number'].values[split_index:]))

        counter += 1

    return [X_train, y_train, X_valid, y_valid]


def prepare_target_features(y_arg, one_hot_encoding):
    y_arg = np.array(y_arg)
    if one_hot_encoding:
        # Convert Label Encoded target data to one-hot encoded format
        y_arg = to_categorical(y_arg)
    return y_arg
