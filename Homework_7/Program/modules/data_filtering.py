import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_three_axes_graph(df, y, x=None, title=None, x_label=None, y_label=None, filename=None):
    """ Function for creating a graph of three axes (X, Y, Z) of the measurement results of the device
    :param df: a dataframe containing the results of the device measurement
    :param y: the list containing the dataframe column names corresponding to the OX, OY, and OZ axis measurements,
    respectively
    :param x: the name of the column of the dataframe that contains the data for the OX axis (for the three lines, this
    is the same data)
    :param title: title of the graph
    :param x_label: the name of the OX axis of the graph
    :param y_label: the name of the OY axis of the graph
    :param filename: the relative path where the file will be saved (with the file name, the file extension is not
    required) or just the filename
    :return: matplolib.figure object
    """
    fig = plt.figure(figsize=(12, 6))
    axes = fig.add_axes([0.1, 0.1, 1, 1])
    if x is None:
        axes.plot(df[y[0]], label='X')
        axes.plot(df[y[1]], label='Y')
        axes.plot(df[y[2]], label='Z')
    else:
        axes.plot(df[x], df[y[0]], label='X')
        axes.plot(df[x], df[y[1]], label='Y', color='orange')
        axes.plot(df[x], df[y[2]], label='Z', color='green')

    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.legend()
    if filename:
        plt.savefig(f'{filename}.png', bbox_inches='tight')

    return fig


def median_filter_data(df_arg, filter_columns_arg, window_size_arg):
    """Function to filter data using a median filter with a selected window of the specified columns of the transmitted
    dataframe
    :param df_arg: the dataframe whose column contents you want to filter
    :param filter_columns_arg: a list of columns whose contents should be filtered
    :param window_size_arg: window size for the median filter
    :return: None
    """
    for column in filter_columns_arg:
        # Apply median filtering to column data
        df_arg[f'{column}_filtered'] = df_arg[column].rolling(window=window_size_arg, center=True, min_periods=1).median()
