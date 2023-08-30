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
    :return: a dataframe that contains the filtered column values
    """
    df = pd.DataFrame()
    for column in filter_columns_arg:
        # Apply median filtering to column data
        df[f'{column}_filtered'] = df_arg[column].rolling(window=window_size_arg, center=True, min_periods=1).median()
    return df


def get_raw_filtered_data_zoom_graph(df, x, y, x_lims, y_lims, title=None, x_label=None, y_label=None, filename=None, zoom_axes=None):
    """Function to get a graph that displays the raw and filtered data on a single graph with scaling to better display the filtering
    :param df: a dataframe containing the results of the device measurement
    :param x: the name of the column of the dataframe that contains the data for the OX axis of the graph
    :param y: the list containing the column names of the data frame corresponding to the raw and filtered data,
    respectively, on a given axis (OX, OY, or OZ)
    :param x_lims: the list of limits along the OX axis for zooming ([zoom_xmin; zoom_xmax])
    :param y_lims: the list of limits along the OY axis for zooming ([zoom_ymin; zoom_ymax])
    :param title: the title of the graph
    :param x_label: the name of the OX axis of the graph
    :param y_label: the name of the OY axis of the graph
    :param filename:  the relative path where the file will be saved (with the file name, the file extension is not
    required) or just the filename
    :param zoom_axes: the list containing the placement coordinates (x, y) and dimensions (width, height) of the smaller graph that contains the zoomed-in image for the specified limits (x_lims and y_lims)
    zoom_axes = [x, y, width, height]
    :return: matplotlib.figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df[x], df[y[0]], label='Raw Data', color='green')
    ax.plot(df[x], df[y[1]], label='Filtered_data', color='orange')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    # inset
    if zoom_axes:
        inset_ax = fig.add_axes(zoom_axes)
    else:
        inset_ax = fig.add_axes([0.25, 0.20, 0.20, 0.20])
    inset_ax.plot(df[x], df[y[0]], label='Raw Data', color='green')
    inset_ax.plot(df[x], df[y[1]], label='Filtered_data', color='orange')
    inset_ax.set_xlim(x_lims[0], x_lims[1])
    inset_ax.set_ylim(y_lims[0], y_lims[1])
    inset_ax.set_title('zoom near origin')
    if filename:
        plt.savefig(f"{filename}.png", bbox_inches='tight')

    return fig