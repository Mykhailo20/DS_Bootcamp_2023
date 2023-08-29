import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def get_avg_period(df_arg, column_name_arg):
    """Function for finding the average value of the measurement period
    :param df_arg: researched dataframe
    :param column_name_arg: the name of the column that contains information about the time of the measurements
    (e.g., the 'time' column)
    :return: the average value of the measurement period
    """
    return df_arg[column_name_arg].diff().mean()


def get_avg_frequency(df_arg, column_name_arg):
    """Function for finding the average value of the measurement frequency
        :param df_arg: researched dataframe
        :param column_name_arg: the name of the column that contains information about the time of the measurements
        (e.g., the 'time' column)
        :return: the average value of the measurement frequency
        """
    time_diffs = df_arg[column_name_arg].diff()
    return 1.0 / time_diffs.mean()


@st.cache_data
def get_measurement_time_df(df_arg, column_name_arg='time'):
    """ A function for building a dataframe that contains information about the time of measurement of each reading of the device
    Args:
        1) df - a dataframe that contains a time column
        2) column_name - the name of the column containing time data (in seconds)
    Returns:
        time_measurement_df - dataframe that contains information about the time of measurement of each reading of the device
    """
    period_dict = {'start_time': [], 'end_time': [], 'measurement_time': []}
    prev_time = None
    curr_time = None
    for index, row in df_arg.iterrows():
        if index == len(df_arg):
            break

        prev_time = curr_time
        curr_time = row[column_name_arg]

        if prev_time != None:
            period_dict['start_time'].append(prev_time)
            period_dict['end_time'].append(curr_time)
            period_dict['measurement_time'].append(curr_time - prev_time)

    time_measurement_df = pd.DataFrame.from_dict(period_dict)
    return time_measurement_df


def get_stability_graph(time_measurement_df_arg, column_name_arg='measurement_time', zoom_near_origin=True, title=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title(title)
    ax.set_ylabel('measurement time, s')
    sns.lineplot(x=range(len(time_measurement_df_arg[column_name_arg])),
                 y=time_measurement_df_arg[column_name_arg])

    if zoom_near_origin:
        inset_axes = fig.add_axes([0.17, 0.35, 0.10, 0.25])
        inset_axes.plot(range(len(time_measurement_df_arg[column_name_arg])),
                        time_measurement_df_arg[column_name_arg])
        inset_axes.set_title('Zoom near origin')
        inset_axes.set_xlim(10, 20)
        inset_axes.set_ylim(0.01, 0.04)

    return fig


