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


def get_stability_graph(time_diffs_arg, zoom_near_origin=True, title=None):
    """A function for plotting the stability graph of the data collection period
    :param time_diffs_arg: an array that contains the value of the differences between two consecutive time intervals
    for the studied dataframe
    :param zoom_near_origin: whether to scale the graph as a smaller subgraph
    :param title: the name of the graph
    :return: matplotlib.figure object, which is a stability plot of the data collection period
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title(title)
    ax.set_ylabel('measurement time, s')
    sns.lineplot(x=range(len(time_diffs_arg)),
                 y=time_diffs_arg)

    if zoom_near_origin:
        inset_axes = fig.add_axes([0.17, 0.35, 0.10, 0.25])
        inset_axes.plot(range(len(time_diffs_arg)),
                        time_diffs_arg)
        inset_axes.set_title('Zoom near origin')
        inset_axes.set_xlim(10, 20)
        inset_axes.set_ylim(0.01, 0.04)

    return fig


