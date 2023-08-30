from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def get_windowed_df(df_arg, window_duration_arg):
    # Calculate the number of data points within a 2-second window
    sampling_frequency = 1.0 / df_arg['time'].diff().mean()  # Hz
    window_size = ceil(sampling_frequency * window_duration_arg)
    step_size = window_size // 2

    # Create a list to store the windowed dataframes
    windowed_dfs = []

    windowed_dict = {'accX': [], 'accY': [], 'accZ': [], 'gyrZ': [], 'activity': []}
    # Divide the entire dataframe into 2-second windows
    for i in range(0, len(df_arg), step_size):
        window_df = df_arg.iloc[i:i + window_size]

        windowed_dict['accX'].append(window_df['accX_filtered'].values)
        windowed_dict['accY'].append(window_df['accY_filtered'].values)
        windowed_dict['accZ'].append(window_df['accZ_filtered'].values)

        windowed_dict['gyrZ'].append(window_df['gyrZ_filtered'].values)

        # Determine the most frequent activity in the window
        most_frequent_activity = window_df['activity'].value_counts().idxmax()
        # Assign the most frequent activity to all rows in the window
        windowed_dict['activity'].append(most_frequent_activity)

        windowed_dfs.append(window_df)

    return pd.DataFrame.from_dict(windowed_dict)


def get_pie_charts(first_df, second_df, column, first_chart_title='First DataFrame',
                   second_chart_title='Second DataFrame', filename=None):
    """Function to get a graph displaying the ratio of the column contents between two data frames in the form of pie
    charts
    :param first_df: the first dataframe that contains the required data
    :param second_df: the second dataframe, which contains the required data
    :param column: the name of the dataframe column whose percentage values are to be found
    :param first_chart_title: the title of the first pie chart
    :param second_chart_title: the title of the second pie chart
    :param filename: the relative path where the file will be saved (with the file name, the file extension is not
    required) or just the filename
    :return: matplotlib.figure object
    """
    # Calculate the percentage of each activity in first_df
    activity_percentages_first_df = first_df[column].value_counts(normalize=True) * 100

    # Calculate the percentage of each activity in second_df
    activity_percentages_second_df = second_df[column].value_counts(normalize=True) * 100

    # Create subplots for pie charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot pie chart for df
    sns.set_palette("Set3")
    axes[0].pie(activity_percentages_first_df, labels=activity_percentages_first_df.index, autopct='%1.1f%%', startangle=140)
    axes[0].set_title(first_chart_title)

    # Plot pie chart for windowed_df
    sns.set_palette("Set3")
    axes[1].pie(activity_percentages_second_df, labels=activity_percentages_first_df.index, autopct='%1.1f%%', startangle=140)
    axes[1].set_title(second_chart_title)

    # Adjust layout
    plt.tight_layout()
    if filename:
        plt.savefig(f'{filename}.png', bbox_inches='tight')

    return fig
