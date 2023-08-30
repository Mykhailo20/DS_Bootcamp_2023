import time
import pandas as pd
from memory_profiler import profile, memory_usage

from modules import frequency_stability, data_filtering, exploratory_data_analysis, windowing, pipeline


def perform_pipeline(df_arg):
    # Exploring measurement period and frequency stability
    df_arg = df_arg[df_arg['time'].diff() <= frequency_stability.get_avg_period(df_arg, 'time') * 1.5]

    # Data Filtering
    df_arg[['accX_filtered', 'accY_filtered', 'accZ_filtered', 'gyrX_filtered', 'gyrY_filtered', 'gyrZ_filtered']] \
        = data_filtering.median_filter_data(df_arg=df_arg,
                                            filter_columns_arg=['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ'],
                                            window_size_arg=10)

    # Exploratory Data Analysis
    df_arg = df_arg[df_arg['activity'] != 'No activity']

    # Perform undersampling to get a balanced dataframe
    df_arg = exploratory_data_analysis.get_undersampled_df(df_arg=df_arg, column_name_arg='activity')

    # Build a correlation matrix and remove certain axes of the accelerometer or gyroscope
    sel_columns = ['accX_filtered', 'accY_filtered', 'accZ_filtered', 'gyrX_filtered', 'gyrY_filtered',
                   'gyrZ_filtered']

    # Calculate the correlation matrix for the selected columns
    corr_matrix = df_arg[sel_columns].corr()
    important_columns = ['accX_filtered', 'accY_filtered', 'accZ_filtered']
    discard_columns = exploratory_data_analysis.get_discard_columns(corr_matrix_arg=corr_matrix,
                                                                    important_columns_arg=important_columns,
                                                                    df_arg=df_arg)
    sel_columns = [col for col in sel_columns if col not in discard_columns]
    sel_columns.append('activity')
    filtered_df = df_arg[['time'] + sel_columns].copy()

    # Windowing
    windowed_df = windowing.get_windowed_df(df_arg=filtered_df, window_duration_arg=2)

    # Feature Engineering
    windowed_df = pipeline.perform_feature_engineering(df_arg=windowed_df)

    # Model training
    return pipeline.model_training_data_preparation(df_arg=windowed_df)


@profile
def perform_pipeline_memory(df_arg):
    # Exploring measurement period and frequency stability
    df_arg = df_arg[df_arg['time'].diff() <= frequency_stability.get_avg_period(df_arg, 'time') * 1.5]

    # Data Filtering
    df_arg[['accX_filtered', 'accY_filtered', 'accZ_filtered', 'gyrX_filtered', 'gyrY_filtered', 'gyrZ_filtered']] \
        = data_filtering.median_filter_data(df_arg=df_arg,
                                            filter_columns_arg=['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ'],
                                            window_size_arg=10)

    # Exploratory Data Analysis
    df_arg = df_arg[df_arg['activity'] != 'No activity']

    # Perform undersampling to get a balanced dataframe
    df_arg = exploratory_data_analysis.get_undersampled_df(df_arg=df_arg, column_name_arg='activity')

    # Build a correlation matrix and remove certain axes of the accelerometer or gyroscope
    sel_columns = ['accX_filtered', 'accY_filtered', 'accZ_filtered', 'gyrX_filtered', 'gyrY_filtered',
                   'gyrZ_filtered']

    # Calculate the correlation matrix for the selected columns
    corr_matrix = df_arg[sel_columns].corr()
    important_columns = ['accX_filtered', 'accY_filtered', 'accZ_filtered']
    discard_columns = exploratory_data_analysis.get_discard_columns(corr_matrix_arg=corr_matrix,
                                                                    important_columns_arg=important_columns,
                                                                    df_arg=df_arg)
    sel_columns = [col for col in sel_columns if col not in discard_columns]
    sel_columns.append('activity')
    filtered_df = df_arg[['time'] + sel_columns].copy()

    # Windowing
    windowed_df = windowing.get_windowed_df(df_arg=filtered_df, window_duration_arg=2)

    # Feature Engineering
    windowed_df = pipeline.perform_feature_engineering(df_arg=windowed_df)

    # Model training
    return pipeline.model_training_data_preparation(df_arg=windowed_df)


def get_time_usage(df_arg):
    # Measure start time
    start_time = time.time()

    # Call the pipeline function
    X_train, y_train, X_valid, y_valid = perform_pipeline(df_arg=df_arg)

    # Measure end time
    end_time = time.time()
    total_time = end_time - start_time
    return total_time


def get_memory_usage(df_arg):
    # Measure memory before running the pipeline
    mem_before = memory_usage()[0]

    # Call the pipeline function
    X_train, y_train, X_valid, y_valid = perform_pipeline_memory(df_arg=df_arg)

    # Measure memory after running the pipeline
    mem_after = memory_usage()[0]
    mem_used = mem_after - mem_before
    return mem_used


def main():
    df = pd.read_csv('data/Train/Train_activities_1_2023-08-23.csv')
    number_of_experiments = 5

    # Investigate pipeline execution time
    time_list = []
    for i in range(number_of_experiments):
        temp_time = get_time_usage(df_arg=df)
        time_list.append(temp_time)
        print(f"{i+1}) time = {temp_time:0.3f} seconds")
    print(f"average execution time = {sum(time_list) / len(time_list): .3f} seconds")

    # Investigate pipeline memory usage
    memory_list = []
    for i in range(number_of_experiments):
        temp_memory = get_memory_usage(df_arg=df)
        memory_list.append(temp_memory)
        print(f"Memory used: {temp_memory:.3f} MiB")

    for i in range(len(memory_list)):
        print(f"{i+1}) Memory used: {memory_list[i]:.3f} MiB")
    print(f"average memory usage = {sum(memory_list) / len(memory_list): .3f} MiB")


if __name__ == "__main__":
    main()
