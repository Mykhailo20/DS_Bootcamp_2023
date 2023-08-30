import numpy as np
from scipy import stats
import streamlit as st

from modules import get_data, display_df, display_results, frequency_stability, data_filtering, \
    exploratory_data_analysis, windowing, feature_engineering, model_training


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


def main():
    display_df.set_page_config(page_title_arg="Physical Activity Recognition", layout_arg="centered")
    streamlit_train_df_filename = 'data/Train/Train_activities_1_2023-08-23.csv'

    df = get_data.read_from_file(filename=streamlit_train_df_filename)

    # Exploring measurement period and frequency stability
    df = df[df['time'].diff() <= frequency_stability.get_avg_period(df, 'time') * 1.5]

    # Data Filtering
    df[['accX_filtered', 'accY_filtered', 'accZ_filtered', 'gyrX_filtered', 'gyrY_filtered', 'gyrZ_filtered']] \
        = data_filtering.median_filter_data(df_arg=df,
                                            filter_columns_arg=['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ'],
                                            window_size_arg=10)

    # Exploratory Data Analysis
    df = df[df['activity'] != 'No activity']

    # Perform undersampling to get a balanced dataframe
    df = exploratory_data_analysis.get_undersampled_df(df_arg=df, column_name_arg='activity')

    # Build a correlation matrix and remove certain axes of the accelerometer or gyroscope
    sel_columns = ['accX_filtered', 'accY_filtered', 'accZ_filtered', 'gyrX_filtered', 'gyrY_filtered',
                   'gyrZ_filtered']

    # Calculate the correlation matrix for the selected columns
    corr_matrix = df[sel_columns].corr()
    important_columns = ['accX_filtered', 'accY_filtered', 'accZ_filtered']
    discard_columns = exploratory_data_analysis.get_discard_columns(corr_matrix_arg=corr_matrix,
                                                                    important_columns_arg=important_columns,
                                                                    df_arg=df)
    sel_columns = [col for col in sel_columns if col not in discard_columns]
    sel_columns.append('activity')
    filtered_df = df[['time'] + sel_columns].copy()

    # Windowing
    windowed_df = windowing.get_windowed_df(df_arg=filtered_df, window_duration_arg=2)

    # Feature Engineering
    windowed_df = perform_feature_engineering(df_arg=windowed_df)

    # Model training
    # Convert string labels to int
    activity_dict = {'Squat': 0, 'Leg land': 1, 'Walk': 2, 'Lateral squat slide': 3, 'Jogging': 4}
    windowed_df['activity_number'] = windowed_df['activity'].apply(lambda x: activity_dict[x])

    X_train, y_train, X_valid, y_valid = model_training.split_train_data(train_df_arg=windowed_df)

    # Display results on the Streamlit page
    with st.expander("General information"):
        display_df.display_gen_df_info(df_arg=df)
    with st.expander("Exploring measurement period and frequency stability"):
        if st.checkbox("Display general information"):
            display_results.show_freq_info(df_arg=df, column_name_arg='time')
        if st.checkbox("Display the results of frequency stabilization"):
            display_results.show_freq_stability(df_arg=df, column_name_arg='time')
    with st.expander("Data Filtering"):
        display_results.show_filtering_results(df_arg=df)
    with st.expander("Exploratory Data Analysis"):
        display_results.show_data_analysis_results(df_arg=filtered_df,
                                                   corr_matrix_arg=corr_matrix,
                                                   discard_columns_arg=discard_columns)
    with st.expander("Data Transformation"):
        st.pyplot(windowing.get_pie_charts(first_df=filtered_df, second_df=windowed_df, column='activity',
                                           first_chart_title="Filtered DataFrame",
                                           second_chart_title="Windowed DataFrame"))
    with st.expander("Feature Engineering"):
        display_df.display_df_info(df_arg=windowed_df, title_arg="##### features_df info")
    with st.expander("Model Training"):
        display_results.show_train_spliting_results(X_train_arg=X_train, y_train_arg=y_train,
                                                    X_valid_arg=X_valid, y_valid_arg=y_valid)


if __name__ == '__main__':
    main()
