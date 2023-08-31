import streamlit as st

from modules import get_data, display_df, display_results, frequency_stability, data_filtering, \
    exploratory_data_analysis, windowing, pipeline


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
    windowed_df = pipeline.perform_feature_engineering(df_arg=windowed_df)

    # Model Training Preparation
    X_train, y_train, X_valid, y_valid = pipeline.model_training_data_preparation(df_arg=windowed_df)

    # Display results on the Streamlit page
    st.subheader("Running a pipeline on training data")
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
                                           first_chart_title="Original DataFrame",
                                           second_chart_title="Windowed DataFrame"))
    with st.expander("Feature Engineering"):
        display_df.display_df_info(df_arg=windowed_df, title_arg="##### features_df info")
        st.info(f"Number of features = {len(windowed_df.columns) - 2}")
    with st.expander("Data Preparation for Model Training"):
        st.write(f"X_train[0] = {X_train[0]}")
        st.write(f"y_train[0] = {y_train[0]}")

        st.write(f"len(X_train) = {len(X_train)}")
        st.write(f"len(y_train) = {len(y_train)}")

        st.write(f"len(X_valid) = {len(X_valid)}")
        st.write(f"len(y_valid) = {len(y_valid)}")


if __name__ == '__main__':
    main()
