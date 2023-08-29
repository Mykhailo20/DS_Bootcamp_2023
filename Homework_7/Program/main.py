import streamlit as st

from modules import get_data
from modules import display_df
from modules import display_results
from modules import frequency_stability


def main():
    display_df.set_page_config(page_title_arg="Physical Activity Recognition", layout_arg="centered")
    streamlit_train_df_filename = 'data/Train/Train_activities_1_2023-08-23.csv'

    df = get_data.read_from_file(filename=streamlit_train_df_filename)

    # Exploring measurement period and frequency stability
    time_measurement_df = frequency_stability.get_measurement_time_df(df)
    avg_period = frequency_stability.get_avg_period(df, 'time')
    df = df[df['time'].diff() <= avg_period * 1.5]

    # Display results on the Streamlit page
    with st.expander("General information"):
        display_df.display_gen_df_info(df_arg=df)
    with st.expander("Exploring measurement period and frequency stability"):
        if st.checkbox("Display general information"):
            display_results.show_freq_info(df_arg=df, column_name_arg='time')
        if st.checkbox("Display the results of frequency stabilization"):
            display_results.show_freq_stability(time_measurement_df, df, avg_period)


if __name__ == '__main__':
    main()
