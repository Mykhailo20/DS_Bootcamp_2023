import streamlit as st

from modules import frequency_stability


def show_freq_info(df_arg, column_name_arg):
    """Function to display general information about the measurement frequency on the Streamlit page
    :param df_arg: researched dataframe
    :param column_name_arg: the name of the column that contains information about the time of the measurements
    (e.g., the 'time' column)
    :return: None
    """
    st.info(f"Measurement time = {df_arg.iloc[-1][column_name_arg] - df_arg.iloc[0][column_name_arg]} s")
    st.info(f"Number of measurements (number of rows in the data set) = {len(df_arg)}")
    st.info(f"Average measurement period = {frequency_stability.get_avg_period(df_arg, column_name_arg):.3f} s")
    st.info(f"Average frequency of measurement = "
            f"{frequency_stability.get_avg_frequency(df_arg, column_name_arg):.3f} Hz")


