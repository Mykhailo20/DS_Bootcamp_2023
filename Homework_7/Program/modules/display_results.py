import streamlit as st

from modules import frequency_stability
from modules import display_df


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


def show_freq_stability(time_measurement_df_arg, df_arg, avg_period_arg):
    """Function for displaying the results of the 'Exploring measurement period and frequency stability' stage
    :param time_measurement_df_arg: a dataframe that contains information about the measurement period
    :param df_arg: the original dataframe, which contains all the necessary information
    :param avg_period_arg: average data collection period
    :return: None
    """
    st.write("##### time_measurement_df")
    display_df.display_gen_df_info(df_arg=time_measurement_df_arg)
    st.pyplot(frequency_stability.get_stability_graph(time_measurement_df_arg=time_measurement_df_arg,
                                                      column_name_arg='measurement_time',
                                                      title=f"Stability of Data Collection (Raw Data)\nAverage "
                                                            f"frequency: "
                                                            f"{1.0 / avg_period_arg:.3f} Hz"))
    display_df.display_df_info(df_arg)
    st.pyplot(frequency_stability.get_stability_graph(time_measurement_df_arg=
                                                      time_measurement_df_arg[time_measurement_df_arg['measurement_time']
                                                                              <= avg_period_arg * 1.5],
                                                      column_name_arg='measurement_time',
                                                      zoom_near_origin=False,
                                                      title=f"Stability of Data Collection (Filtered Data)\n"
                                                            f"Average frequency: "
                                                            f"{1.0 / avg_period_arg:.3f} Hz"))
