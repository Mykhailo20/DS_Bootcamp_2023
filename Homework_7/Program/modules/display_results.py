import streamlit as st

from modules import frequency_stability
from modules import data_filtering


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


def show_freq_stability(df_arg, column_name_arg='time'):
    """Function for displaying the results of the 'Exploring measurement period and frequency stability' stage
    :param df_arg: the original dataframe, which contains all the necessary information
    :param column_name_arg: the name of the column that contains information about the time of the measurements
    (e.g., the 'time' column)
    :return: None
    """
    st.write("##### Exploring measurement period and frequency stability")
    time_diffs = df_arg[column_name_arg].diff()
    avg_period = time_diffs.mean()
    st.pyplot(frequency_stability.get_stability_graph(time_diffs.values,
                                                      title=f"Stability of Data Collection (Raw Data)\nAverage "
                                                            f"frequency: "
                                                            f"{1.0 / avg_period:.3f} Hz"))
    st.pyplot(frequency_stability.get_stability_graph(time_diffs.values[time_diffs.values <= avg_period * 1.5],
                                                      zoom_near_origin=False,
                                                      title=f"Stability of Data Collection (Filtered Data)\n"
                                                            f"Average frequency: "
                                                            f"{1.0 / avg_period:.3f} Hz"))


def show_filtering_results(df_arg):
    if st.checkbox("Display raw data"):
        st.pyplot(data_filtering.get_three_axes_graph(df=df_arg,
                                                      x='time',
                                                      y=['accX', 'accY', 'accZ'],
                                                      title='Time Dependence of Linear Acceleration (Raw Data)',
                                                      x_label='Time, s',
                                                      y_label='Linear acceleration, m/s^2'))
        st.pyplot(data_filtering.get_three_axes_graph(df=df_arg,
                                                      x='time',
                                                      y=['gyrX', 'gyrY', 'gyrZ'],
                                                      title='Time Dependence of Angular Velocity (Raw Data)',
                                                      x_label='Time, s',
                                                      y_label='Angular velocity, rad/s'))
    st.pyplot(data_filtering.get_three_axes_graph(df=df_arg,
                                                  x='time',
                                                  y=['accX_filtered', 'accY_filtered', 'accZ_filtered'],
                                                  title='Time Dependence of Linear Acceleration (Filtered Data)',
                                                  x_label='Time, s',
                                                  y_label='Linear acceleration, m/s^2'))
    st.pyplot(data_filtering.get_three_axes_graph(df=df_arg,
                                                  x='time',
                                                  y=['gyrX_filtered', 'gyrY_filtered', 'gyrZ_filtered'],
                                                  title='Time Dependence of Angular Velocity (Filtered Data)',
                                                  x_label='Time, s',
                                                  y_label='Angular velocity, rad/s'))
    if st.checkbox("Consider the implications of data filtering"):
        st.pyplot(data_filtering.get_raw_filtered_data_zoom_graph(df=df_arg,
                                                                  x='time',
                                                                  y=['accX', 'accX_filtered'],
                                                                  x_lims=[14, 20],
                                                                  y_lims=[-1.5, 1.5],
                                                                  x_label='Time, s',
                                                                  y_label='Linear acceleration, m/s^2',
                                                                  title='Accelerometer OX: Raw vs Filtered',
                                                                  zoom_axes=[0.18, 0.18, 0.15, 0.15]
                                                                  ))