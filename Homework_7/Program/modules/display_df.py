import pandas as pd

import streamlit as st
import io
import contextlib


def display_gen_df_info(df_arg):
    """
    Function to display general information about the df_arg dataframe on the Streamlit page: df_arg.head() and
    df_arg.info()
    :param df_arg: df_arg - dataframe, information about which should be displayed
    :return:  None
    """
    st.dataframe(df_arg.head(), use_container_width=True)

    # Display the df.info() output with formatting
    # Capture the info() output as a string
    # Redirect stdout to capture printed information
    info_output = io.StringIO()
    with contextlib.redirect_stdout(info_output):
        df_arg.info()

    # Display the captured information
    st.write("##### DataFrame Info")
    st.code(info_output.getvalue())