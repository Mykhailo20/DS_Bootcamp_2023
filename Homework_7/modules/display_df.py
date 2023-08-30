import pandas as pd

import streamlit as st
import io
import contextlib


def set_page_config(page_title_arg, layout_arg):
    """Function to set Streamlit page parameters
    :param page_title_arg: The page title, shown in the browser tab
    :param layout_arg: how the page content should be laid out ("centered" or "wide")
    :return: None
    """
    st.set_page_config(page_title=page_title_arg, layout=layout_arg)


def display_gen_df_info(df_arg, title_arg="##### DataFrame Info"):
    """Function to display general information about the df_arg dataframe on the Streamlit page: df_arg.head() and
    df_arg.info()
    :param df_arg: dataframe, information about which should be displayed
    :param title_arg: title that will be placed above the dataframe information
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
    st.write(title_arg)
    st.code(info_output.getvalue())


def display_df_info(df_arg, title_arg="##### DataFrame Info"):
    """Function to display df.info() on a Streamlit page
    :param df_arg: dataframe, information about which should be displayed
    :param title_arg: title that will be placed above the dataframe information
    :return: None
    """
    # Capture the info() output as a string
    # Redirect stdout to capture printed information
    info_output = io.StringIO()
    with contextlib.redirect_stdout(info_output):
        df_arg.info()

    # Display the captured information
    st.write(title_arg)
    st.code(info_output.getvalue())
