import pandas as pd
import streamlit as st


@st.cache_data
def read_from_file(filename):
    """Function for reading a dataset from a file
    :param filename: The filename for the training dataset
    :return: A dataset read from a file
    """
    print('read_from_file')
    df_local = pd.read_csv(filename)
    return df_local
