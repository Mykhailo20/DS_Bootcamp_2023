import pandas as pd

import streamlit as st
import io
import contextlib


@st.cache_data
def get_data(train_filename, test_filename=None):
    """Function for reading data of training and/or test datasets
    :param train_filename: The filename for the training dataset
    :param test_filename: Test dataset file name (if a separate test dataset exists)
    :return: Training dataset or a list of training and test datasets (if the test dataset is placed in a separate file)
    """
    df_local = pd.read_csv(train_filename)
    if test_filename:
        test_df_local = pd.read_csv(test_filename)
        return [df_local, test_df_local]
    return df_local


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


def main():
    streamlit_train_df_filename = 'Homework_7/Program/data/Train/Train_activities_1_2023-08-23.csv'
    streamlit_test_df_filename = 'Homework_7/Program/data/Test/Test_activities_1_2023-08-23.csv'
    df, test_df = get_data(train_filename=streamlit_train_df_filename,
                           test_filename=streamlit_test_df_filename)
    with st.expander('Train data:'):
        display_gen_df_info(df_arg=df)

    with st.expander('Test data:'):
        display_gen_df_info(df_arg=test_df)


if __name__ == '__main__':
    main()
