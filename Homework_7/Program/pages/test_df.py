import streamlit as st

from modules import get_data
from modules import display_df


def main():
    streamlit_test_df_filename = 'data/Test/Test_activities_1_2023-08-23.csv'
    df = get_data.read_from_file(filename=streamlit_test_df_filename)
    with st.expander("General information"):
        display_df.display_gen_df_info(df_arg=df)


if __name__ == '__main__':
    main()
