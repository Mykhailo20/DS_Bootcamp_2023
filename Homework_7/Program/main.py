from modules import get_data
from modules import display_df


def main():
    streamlit_train_df_filename = 'data/Train/Train_activities_1_2023-08-23.csv'

    df = get_data.read_from_file(filename=streamlit_train_df_filename)
    display_df.display_gen_df_info(df_arg=df)


if __name__ == '__main__':
    main()
