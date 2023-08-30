import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


@st.cache_data
def get_undersampled_df(df_arg, column_name_arg):
    # First, calculate the minimum number of samples across all classes
    min_samples = df_arg[column_name_arg].value_counts().min()

    # Initialize an empty DataFrame to store the undersampled data
    undersampled_df = pd.DataFrame()

    # Loop through each unique activity class and select the first min_samples for each class
    for activity_class in df_arg[column_name_arg].unique():
        class_subset = df_arg[df_arg[column_name_arg] == activity_class].iloc[:min_samples]
        undersampled_df = pd.concat([undersampled_df, class_subset])

    return undersampled_df


def get_correlation_matrix(corr_matrix_df_arg, filename=None):

    # Create a heatmap using seaborn
    fig = plt.figure(figsize=(10, 8))
    axes = fig.add_axes([0.1, 0.1, 1, 1])
    sns.heatmap(corr_matrix_df_arg, annot=True, cmap='coolwarm', center=0)
    axes.set_title('Correlation Matrix Heatmap')
    if filename:
        plt.savefig('graphs/correlation_matrix.png')

    return fig


def get_discard_columns(corr_matrix_arg, important_columns_arg, df_arg):

    columns_to_discard = set()
    for column in corr_matrix_arg.columns:
        correlated_columns = corr_matrix_arg.index[
            (corr_matrix_arg[column] > 0.5) | (corr_matrix_arg[column] < -0.5)
            ]

        for correlated_column in correlated_columns:
            if column != correlated_column:
                # Prioritize which column to keep based on your criteria
                # For example, keep the column with higher variance
                if column not in important_columns_arg:
                    columns_to_discard.add(column)
                elif (column in important_columns_arg) and (correlated_column in important_columns_arg):
                    pass
                elif (column in important_columns_arg) and (correlated_column not in important_columns_arg):
                    columns_to_discard.add(correlated_column)
                else:  # both columns are not in important_columns
                    columns_to_discard.add(
                        correlated_column if df_arg[correlated_column].var() < df_arg[column].var() else column)

    return columns_to_discard
