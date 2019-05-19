import numpy as np


def difference(df, columns):
    transformed_df = df.copy()

    for column in columns:
        transformed_df[column] = transformed_df[column] - \
            transformed_df[column].shift(1)

    transformed_df = transformed_df.fillna(method='bfill')

    return transformed_df


def log_and_difference(df, columns):
    transformed_df = df.copy()

    for column in columns:
        transformed_df.loc[df[column] == 0] = 1E-10
        transformed_df[column] = np.log(
            transformed_df[column]) - np.log(transformed_df[column]).shift(1)

    transformed_df = transformed_df.fillna(method='bfill')

    return transformed_df
