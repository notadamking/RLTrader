import tensorflow as tf


def transform(df, transform_fn, columns=None):
    transformed_df = df.copy()

    if columns is None:
        transformed_df = transform_fn(transformed_df)

    for column in columns:
        transformed_df[column] = transform_fn(transformed_df[column])

    transformed_df = transformed_df.fillna(method='bfill')

    return transformed_df


def max_min_normalize(df, columns):
    def transform_fn(transform_df):
        return (transform_df - transform_df.min()) / (transform_df.max() - transform_df.min())

    return transform(df, transform_fn, columns)


def difference(df, columns):
    def transform_fn(transform_df):
        return transform_df - transform_df.shift(1)

    return transform(df, transform_fn, columns)


def log_and_difference(df, columns):
    def transform_fn(transform_df):
        transform_df.loc[transform_df == 0] = 1E-10
        return tf.log(transform_df) - tf.log(transform_df.shift(1))

    return transform(df, transform_fn, columns)
