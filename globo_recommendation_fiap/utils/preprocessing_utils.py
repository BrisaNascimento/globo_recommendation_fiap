import pandas as pd


def split_multivalued_df(
    df: pd.DataFrame, split_columns: list
) -> pd.DataFrame:
    df[split_columns] = df[split_columns].apply(lambda col: col.str.split(','))
    expanded_df = df.explode(split_columns, ignore_index=True)
    return expanded_df


def drop_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    dropped_df = df.drop(columns=columns_to_drop, axis=1)
    return dropped_df


def merge_dfs(
    df_a: pd.DataFrame, df_b: pd.DataFrame, df_a_key: str, df_b_key: str
) -> pd.DataFrame:
    merged_df = pd.merge(
        df_a, df_b, left_on=df_a_key, right_on=df_b_key, how='inner'
    )
    return merged_df
