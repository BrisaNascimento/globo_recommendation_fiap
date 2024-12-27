from typing import List

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from globo_recommendation_fiap.utils.preprocessing_utils import (
    drop_columns,
    merge_dfs,
    split_multivalued_df,
)


def main(
    split_columns: list,
    columns_to_drop: list,
    dfs: List[pd.DataFrame],
    df_a_key: str,
    df_b_key: str,
) -> None:
    split_cols = split_columns
    drop_cols = columns_to_drop
    a_key = df_a_key
    b_key = df_b_key

    pipeline = Pipeline(
        steps=[
            (
                'drop_columns',
                FunctionTransformer(
                    drop_columns, kw_args={'columns_to_drop': drop_cols}
                ),
            ),
            (
                'split_multivalued_df',
                FunctionTransformer(
                    split_multivalued_df, kw_args={'split_columns': split_cols}
                ),
            ),
            (
                'merge',
                FunctionTransformer(
                    merge_dfs,
                    kw_args={
                        'df_b': dfs[1],
                        'df_a_key': a_key,
                        'df_b_key': b_key,
                    },
                ),
            ),
        ]
    )

    return pipeline.transform(dfs[0])


if __name__ == '__main__':
    """
        To run this script, execute:
        poetry run python -m globo_recommendation_fiap.data.data_pipeline
    """

    df_a = pd.read_parquet(
        'globo_recommendation_fiap/data/challenge_files/local/treino.parquet'
    )
    df_b = pd.read_parquet(
        'globo_recommendation_fiap/data/challenge_files/local/itens.parquet'
    )
    split_columns = [
        'history',
        'timestampHistory',
        'numberOfClicksHistory',
        'timeOnPageHistory',
        'scrollPercentageHistory',
        'pageVisitsCountHistory',
    ]
    columns_to_drop = ['userType', 'historySize', 'timestampHistory_new']
    df_a_key = 'history'
    df_b_key = 'page'
    df = main(
        dfs=[df_a, df_b],
        df_a_key=df_a_key,
        df_b_key=df_b_key,
        columns_to_drop=columns_to_drop,
        split_columns=split_columns,
    )
    print(df.head())
