# I am comenting this test for now.
# To make this test automated we need a smaller file in the lake

# import pandas as pd

# from globo_recommendation_fiap.data.data_pipeline import main


# def test_running_pipeline_must_return_a_df_with_no_NA():
#     df_a = pd.read_parquet(
#         'globo_recommendation_fiap/data/challenge_files/local/treino.parquet'
#     )
#     df_b = pd.read_parquet(
#         'globo_recommendation_fiap/data/challenge_files/local/itens.parquet'
#     )
#     split_columns = [
#         'history',
#         'timestampHistory',
#         'numberOfClicksHistory',
#         'timeOnPageHistory',
#         'scrollPercentageHistory',
#         'pageVisitsCountHistory',
#     ]
#     columns_to_drop = ['userType', 'historySize', 'timestampHistory_new']
#     df_a_key = 'history'
#     df_b_key = 'page'
#     df = main(
#         dfs=[df_a, df_b],
#         df_a_key=df_a_key,
#         df_b_key=df_b_key,
#         columns_to_drop=columns_to_drop,
#         split_columns=split_columns,
#     )
#     assert not df.isna().any().any()
