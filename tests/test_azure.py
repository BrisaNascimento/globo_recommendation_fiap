import pandas as pd

from globo_recommendation_fiap.data.download_data import download_from_adls
from globo_recommendation_fiap.utils.lake_connector import connect_to_adls


def test_connection_to_aws():
    container_name = 'bronze'
    blob_name = 'raw/globo/teste.csv'
    blob = connect_to_adls(container_name=container_name, blob_name=blob_name)

    assert blob is not None


def test_download_data_must_return_a_dataframe():
    container_name = 'bronze'
    file_path = 'raw/globo/validacao/validacao.parquet'
    data = download_from_adls(container_name, file_path)
    MINIMUM_ROWS = 5

    assert isinstance(data, pd.DataFrame)
    assert len(data.head()) >= MINIMUM_ROWS
