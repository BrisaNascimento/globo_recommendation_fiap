import io
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.lake_connector import connect_to_adls


def download_from_adls(container_name: str, file_path: str) -> pd.DataFrame:
    """
    Download the intended file from Datalake blob.

    Params:
    container_name (str): Specific container that usually represents a layer
    in a Data Lake.
    file_path: (str): The full path to the intended file within a container.
    """
    # Connecto to ADLS
    blob_name = file_path
    blob = connect_to_adls(container_name, blob_name)
    # Create buffer in Parquet format
    blob_data = blob.download_blob()
    blob_content = blob_data.readall()
    buffer = io.BytesIO(blob_content)
    # Transform blob into a DataFrame
    data = pd.read_parquet(buffer)

    print('Download from lake was sucessful.')
    return data


if __name__ == "__main__":
    container_name = "bronze"
    file_path = "raw/globo/validacao/validacao.parquet"
    # file_path = "raw/globo/itens/itens.parquet"
    # file_path = "raw/globo/teste/teste.parquet"
    # file_path = "raw/globo/treino/treino.parquet"
    # file_path = "raw/globo/validacao_k/validacao_k.parquet"
    # file_path = "raw/globo/validacao/validacao.parquet"
    data = download_from_adls(container_name, file_path)
    print(data.head())
