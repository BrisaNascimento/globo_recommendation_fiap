import os
import sys
from io import StringIO

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.lake_connector import connect_to_adls


def download_from_adls(container_name: str, file_path: str):
    """
    Download the intended file from Datalake blob.

    Params:
    container_name (str): Specific container that usually represents a layer
    in a Data Lake.
    file_path: (str): The full path to the intended file within a container.
    """
    blob_name = file_path
    blob = connect_to_adls(container_name, blob_name)
    blob_data = blob.download_blob()
    blob_content = blob_data.readall()
    csv_data = blob_content.decode('utf-8')
    data = pd.read_csv(StringIO(csv_data))
    print('Download from lake was sucessful.')
    return data


if __name__ == "__main__":
    container_name = "bronze"
    file_path = "raw/globo/validacao_k.csv"
    # file_path = "raw/globo/itens.csv"
    # file_path = "raw/globo/teste.csv"
    # file_path = "raw/globo/treino.csv"
    # file_path = "raw/globo/validacao_k.csv"
    # file_path = "raw/globo/validacao.csv"
    data = download_from_adls(container_name, file_path)
    print(data.head())
