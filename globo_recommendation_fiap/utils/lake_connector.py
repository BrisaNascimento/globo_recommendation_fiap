import os
import sys

from azure.storage.blob import BlobClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.settings import Settings


def connect_to_adls(container_name: str, blob_name: str):
    """
    Connect to Azure Data Lake storage.

    Params:
    container_name (str): Specific container that usually represents a layer
    in a Data Lake.
    blob_name (str): The full path to the intended file within a container.
    """
    connection_string = Settings().BLOB_API_KEY
    blob = BlobClient.from_connection_string(
        conn_str=connection_string,
        container_name=container_name,
        blob_name=blob_name
    )
    return blob


if __name__ == "__main__":
    container_name = "bronze"
    blob_name = 'raw/globo/teste.csv'
    blob = connect_to_adls(container_name, blob_name)
    print(blob)
