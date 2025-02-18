import io
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from globo_recommendation_fiap.utils.lake_connector import connect_to_adls


def upload_to_dl(df, container_name: str, file_path: str
) -> None:
    """
    Convert Dataframe to Parquet and saves the combined data into
        a Datalake blob.
    """
    try:
        # Connecto to ADLS
        blob_name = file_path
        blob = connect_to_adls(container_name, blob_name)
        # Create Buffer in Parquet format
        buffer = io.BytesIO()
        table = pa.Table.from_pandas(df)
        pq.write_table(table, buffer)
        buffer.seek(0)
        # Upload blob
        blob.upload_blob(buffer.getvalue(), overwrite=True)
    except Exception as e:
        print(f'Upload parquet to lake was failed.{e}')
    else:
        print('Upload parquet to lake was sucessful.')
        return True


if __name__ == '__main__':  # pragma: no cover
    container_name = 'silver'
    file_path = 'globo/teste/teste.parquet'
    df = pd.read_parquet(
         'globo_recommendation_fiap/data/challenge_files/local/teste.parquet'
         )
    print(df)
    print(upload_to_dl(df, container_name, file_path))
