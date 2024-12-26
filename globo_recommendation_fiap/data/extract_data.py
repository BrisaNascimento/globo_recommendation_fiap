import io
import os
import sys

import pandas as pd

# from io import StringIO
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from globo_recommendation_fiap.utils.lake_connector import connect_to_adls


class CombineData:
    def __init__(self, data_path: str):
        """
        Initializes the class with the path to the folder containing raw CSV
        files.
        """
        self.data_path = data_path
        self.dataframes = []

    @staticmethod
    def clean_series(series):
        """
        Cleans specific characters for each column in dataframe.
        """

        return series.map(
            lambda x: str(x)
            .replace(r'\n', ' ')
            .replace(r"'", ' ')
            .replace(r'[', ' ')
            .replace(r']', ' ')
        )

    def load_files(self) -> list:
        """
        Loads all CSV files into a list of DataFrames.
        """
        for file in os.listdir(self.data_path):
            if file.endswith('.csv'):
                file_path = os.path.join(self.data_path, file)
                df = pd.read_csv(file_path)
                df_cleaned = df.apply(self.clean_series)
                self.dataframes.append(df_cleaned)

        print(f'{len(self.dataframes)} CSV files loaded.')
        return self.dataframes

    def combine_files(self) -> pd.DataFrame:
        """
        Combines all loaded DataFrames into a single DataFrame.
        """
        if not self.dataframes:
            raise ValueError('CSV file was not loaded.')
        self.combined_df = pd.concat(self.dataframes, ignore_index=True)

        print('CSV files successfully combined.')
        return self.combined_df

    def convert_upload_to_dl(
        self, container_name: str, file_path: str
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
            table = pa.Table.from_pandas(self.combined_df)
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
    data_path = 'globo_recommendation_fiap/data/train_data/'
    container_name = 'bronze'
    file_path = 'raw/globo/treino/treino.parquet'

    comb = CombineData(data_path)
    comb.load_files()
    comb.combine_files()
    comb.convert_upload_to_dl(container_name, file_path)
