import os
import sys
from io import StringIO

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.lake_connector import connect_to_adls


class CombineData:
    def __init__(self, data_path):
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
            .replace(r"[", ' ')
            .replace(r"]", ' ')
        )

    def load_files(self):
        """
        Loads all CSV files into a list of DataFrames.
        """
        for file in os.listdir(self.data_path):
            if file.endswith(".csv"):
                file_path = os.path.join(self.data_path, file)
                df = pd.read_csv(file_path)
                df_cleaned = df.apply(self.clean_series)
                self.dataframes.append(df_cleaned)
        print(f"{len(self.dataframes)} CSV files loaded.")

    def combine_files(self):
        """
        Combines all loaded DataFrames into a single DataFrame.
        """
        if not self.dataframes:
            raise ValueError("CSV file was not loaded.")
        self.combined_df = pd.concat(self.dataframes, ignore_index=True)
        print("CSV files successfully combined")

    def upload_to_dl(self, container_name: str, file_path: str):
        """
        Saves the combined DataFrame into a Datalake blob.
        """
        try:
            blob_name = file_path
            blob = connect_to_adls(container_name, blob_name)
            csv_buffer = StringIO()
            self.combined_df.to_csv(csv_buffer, index=False, encoding='utf-8')
            csv_buffer.seek(0)
            blob.upload_blob(csv_buffer.getvalue(), overwrite=True)
        except Exception as e:
            print(f"Upload csv to lake was failed.{e}")
        else:
            print('Upload csv to lake was sucessful')
            return True


if __name__ == "__main__":
    data_path = "globo_recommendation_fiap/data/train_data/"
    container_name = 'bronze'
    file_path = 'raw/globo/treino.csv'

    comb = CombineData(data_path)
    comb.load_files()
    comb.combine_files()
    comb.upload_to_dl(container_name, file_path)
