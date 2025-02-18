import pandas as pd
import re

# Function to load local data in parquet format
def load_parquet(path):
     return pd.read_parquet(path)

# Function to load local data in csv format
def load_csv(file_path):
    return pd.read_csv(file_path)

# Function to remove unwanted columns
def drop_cols(df, columns_to_drop):
    return df.drop(columns=columns_to_drop, errors='ignore')

# Function to clean spaces and other regex in specific columns
def clean_text_columns(df, columns, pattern="\\s+", replacement=""):
    for col in columns:
        df[col] = df[col].astype(str).apply(lambda x: re.sub(pattern, replacement, x))
    return df

# Function to clean spaces in specific columns
def clean_column_spaces(df, columns):
    for col in columns:
        df[col] = df[col].str.replace(r"\s+", "", regex=True)
    return df

# Function to convert columns in datetime to date
def convert_to_date(df, columns):
    for col in columns:
        df[f'{col}_date'] = pd.to_datetime(df[col]).dt.date
    return df