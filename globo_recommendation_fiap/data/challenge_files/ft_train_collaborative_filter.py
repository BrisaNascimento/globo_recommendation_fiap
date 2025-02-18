import os
import re
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from globo_recommendation_fiap.data.download_data import download_from_adls
from globo_recommendation_fiap.data.upload_data import upload_to_dl
from globo_recommendation_fiap.utils.data_utils import (
    clean_text_columns,
    drop_cols,
)

# CREATE ACESSOS WITH FLAG_READ VIEW

# Download data from Lake
itens = download_from_adls('bronze', 'raw/globo/itens/itens.parquet')
acessos = download_from_adls('bronze', 'conformed/globo/users/users.csv')

# Remove unecessary columns to this view
columns_to_drop = ['timestampHistory', 'timestampHistory_new']
acessos = drop_cols(acessos, columns_to_drop)

# Clean spaces or other strings in necessary columns
acessos = clean_text_columns(acessos, ['history'])
itens = clean_text_columns(itens, ['page'])
itens = clean_text_columns(itens, ['body'], pattern="\\n", replacement=" ")

# Join access data to items data
acessos = acessos.merge(itens, left_on='history', right_on='page', how='left')
columns_to_drop = ['page', 'url', 'issued', 'modified', 'title', 'caption']
acessos = drop_cols(acessos, columns_to_drop)

# Count words in news body (perform much better in Spark)
acessos['body_ajust'] = acessos['body'].astype(str).apply(
    lambda x: re.sub(r"[.,-]", " ", x)
    )
acessos['words_count'] = acessos['body_ajust'].apply(lambda x: len(x.split()))
columns_to_drop = ['body', 'body_ajust']
acessos = drop_cols(acessos, columns_to_drop)


# Create flag for readed news (perform much better in Spark)
def create_flag_lidos(df):
    reading_rate = 300 / 60  # 300 words per 60 seconds
    min_scroll_percent = 0.50
    df['scrollPercentageHistory_perc'] = df['scrollPercentageHistory'] / 100
    df['words_read'] = df['words_count'] * df['scrollPercentageHistory_perc']
    df['min_req_time_sec'] = df['words_read'] / reading_rate
    df['timeOnPageHistory_sec'] = df['timeOnPageHistory'] / 1000
    df['flag_read'] = np.where(
        (df['timeOnPageHistory_sec'] >= df['min_req_time_sec'])
        & (df['scrollPercentageHistory_perc'] >= min_scroll_percent), 1, 0
    )
    return df


acessos = create_flag_lidos(acessos)
columns_to_drop = [
    'scrollPercentageHistory_perc',
    'words_read',
    'min_req_time_sec',
    'timeOnPageHistory_sec',
    'words_count']
acessos = drop_cols(acessos, columns_to_drop)

# Upload data to lake
# acessos.to_parquet("local/acessos/", compression='snappy', index=False)
upload_to_dl(acessos, 'silver', 'globo/acessos_flag_read/acessos.parquet')

# CREATE ACESSOS_FILTRADOS VIEW
# Filter users with 10 news entried at least


def filtro_leitores(df, min_entries):
    user_counts = df.groupby('userId').size().reset_index(name='count')
    filtered_users = user_counts[user_counts['count'] >= min_entries]
    return df[df['userId'].isin(filtered_users['userId'])]


acessos_filt = filtro_leitores(acessos, 10)[
    ['userId', 'history', 'timestamp', 'flag_read']
    ]

upload_to_dl(
    acessos_filt,
    'silver',
    'globo/acessos_filtrados/acessos_filtrados.parquet'
    )

# CREATE ACESSOS_VAL VIEW

# Load and join aditional data
df_validacao = download_from_adls(
    'bronze',
    'conformed/globo/validation_cleanned/validation_cleanned.csv'
    )
df_classes = download_from_adls(
    'silver',
    'globo/itens_text_db_scan/itens_text_db_scan.parquet'
    )

# Remove unecessary columns
drop_columns = [
    'caption', 'title_sentiment_label', 'title_sentiment_score',
    'caption_sentiment_label', 'caption_sentiment_score',
    'cleaned_title', 'embbed_title'
]
df_classes = drop_cols(df_classes, drop_columns)

# Clean spaces
df_classes = clean_text_columns(df_classes, ['page'])

# Join data
df_acessos_val = acessos.merge(
    df_validacao,
    left_on=['userId', 'history'], right_on=['userId', 'history_adjusted'],
    how='left'
    )
df_acessos_val = df_acessos_val.merge(
    df_classes,
    left_on='history', right_on='page',
    how='left')
# Remove unecessary columns
drop_columns = [
        'userType_y', 'date', 'page',
]
df_acessos_val = drop_cols(df_acessos_val, drop_columns)

# Renaming columns before ingestion
df_acessos_val.rename(columns={
                                'userType_x': 'userType',
                                'history_adjusted': 'history_val',
                                'timestampHistory_adjusted':
                                'timestampHistory_val'
                                }, inplace=True)

upload_to_dl(df_acessos_val, 'silver', 'globo/acessos_val/acessos_val.parquet')
