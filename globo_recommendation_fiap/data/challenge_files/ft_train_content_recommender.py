import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from globo_recommendation_fiap.data.download_data import download_from_adls

from globo_recommendation_fiap.data.upload_data import upload_to_dl
from globo_recommendation_fiap.utils.data_utils import (
                                                        clean_column_spaces, 
                                                        convert_to_date
                                                        )

# CREATE LAST ENTRIES VIEW

# Load Data from Lake
df_acessos = download_from_adls('silver', 'globo/acessos_flag_read/acessos.parquet')
df_itens = download_from_adls('bronze', 'raw/globo/itens/itens.parquet')[['page', 'issued']]
df_noticias = download_from_adls('silver', 'globo/content_embbeding/content_embedding_cleaned.parquet')

# Clean col spaces
df_itens = clean_column_spaces(df_itens, ['page'])
df_noticias = clean_column_spaces(df_noticias, ['page'])
df_acessos = clean_column_spaces(df_acessos, ['history', 'userId'])

# Create date columns from datetime
df_itens = convert_to_date(df_itens, ['issued'])
df_acessos = convert_to_date(df_acessos, ['timestamp'])
df_itens.rename(columns={'page': 'page_2'}, inplace=True)
df_acessos.rename(columns={'timestamp_date': 'date'}, inplace=True)

# Join embbeding information and issued date
df_noticias = df_noticias.merge(df_itens, left_on='page', right_on='page_2', how='inner')
df_noticias = df_noticias[['page', 'content_embbeding', 'issued', 'issued_date']]

# Join access information to issued date
acessos = df_acessos.merge(df_noticias, left_on='history', right_on='page', how='inner')
acessos = acessos[['userId', 'history', 'content_embbeding','timestamp', 'date', 'issued', 'issued_date']]

# Clean access and keeps only data accessed before news creation date - Consistency
acessos_cleanned = acessos[acessos['issued'] <= acessos['timestamp']].copy()

# Keep last 5 entries for user
acessos_cleanned.loc[:, 'rank'] = acessos_cleanned.groupby('userId')['timestamp'].rank(method='first', ascending=False)
ult_acessos = acessos_cleanned[acessos_cleanned['rank'] <= 5][['userId', 'history', 'content_embbeding', 'timestamp', 'date']]
ult_acessos.rename(columns={'date': 'access_date'}, inplace=True)
ult_acessos.reset_index(drop=True, inplace=True)

upload_to_dl(ult_acessos, 'silver', 'globo/acessos/ultimos_acessos.parquet')

# CREATE LAST NEWS VIEW

# More recent news (created after 13-08-2022)
noticias = df_noticias[df_noticias['issued_date'] >= pd.to_datetime('2022-08-13').date()]
noticias.reset_index(drop=True,inplace=True)

upload_to_dl(noticias, 'silver', 'globo/noticias/ultimas_noticias.parquet')

# CRATE TOP 20 MORE ENTRIED NEWS

# Create ranking for 20 more entried news
mais_acessadas = acessos_cleanned.groupby('history').size().reset_index(name='access_count')
mais_acessadas = mais_acessadas.sort_values(by='access_count', ascending=False)
mais_acessadas['access_rank'] = mais_acessadas['access_count'].rank(method='first', ascending=False)
mais_acessadas = mais_acessadas.merge(noticias, left_on='history', right_on='page', how='inner')
mais_acessadas = mais_acessadas[mais_acessadas['issued_date'] >= pd.to_datetime('2022-08-13').date()]
mais_acessadas['last_access_rank'] = mais_acessadas['access_count'].rank(method='first', ascending=False)
noticias_mais_acessadas = mais_acessadas[mais_acessadas['last_access_rank'] <= 20]

noticias_mais_acessadas = noticias_mais_acessadas[['history', 'last_access_rank']]
noticias_mais_acessadas['last_access_rank'] = noticias_mais_acessadas['last_access_rank'].astype(int)

upload_to_dl(noticias_mais_acessadas, 'silver', 'globo/noticias_mais_acessadas/mais_acessadas.parquet')