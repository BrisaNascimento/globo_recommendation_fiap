import pandas as pd
from sqlalchemy import text

from globo_recommendation_fiap.data.download_data import download_from_adls
from globo_recommendation_fiap.utils.db_connect import connect_to_db_local
from globo_recommendation_fiap.utils.settings import Settings


def load_last_access_table(df: pd.DataFrame) -> None:  # pragma no_cover
    engine = connect_to_db_local()
    table_name = 'user_last_access'
    df.applymap(
        lambda x: x.encode('utf-8', 'ignore').decode('utf-8')
        if isinstance(x, str)
        else x
    )
    df['content_embbeding'] = df['content_embbeding'].apply(
        lambda x: ', '.join(map(str, x))
    )
    # Create the table in PostgreSQL
    with engine.connect() as connection:
        # Drop the table if it already exists
        connection.execute(text(f'DROP TABLE IF EXISTS {table_name};'))

        # Create the table
        create_table_query = f"""
            CREATE TABLE {table_name} (
                userId INT,
                history TEXT,
                content_embbeding BYTEA,
                timestamp TIMESTAMP,
                access_date DATE,
                PRIMARY KEY (userId, timestamp)
            );
        """
        connection.execute(text(create_table_query))

    # Insert data into the table
    df.to_sql(
        table_name,
        engine,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=10000
    )

    print(f"Table '{table_name}' created and data inserted successfully!")


def load_last_news(df: pd.DataFrame) -> None:  # pragma no_cover
    engine = connect_to_db_local()
    table_name = 'last_news'
    df.applymap(
        lambda x: x.encode('utf-8', 'ignore').decode('utf-8')
        if isinstance(x, str)
        else x
    )
    df['content_embbeding'] = df['content_embbeding'].apply(
        lambda x: ', '.join(map(str, x))
    )
    # Create the table in PostgreSQL
    with engine.connect() as connection:
        # Drop the table if it already exists
        connection.execute(text(f'DROP TABLE IF EXISTS {table_name};'))

        # Create the table
        create_table_query = f"""
            CREATE TABLE {table_name} (
                page TEXT,
                content_embbeding BYTEA,
                issued TIMESTAMP,
                issued_date DATE,
                PRIMARY KEY (page)
            );
        """
        connection.execute(text(create_table_query))

    # Insert data into the table
    df.to_sql(
        table_name,
        engine,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=10000
    )

    print(f"Table '{table_name}' created and data inserted successfully!")


def load_ranking(df: pd.DataFrame) -> None:  # pragma no_cover
    engine = connect_to_db_local()
    table_name = 'last_news_ranking'
    df.applymap(
        lambda x: x.encode('utf-8', 'ignore').decode('utf-8')
        if isinstance(x, str)
        else x
    )
    # Create the table in PostgreSQL
    with engine.connect() as connection:
        # Drop the table if it already exists
        connection.execute(text(f'DROP TABLE IF EXISTS {table_name};'))

        # Create the table
        create_table_query = f"""
            CREATE TABLE {table_name} (
                history TEXT,
                last_access_rank INT,
                PRIMARY KEY (history)
            );
        """
        connection.execute(text(create_table_query))

    # Insert data into the table
    df.to_sql(
        table_name,
        engine,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=10000
    )

    print(f"Table '{table_name}' created and data inserted successfully!")


if __name__ == '__main__':

    last_access = download_from_adls(
            Settings().CONTAINER_NAME, Settings().LAST_ACCESS
        )

    last_news = download_from_adls(
            Settings().CONTAINER_NAME, Settings().LAST_NEWS
        )

    news_rank = download_from_adls(
            Settings().CONTAINER_NAME, Settings().LAST_NEWS_RANK
        )

    load_last_access_table(last_access)
    load_last_news(last_news)
    load_ranking(news_rank)
