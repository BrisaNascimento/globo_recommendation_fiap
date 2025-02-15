from sqlalchemy import create_engine

from globo_recommendation_fiap.utils.settings import Settings


def connect_to_db():
    db_url = (
        f'postgresql+psycopg://{Settings().POSTGRES_USER}:'
        f'{Settings().POSTGRES_PASSWORD}@{Settings().HOSTNAME_SERVER}:'
        f'{Settings().PG_PORT}/{Settings().POSTGRES_DB}'
    )
    return create_engine(db_url)


def connect_to_db_local():
    db_url = (
        f'postgresql+psycopg://{Settings().POSTGRES_USER}:'
        f'{Settings().POSTGRES_PASSWORD}@localhost:'
        f'{Settings().PG_PORT}/{Settings().POSTGRES_DB}'
    )
    return create_engine(db_url)
