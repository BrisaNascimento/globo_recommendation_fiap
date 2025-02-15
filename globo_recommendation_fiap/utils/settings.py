from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8'
    )
    BLOB_API_KEY: str
    CONTAINER_NAME: str
    LAST_NEWS: str
    LAST_ACCESS: str
    LAST_ACCESS_PART: str
    LAST_NEWS_RANK: str
    POSTGRES_PASSWORD: str
    POSTGRES_USER: str
    POSTGRES_DB: str
    PGADMIN_DEFAULT_EMAIL: str
    PGADMIN_DEFAULT_PASSWORD: str
    HOSTNAME_SERVER: str
    DB_CONTAINER_NAME: str
    PG_CONTAINER_NAME: str
    APP_CONTAINER_NAME: str
    PG_PORT: str
