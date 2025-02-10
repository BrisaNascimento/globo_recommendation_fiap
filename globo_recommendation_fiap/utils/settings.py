from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8'
    )
    BLOB_API_KEY: str
    CONTAINER_NAME: str
    LAST_NEWS: str
    LAST_ACCESS: str
