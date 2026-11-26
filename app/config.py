from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    model_name: str = "deepseek-chat"
    request_timeout: float = 30.0
    max_retries: int = 2
    temperature: float = 0.0
    max_output_tokens: int = 0
    prompt_variant: str = "on"
    input_price_per_1m: float = 0.0
    output_price_per_1m: float = 0.0


@lru_cache
def get_settings() -> Settings:
    return Settings()