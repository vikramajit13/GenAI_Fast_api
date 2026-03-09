from dotenv import find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "GENAI_FAST_API"
    DATABASE_URL: str
    EMBEDDING_MODEL: str
    ENCODER_MODEL: str
    OLLAMA_HOST: str

    # This tells Pydantic to look for a .env file
    model_config = SettingsConfigDict(env_file=find_dotenv())


# Instantiate once to be used across the app
settings = Settings()
