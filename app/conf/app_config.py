import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum
from dotenv import load_dotenv


load_dotenv()

class Environment(str, Enum):
    """Application environment types.

    Defines the possible environments the application can run in:
    development, staging, production, and test.
    """

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


# Determine environment
def get_environment() -> Environment:
    """Get the current environment.

    Returns:
        Environment: The current environment (development, staging, production, or test)
    """
    match Field(alias="APP_ENV", default="development").lower():
        case "production" | "prod":
            return Environment.PRODUCTION
        case "staging" | "stage":
            return Environment.STAGING
        case "test":
            return Environment.TEST
        case _:
            return Environment.DEVELOPMENT


class Settings(BaseSettings):
    """
    Configuration settings for the application.
    """
    model_config = SettingsConfigDict(extra="ignore")

    ENVIRONMENT: str = get_environment()
    PROJECT_NAME: str = Field(alias="PROJECT_NAME", default="multi-agent")

    LLM_KEY: str = Field(alias="GOOGLE_API_KEY", default="")
    LLM_MODEL: str = Field(alias="LLM_MODEL", default="gemini-2.0-flash")
    LLM_PROVIDER: str = Field(alias="LLM_PROVIDER", default="google_genai")

    FALLBACK_LLM_KEY: str = Field(alias="GOOGLE_API_KEY", default="")
    FALLBACK_LLM_MODEL: str = Field(alias="FALLBACK_LLM_MODEL", default="gemini-2.0-flash")
    FALLBACK_LLM_PROVIDER: str = Field(alias="FALLBACK_LLM_PROVIDER", default="google_genai")

    EMBEDER_MODEL: str = Field(alias="EMBEDER_MODEL", default="models/gemini-embedding-001")
    EMBEDER_PROVIDER: str = Field(alias="EMBEDER_PROVIDER", default="google_genai")

    LOGGING_DEFAULT_LEVEL: str = Field(alisas="DEFAULT_CONFIG_LEVEL", default="INFO")
    LOGGING_FILE_HANDLER_FILE: str = Field(alias="LOGGING_FILE_HANDLER_FILE", default="logs/app.log")
    LOGGING_LOG_HANDLER: list[str] = Field(alias="LOGGING_LOG_HANDLER", default= ["default"])
    MAX_LLM_CALL_RETRIES: int = Field(alias="MAX_LLM_CALL_RETRIES", default=3)

settings = Settings()