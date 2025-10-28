import os

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from conf.app_config import settings


class LLMConfig:
    # Configuration class for managing LLM settings and initialization.
    @staticmethod
    def get_llm(provider: str) -> init_chat_model:
        """
        Initialize and return a chat model based on the provider and model name.
        
        Args:
            provider (str): The name of the LLM provider.
            model (str): The name of the LLM model.
        
        Returns:
            An initialized chat model.
        """
        if provider == "google_genai":
            os.environ["GOOGLE_API_KEY"]
            return init_chat_model(settings.LLM_MODEL, model_provider=provider)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Supported providers: google_genai.")
        

class EmbederConfig:
    # Configuration class for managing embedding model settings.
    @staticmethod
    def get_embeder(provider: str) -> GoogleGenerativeAIEmbeddings:
        """
        Initialize and return an embedding model.
        
        Returns:
            An initialized embedding model.
        """
        if provider == "google_genai":
            os.environ["GOOGLE_API_KEY"]
            return GoogleGenerativeAIEmbeddings(model_name = settings.EMBEDER_MODEL)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}. Supported providers: google_genai.")
        

llm = LLMConfig.get_llm(settings.LLM_PROVIDER)
embeder = EmbederConfig.get_embeder(settings.EMBEDER_PROVIDER)