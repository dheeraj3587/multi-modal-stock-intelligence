"""
AI Model configuration helper for sentiment analysis and embeddings.
"""

import os
from typing import Optional, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AIModelType(Enum):
    """Supported AI model types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LOCAL = "local"


class EmbeddingModelType(Enum):
    """Supported embedding model types."""
    OPENAI = "openai"
    GEMINI = "gemini"
    LOCAL = "local"


class AIConfig:
    """Configuration for AI models used in sentiment analysis."""
    
    def __init__(self):
        """Load configuration from environment variables."""
        # Main AI model for sentiment analysis
        self.ai_model_type = os.getenv("AI_MODEL_TYPE", "local").lower()
        
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        # Model names
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Embedding configuration
        self.embedding_model_type = os.getenv("EMBEDDING_MODEL_TYPE", "local").lower()
        self.openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
        # Hugging Face configuration
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        self.huggingface_model = os.getenv("HUGGINGFACE_MODEL", "google/embeddinggemma-300m")
        
        # Application settings
        self.use_rag_sentiment = os.getenv("USE_RAG_SENTIMENT", "true").lower() == "true"
        self.news_cache_hours = int(os.getenv("NEWS_CACHE_HOURS", "2"))
        self.vector_store_dir = os.getenv("VECTOR_STORE_DIR", "./data/vector_store")
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration and log warnings."""
        # Check if any API key is available for cloud models
        if self.ai_model_type == "openai" and not self.openai_api_key:
            logger.warning("AI_MODEL_TYPE is 'openai' but OPENAI_API_KEY is not set. Falling back to local.")
            self.ai_model_type = "local"
        
        if self.ai_model_type == "anthropic" and not self.anthropic_api_key:
            logger.warning("AI_MODEL_TYPE is 'anthropic' but ANTHROPIC_API_KEY is not set. Falling back to local.")
            self.ai_model_type = "local"
        
        if self.ai_model_type == "gemini" and not self.gemini_api_key:
            logger.warning("AI_MODEL_TYPE is 'gemini' but GEMINI_API_KEY is not set. Falling back to local.")
            self.ai_model_type = "local"
        
        # Check embedding configuration
        if self.embedding_model_type == "openai" and not self.openai_api_key:
            logger.warning("EMBEDDING_MODEL_TYPE is 'openai' but OPENAI_API_KEY is not set. Falling back to local.")
            self.embedding_model_type = "local"
        
        if self.embedding_model_type == "huggingface" and not self.huggingface_token:
            logger.warning("EMBEDDING_MODEL_TYPE is 'huggingface' but HUGGINGFACE_TOKEN is not set. Falling back to local.")
            self.embedding_model_type = "local"

        if self.embedding_model_type == "gemini" and not self.gemini_api_key:
            logger.warning("EMBEDDING_MODEL_TYPE is 'gemini' but GEMINI_API_KEY is not set. Falling back to local.")
            self.embedding_model_type = "local"
    
    def get_sentiment_model_config(self) -> Dict:
        """
        Get configuration for sentiment analysis model.
        
        Returns:
            Dict with model_type, model_name, and api_key
        """
        config = {
            "model_type": self.ai_model_type
        }
        
        if self.ai_model_type == "openai":
            config["model_name"] = self.openai_model
            config["api_key"] = self.openai_api_key
        elif self.ai_model_type == "anthropic":
            config["model_name"] = self.anthropic_model
            config["api_key"] = self.anthropic_api_key
        elif self.ai_model_type == "gemini":
            config["model_name"] = self.gemini_model
            config["api_key"] = self.gemini_api_key
        elif self.ai_model_type == "local":
            config["model_name"] = self.ollama_model
            config["base_url"] = self.ollama_base_url
        
        return config
    
    def get_embedding_config(self) -> Dict:
        """
        Get configuration for embedding model.
        
        Returns:
            Dict with model_type and api_key
        """
        config = {
            "model_type": self.embedding_model_type
        }
        
        if self.embedding_model_type == "openai":
            config["model_name"] = self.openai_embedding_model
            config["api_key"] = self.openai_api_key
        elif self.embedding_model_type == "gemini":
            config["model_name"] = "gemini-embedding-001"
            config["api_key"] = self.gemini_api_key
        elif self.embedding_model_type == "huggingface":
            config["huggingface_token"] = self.huggingface_token
            config["huggingface_model"] = self.huggingface_model
        else:  # local
            config["model_name"] = "all-MiniLM-L6-v2"  # Default local model
        
        return config
    
    def get_status(self) -> Dict:
        """
        Get current AI configuration status.
        
        Returns:
            Dict with configuration details
        """
        return {
            "sentiment_model": {
                "type": self.ai_model_type,
                "model": self.get_sentiment_model_config().get("model_name", "unknown"),
                "available": self._is_model_available(self.ai_model_type)
            },
            "embedding_model": {
                "type": self.embedding_model_type,
                "model": self.get_embedding_config().get("model_name", "unknown"),
                "available": self._is_embedding_available()
            },
            "use_rag": self.use_rag_sentiment,
            "vector_store": self.vector_store_dir
        }
    
    def _is_model_available(self, model_type: str) -> bool:
        """Check if the specified model type is available."""
        if model_type == "openai":
            return bool(self.openai_api_key)
        elif model_type == "anthropic":
            return bool(self.anthropic_api_key)
        elif model_type == "gemini":
            return bool(self.gemini_api_key)
        elif model_type == "local":
            return True  # Assume Ollama is always available
        return False
    
    def _is_embedding_available(self) -> bool:
        """Check if embedding model is available."""
        if self.embedding_model_type == "openai":
            return bool(self.openai_api_key)
        elif self.embedding_model_type == "huggingface":
            return bool(self.huggingface_token)
        return True  # Local embeddings always available
    
    def __str__(self) -> str:
        """String representation of config."""
        return f"AIConfig(sentiment={self.ai_model_type}, embedding={self.embedding_model_type}, rag={self.use_rag_sentiment})"


# Singleton instance
ai_config = AIConfig()


# Convenience functions
def get_sentiment_config() -> Dict:
    """Get sentiment model configuration."""
    return ai_config.get_sentiment_model_config()


def get_embedding_config() -> Dict:
    """Get embedding model configuration."""
    return ai_config.get_embedding_config()


def get_ai_status() -> Dict:
    """Get AI configuration status."""
    return ai_config.get_status()
