"""
API model availability validation and health checks.

Prevents the "404 NOT_FOUND" cascade documented in error analysis:
- Validates model names against live API before startup
- Detects zero-vector poisoning in embedding calls
- Provides graceful degradation when models unavailable
"""

import logging
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ModelValidator:
    """Validates model availability and API health."""

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_models_cache: Optional[List[str]] = None

    def validate_gemini_models(self) -> Dict[str, bool]:
        """
        Query Gemini API to discover available models.
        
        Returns:
            Dict with model availability: {'gemini-3-flash-preview': True, ...}
        
        Context:
            The 404 error "models/gemini-3-flash is not found for API version v1beta"
            occurs because the API key may not have access to all released models.
            This validates models BEFORE attempting to use them, preventing
            cascading failures in RAG pipeline initialization.
        """
        if not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY not set. Skipping model validation.")
            return {}

        try:
            from google import genai
            
            client = genai.Client(api_key=self.gemini_api_key)
            available_models = client.models.list()
            
            model_names = [m.name for m in available_models]
            self.gemini_models_cache = model_names
            
            # Extract clean names (remove "models/" prefix)
            clean_names = [m.split('/')[-1] for m in model_names]
            
            logger.info(f"Discovered {len(clean_names)} available Gemini models: {clean_names}")
            
            # Validate requested model availability
            result = {}
            for model_variant in ["gemini-3-flash-preview", "gemini-3-flash", 
                                   "gemini-3-pro-preview", "gemini-3-pro",
                                   "gemini-embedding-001"]:
                is_available = any(model_variant in name for name in clean_names)
                result[model_variant] = is_available
                status = "✓ Available" if is_available else "✗ Not available"
                logger.info(f"  {model_variant}: {status}")
            
            return result
        except Exception as e:
            logger.error(f"Failed to validate Gemini models: {e}")
            return {}

    def validate_model_name(self, model_name: str, available_models: Dict[str, bool]) -> bool:
        """
        Check if requested model is in available models dict.
        
        Args:
            model_name: Model to validate (e.g., 'gemini-3-flash-preview')
            available_models: Result from validate_gemini_models()
        
        Returns:
            True if model is available, False otherwise
            
        Raises:
            Logs error if model unavailable (doesn't raise to allow graceful fallback)
        """
        is_available = available_models.get(model_name, False)
        if not is_available:
            logger.error(
                f"Model '{model_name}' not available for this API key. "
                f"Available: {[m for m, avail in available_models.items() if avail]}"
            )
        return is_available


class EmbeddingHealthCheck:
    """
    Detects zero-vector poisoning in embeddings.
    
    Context:
        When embedding API calls fail with 410 Gone or similar errors,
        code may catch exceptions but return all-zero vectors (numerically
        valid but semantically meaningless). These poison the vector store,
        causing queries to retrieve documents with random cosine similarity.
        
        This detector identifies poisoned embeddings before they corrupt
        the vector store.
    """

    @staticmethod
    def is_zero_vector(embedding: list, threshold: float = 0.0001) -> bool:
        """
        Check if embedding is suspiciously all-zero or near-zero.
        
        Args:
            embedding: Vector from embedding model
            threshold: Maximum absolute value for any dimension to flag as zero
        
        Returns:
            True if vector appears to be zero-vector (poisoned)
        """
        if not embedding:
            return True
        
        max_val = max(abs(v) for v in embedding) if embedding else 0
        is_zero = max_val < threshold
        
        if is_zero:
            logger.warning(
                f"Detected zero-vector embedding (max magnitude: {max_val}). "
                "This may indicate embedding API failure or 410 Gone error handling."
            )
        return is_zero

    @staticmethod
    def validate_embedding_batch(embeddings: List[list]) -> Dict[str, any]:
        """
        Validate a batch of embeddings for poisoning.
        
        Args:
            embeddings: List of embedding vectors
        
        Returns:
            Dict with validation results: 
            - total: int, number of embeddings checked
            - zero_count: int, number of zero-vectors detected
            - health_percentage: float, % of valid embeddings
            - contaminated: bool, True if >50% are zero-vectors
        """
        zero_count = sum(
            1 for emb in embeddings 
            if EmbeddingHealthCheck.is_zero_vector(emb)
        )
        total = len(embeddings)
        health = ((total - zero_count) / total * 100) if total > 0 else 0
        
        is_contaminated = (zero_count / total) > 0.5 if total > 0 else False
        
        if is_contaminated:
            logger.error(
                f"Vector store contamination detected: {zero_count}/{total} embeddings are zero-vectors. "
                f"This suggests embedding API failure. Consider clearing vector store and re-embedding."
            )
        
        return {
            "total": total,
            "zero_count": zero_count,
            "health_percentage": health,
            "contaminated": is_contaminated
        }
