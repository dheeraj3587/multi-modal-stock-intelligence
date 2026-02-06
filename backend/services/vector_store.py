"""
Vector store service for news embeddings and semantic search.
Supports OpenAI, Hugging Face, and local embedding models.
"""

import os
import json
import numpy as np
import requests
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path
from backend.services.model_validator import EmbeddingHealthCheck

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates embeddings using OpenAI, Hugging Face, or local models."""
    
    def __init__(
        self, 
        model_type: str = "local", 
        openai_api_key: Optional[str] = None,
        huggingface_token: Optional[str] = None,
        huggingface_model: Optional[str] = None,
        gemini_api_key: Optional[str] = None
    ):
        """
        Initialize embedding service.
        
        Args:
            model_type: "openai", "huggingface", "gemini", or "local" (sentence-transformers)
            openai_api_key: OpenAI API key if using OpenAI embeddings
            huggingface_token: Hugging Face API token
            huggingface_model: HF model name (default: google/embeddinggemma-300m)
            gemini_api_key: Gemini API key if using Gemini embeddings
        """
        self.model_type = model_type
        self.embedding_dim = 384  # Default for sentence-transformers
        
        if model_type == "openai":
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
                self.embedding_model = "text-embedding-3-small"
                self.embedding_dim = 1536
                logger.info("Initialized OpenAI embeddings")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}. Falling back to local model.")
                self.model_type = "local"
        
        elif model_type == "gemini":
            self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not self.gemini_api_key:
                logger.error("Gemini API key not found. Falling back to local model.")
                self.model_type = "local"
            else:
                self.gemini_embed_model = "models/embedding-001"
                self.embedding_dim = 768
                logger.info(f"Initialized Gemini embeddings: {self.gemini_embed_model}")
        
        elif model_type == "huggingface":
            self.hf_token = huggingface_token or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
            self.hf_model = huggingface_model or os.getenv("HUGGINGFACE_MODEL", "google/embeddinggemma-300m")
            self.hf_api_url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
            
            if not self.hf_token:
                logger.error("Hugging Face token not found. Falling back to local model.")
                self.model_type = "local"
            else:
                # Determine embedding dimension based on model
                if "embeddinggemma-300m" in self.hf_model:
                    self.embedding_dim = 256
                elif "embeddinggemma-768m" in self.hf_model:
                    self.embedding_dim = 768
                elif "e5-large" in self.hf_model:
                    self.embedding_dim = 1024
                else:
                    self.embedding_dim = 384
                
                logger.info(f"Initialized Hugging Face embeddings: {self.hf_model}")
        
        if self.model_type == "local":
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dim = 384
                logger.info("Initialized local sentence-transformers model")
            except Exception as e:
                logger.error(f"Failed to initialize sentence-transformers: {e}")
                raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            return np.zeros(self.embedding_dim)
        
        try:
            if self.model_type == "openai":
                response = self.openai_client.embeddings.create(
                    input=text,
                    model=self.embedding_model
                )
                return np.array(response.data[0].embedding)
            
            elif self.model_type == "gemini":
                return self._embed_gemini([text])[0]
            
            elif self.model_type == "huggingface":
                return self._embed_huggingface([text])[0]
            
            else:  # local
                return self.model.encode(text, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def _embed_gemini(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using Google Gemini REST API (v1).
        Uses direct REST calls to avoid deprecated SDK issues.
        """
        try:
            embeddings = []
            for text in texts:
                truncated = text[:2000] if len(text) > 2000 else text
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={self.gemini_api_key}"
                payload = {
                    "model": "models/gemini-embedding-001",
                    "content": {"parts": [{"text": truncated}]},
                    "taskType": "SEMANTIC_SIMILARITY"
                }
                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                embedding = result.get("embedding", {}).get("values", [])
                if embedding:
                    embeddings.append(embedding)
                else:
                    embeddings.append([0.0] * self.embedding_dim)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            return np.zeros((len(texts), self.embedding_dim))
    
    def _embed_huggingface(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using Hugging Face Inference API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            2D numpy array of embeddings
        """
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        
        # HF API can handle batch requests
        payload = {"inputs": texts if len(texts) > 1 else texts[0]}
        
        try:
            response = requests.post(
                self.hf_api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Handle response format
            if isinstance(result, list):
                if len(texts) == 1:
                    # Single text response
                    return np.array([result[0] if isinstance(result[0], list) else result])
                else:
                    # Batch response
                    return np.array(result)
            else:
                logger.error(f"Unexpected HF API response format: {type(result)}")
                return np.zeros((len(texts), self.embedding_dim))
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Hugging Face API error: {e}")
            return np.zeros((len(texts), self.embedding_dim))
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            2D numpy array of embeddings
        """
        if not texts:
            return np.zeros((0, self.embedding_dim))
        
        try:
            if self.model_type == "openai":
                response = self.openai_client.embeddings.create(
                    input=texts,
                    model=self.embedding_model
                )
                embeddings = [data.embedding for data in response.data]
                return np.array(embeddings)
            
            elif self.model_type == "gemini":
                return self._embed_gemini(texts)
            
            elif self.model_type == "huggingface":
                return self._embed_huggingface(texts)
            
            else:  # local
                return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return np.zeros((len(texts), self.embedding_dim))


class VectorStore:
    """Simple in-memory vector store with persistence."""
    
    def __init__(self, embedding_service: EmbeddingService, persist_dir: Optional[str] = None):
        """
        Initialize vector store.
        
        Args:
            embedding_service: EmbeddingService instance
            persist_dir: Directory to persist vectors (optional)
        """
        self.embedding_service = embedding_service
        self.persist_dir = Path(persist_dir) if persist_dir else None
        
        # In-memory storage
        self.vectors = []  # List of embeddings
        self.metadata = []  # List of metadata dicts
        self.texts = []  # List of original texts
        
        # Load from disk if exists
        if self.persist_dir and self.persist_dir.exists():
            self.load()
    
    def add_documents(self, documents: List[Dict], text_key: str = 'content'):
        """
        Add documents to vector store.
        
        Args:
            documents: List of document dictionaries
            text_key: Key in document dict containing the text to embed
        """
        if not documents:
            return
        
        texts_to_embed = []
        valid_docs = []
        
        for doc in documents:
            text = doc.get(text_key, '')
            if text and text.strip():
                texts_to_embed.append(text)
                valid_docs.append(doc)
        
        if not texts_to_embed:
            return
        
        # Generate embeddings
        embeddings = self.embedding_service.embed_batch(texts_to_embed)
        
        # Check for zero-vector poisoning (prevents 410 Gone/API failure corruption)
        health = EmbeddingHealthCheck.validate_embedding_batch(embeddings.tolist())
        if health["contaminated"]:
            logger.error(
                f"Vector store contamination detected: Skipping batch add. "
                f"Health: {health['health_percentage']:.1f}% valid embeddings."
            )
            return
        
        # Add to storage
        for i, doc in enumerate(valid_docs):
            self.vectors.append(embeddings[i])
            self.texts.append(texts_to_embed[i])
            self.metadata.append(doc)
        
        logger.info(f"Added {len(valid_docs)} documents to vector store")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        score_threshold: float = 0.0
    ) -> List[Tuple[Dict, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if not self.vectors:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        # Calculate cosine similarity
        vectors_array = np.array(self.vectors)
        similarities = self._cosine_similarity(query_embedding, vectors_array)
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score >= score_threshold:
                results.append((self.metadata[idx], float(score)))
        
        return results
    
    def _cosine_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and documents."""
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
        return np.dot(doc_norms, query_norm)
    
    def clear(self):
        """Clear all stored vectors and metadata."""
        self.vectors = []
        self.metadata = []
        self.texts = []
    
    def save(self):
        """Persist vector store to disk."""
        if not self.persist_dir:
            return
        
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save vectors
            np.save(self.persist_dir / 'vectors.npy', np.array(self.vectors))
            
            # Save metadata and texts
            with open(self.persist_dir / 'metadata.json', 'w') as f:
                json.dump({
                    'metadata': self.metadata,
                    'texts': self.texts,
                    'saved_at': datetime.now().isoformat()
                }, f)
            
            logger.info(f"Saved vector store to {self.persist_dir}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def load(self):
        """Load vector store from disk."""
        if not self.persist_dir or not self.persist_dir.exists():
            return
        
        try:
            vectors_path = self.persist_dir / 'vectors.npy'
            metadata_path = self.persist_dir / 'metadata.json'
            
            if vectors_path.exists() and metadata_path.exists():
                # Load vectors
                vectors_array = np.load(vectors_path)
                self.vectors = list(vectors_array)
                
                # Load metadata and texts
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    self.metadata = data.get('metadata', [])
                    self.texts = data.get('texts', [])
                
                logger.info(f"Loaded {len(self.vectors)} vectors from {self.persist_dir}")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents in the store."""
        return self.metadata
    
    def __len__(self):
        """Return number of documents in store."""
        return len(self.vectors)


# Factory function to create vector store
def create_vector_store(
    model_type: str = "local",
    openai_api_key: Optional[str] = None,
    huggingface_token: Optional[str] = None,
    huggingface_model: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    persist_dir: Optional[str] = None
) -> VectorStore:
    """
    Create a vector store with the specified configuration.
    
    Args:
        model_type: "openai", "huggingface", "gemini", or "local"
        openai_api_key: OpenAI API key (if using OpenAI)
        huggingface_token: Hugging Face API token (if using HF)
        huggingface_model: HF model name (e.g., "google/embeddinggemma-300m")
        gemini_api_key: Gemini API key (if using Gemini)
        persist_dir: Directory to persist vectors
        
    Returns:
        VectorStore instance
    """
    embedding_service = EmbeddingService(
        model_type=model_type,
        openai_api_key=openai_api_key,
        huggingface_token=huggingface_token,
        huggingface_model=huggingface_model,
        gemini_api_key=gemini_api_key
    )
    return VectorStore(embedding_service=embedding_service, persist_dir=persist_dir)
