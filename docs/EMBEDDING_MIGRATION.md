# Embedding Model Migration Guide

## Overview
This document describes the costs, implications, and procedures for migrating between embedding models in the RAG pipeline. Based on error analysis from February 2026 production deployments.

## Critical Principle: Vector Space Incompatibility

**⚠️ CRITICAL WARNING**: Embedding models are NOT interchangeable.

When you switch from one embedding model to another, the vector coordinates for the same document become completely different. This is because each model has its own unique vector space.

### Example
The text "Apple stock up 5%" produces:
- **Gemini-embedding-001**: `[0.123, 0.456, 0.789, ... 768 dimensions]`
- **BAAI/bge-large-en-v1.5**: `[0.234, 0.567, 0.890, ... 1024 dimensions]`
- **sentence-transformers/all-MiniLM-L6-v2**: `[0.345, 0.678, 0.901, ... 384 dimensions]`

Your existing vector store (built with one model) becomes **completely useless** when you switch to another model.

## Current Production Config

```yaml
Embedding Model: Gemini (gemini-embedding-001)
Dimension: 768
Cost: $0.02 per 1M input tokens
Method: REST API (https://generativelanguage.googleapis.com)
Availability: Stable, widely available
```

## Alternative Models & Migration Paths

### Option 1: Local Deployment (BAAI/bge-large-en-v1.5)
**Best for**: Self-hosted deployments, eliminating API dependency
```
Model: BAAI/bge-large-en-v1.5
Parameters: 335M
Dimension: 1024
MTEB Score: 63.0 (top-tier benchmark)
Cost: Zero (runs locally)
License: Apache 2.0
```

**Migration Steps**:
1. Back up current vector store: `cp -r data/vector_store data/vector_store.backup.gemini`
2. Update `.env`:
   ```
   EMBEDDING_MODEL_TYPE=local
   HUGGINGFACE_MODEL=BAAI/bge-large-en-v1.5
   ```
3. Clear existing vectors: `rm -rf data/vector_store/* data/vector_store/.chroma/`
4. Restart service (will re-embed all documents on first access)
5. **Expected time to completion**: ~10-30 minutes for 1000 articles (depends on CPU)

**Re-embedding Cost**:
- Time: `articles_count * avg_embedding_time`
- CPU: ~500ms per article on M1 Mac
- No API costs, but blocks service during re-embedding

### Option 2: Hugging Face Inference Endpoints (google/embeddinggemma-300m)
**Best for**: Matching original model behavior with guaranteed availability
```
Model: google/embeddinggemma-300m
Parameters: 300M  
Dimension: 256
Cost: ~$0.50-2.00/month (cheapest paid option)
Availability Issues: Free API now returns 410 Gone (reason for this migration)
```

**Why we left this**: The free HF Inference API deprecated this model in late 2025. Paid Inference Endpoints are available but require separate setup.

**Migration Steps**:
1. Create HF Inference Endpoint (follow [HF docs](https://huggingface.co/docs/inference-endpoints/index))
2. Update `.env`:
   ```
   EMBEDDING_MODEL_TYPE=huggingface
   HUGGINGFACE_TOKEN=hf_xxxxx
   HUGGINGFACE_MODEL=google/embeddinggemma-300m
   ```
3. Clear vector store and restart
4. Service will re-embed using endpoint

### Option 3: OpenAI Embeddings (text-embedding-3-small)
**Best for**: Applications heavily invested in OpenAI ecosystem
```
Model: text-embedding-3-small
Dimension: 1536
Cost: $0.02 per 1M tokens
Availability: Enterprise SLA
```

**Migration Steps**:
1. Ensure `OPENAI_API_KEY` is set
2. Update `.env`:
   ```
   EMBEDDING_MODEL_TYPE=openai
   OPENAI_API_KEY=sk_xxxxx
   ```
3. Clear vector store and restart
4. Service re-embeds using OpenAI API

## Re-embedding Procedures

### Automatic Re-embedding (Recommended)
When you start the service with a new embedding model configuration:
1. Vector store detects dimension mismatch
2. Automatically clears old vectors
3. On next news fetch, documents are re-embedded with new model
4. **Downside**: Service may return empty/partial results during re-embedding

### Manual Force Re-embedding
```bash
# Backup current store
cp -r data/vector_store data/vector_store.backup

# Clear the store
rm -rf data/vector_store/*

# Restart service
docker compose restart backend

# Re-embedding begins on next news fetch
curl http://localhost:8000/news/AAPL
```

### Bulk Re-embedding Script
For large document sets, consider:
```python
# backend/scripts/rebatch_vectors.py
from backend.services.vector_store import create_vector_store
from backend.utils.dataset import load_articles

# Load all articles from cache
articles = load_articles()  # ~1000s of articles

# Create fresh vector store with new model
store = create_vector_store(model_type="local")

# Re-embed in batches
for batch in chunks(articles, batch_size=50):
    store.add_documents(batch)
    print(f"Embedded {len(batch)} docs...")

# Verify embedding health
health = store.embedding_service.health_check()
print(f"Store health: {health}")
```

## Operational Costs & Timing

| Model | Setup Time | Re-embed Cost | Monthly Cost | Failure Risk |
|-------|-----------|--------------|-------------|------------|
| Gemini (current) | 5 min | API calls (~$0.05 for 1K docs) | $1-5 | Medium (API dependent) |
| BAAI local | 15 min | Zero | Zero | Low (no API) |
| HF Endpoints | 30 min | API calls | $0.50-2.00 | Medium (paid API) |
| OpenAI | 5 min | API calls (~$0.02) | $5-20 | Medium (API dependent) |

## Data Validation After Migration

After switching models, validate:

```bash
# 1. Check vector store health
docker compose exec backend python3 -c "
from backend.services.vector_store import create_vector_store
store = create_vector_store(model_type='...')
print(f'Documents in store: {len(store)}')
print(f'Embedding dimension: {store.embedding_service.embedding_dim}')
"

# 2. Test sentiment pipeline
curl http://localhost:8000/news/RELIANCE

# 3. Verify no zero-vectors exist
python3 backend/scripts/validate_vectors.py
```

## Zero-Vector Detection

If migration causes zero-vector poisoning (all dimensions = 0):
1. Embeddings will be rejected by health check in `model_validator.py`
2. Vector store will refuse to add contaminated batch
3. Check logs for: `"Vector store contamination detected"`
4. **Recovery**: Clear store and restart re-embedding

## Rollback Procedure

If migration fails:

```bash
# 1. Stop service
docker compose down

# 2. Restore backup
rm -rf data/vector_store
cp -r data/vector_store.backup.gemini data/vector_store

# 3. Restore .env to previous config
git checkout .env

# 4. Restart
docker compose up -d
```

## Production Migration Checklist

- [ ] Backup current vector store: `cp -r data/vector_store data/vector_store.backup.$(date +%s)`
- [ ] Update `.env` with new model config
- [ ] Test in development environment first
- [ ] Schedule migration during low-traffic window
- [ ] Monitor logs: `docker compose logs -f backend | grep -i "vector\|embedding\|contamination"`
- [ ] Verify re-embedding completes: check for "Added X documents" logs
- [ ] Run validation script: `curl http://localhost:8000/health`
- [ ] Test sentiment analysis on known stock: `curl http://localhost:8000/news/RELIANCE`
- [ ] Keep backup for 7 days before deletion

## Why Vector Spaces Can't Be Mixed

Each model learns a unique representation during training:
- **Gemini-embedding-001**: Trained on Google's corpus, optimized for semantic similarity
- **BAAI/bge-large-en-v1.5**: Trained on diverse multi-lingual corpus
- **sentence-transformers**: Trained on NLP tasks with different objectives

Even small differences in model architecture cause embeddings to diverge exponentially across 700+ dimensions.

**Example**: If Model A puts "Apple Inc stock" at coordinate `[0.8, 0.2, ...]` and Model B puts it at `[0.3, 0.7, ...]`, then query similarity calculations become nonsensical across models.

## References

- [Hugging Face MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard)
- [BAAI/bge-large-en-v1.5 Model Card](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [Gemini Embeddings API](https://ai.google.dev/docs/embeddings_api_overview)
- [Error Analysis: February 2026 RAG Failures](./ERROR_ANALYSIS.md)
