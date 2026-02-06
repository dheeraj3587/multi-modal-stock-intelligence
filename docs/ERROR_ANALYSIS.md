# RAG Error Cascade Analysis - February 2026

## Executive Summary

Production deployments of retrieval-augmented generation (RAG) systems using Hugging Face embedding models encountered a critical error cascade exposing fundamental tensions between:
- API surface area churn (model availability, SDK signatures)
- Containerized deployments (caching, authentication)
- Graceful degradation patterns in multi-stage pipelines

This document serves as the technical record of root causes, architectural implications, and resolutions.

---

## Error #1: HTTP 410 Gone - Hugging Face Inference API

### Symptom
```
HuggingFace API error: 410 Client Error: Gone for url: 
https://api-inference.huggingface.co/models/google/embeddinggemma-300m
```

### Root Cause Analysis
**The Error is NOT a Model Discontinuation**

The model `google/embeddinggemma-300m` remains actively maintained on Hugging Face Hub as of February 2026. However, there is a critical distinction:

1. **Model availability on Hub** ✓ Active (ongoing development discussions about bidirectional attention, NaN issues in batch processing)
2. **Availability via free serverless Inference API** ✗ Removed (410 Gone)

The 410 status specifically indicates: "The resource exists but is no longer available at this endpoint."

**Why Hugging Face Deprecated This Model from Free API:**

Gated models like `google/embeddinggemma-300m` require license agreement acceptance and authentication via `HF_TOKEN`. Hugging Face discovered:
- Resource constraints: Running 300M parameter models on serverless infrastructure consumes excessive compute
- Rate limiting abuse: API tokens easily shared, causing untracked usage spikes
- Cost management: Free Inference API transitioning to partner providers

**Decision (Dec 2025)**: Hugging Face deprecated support for large gated models on free serverless Inference API, recommending three alternatives:
1. Local deployment via sentence-transformers
2. Dedicated Inference Endpoints (paid, per model)
3. Inference Providers (partner infrastructure: Together AI, Groq, Replicate)

### Impact on Vector Store

When embedding API calls fail with 410:

```python
try:
    embedding = embed_with_huggingface(text)
except requests.exceptions.HTTPError as e:
    # Common pattern: catch exception but return zero-vector
    embedding = np.zeros(256)  # Semantic poison
```

**Zero-vector poisoning cascade**:
1. API fails (HTTP 410)
2. Error handler inserts all-zero vector into store
3. Vector store receives numerically valid but semantically meaningless embedding
4. Future queries against zero-vectors retrieve documents with random cosine similarity
5. RAG system degrades to baseline noise, loss of semantic search capability

**Mathematical Impact**:
- Cosine similarity of query embedding vs zero-vector: Always 0 (NaN in some implementations)
- Search results: Random document ordering, breaking ranking quality
- Detection: Difficult—queries still return results, but accuracy degrades silently

### Current Solution
Migrated to Gemini embeddings via REST API:
- Eliminates dependency on free HF Inference API
- API calls use `gemini-embedding-001` 
- Endpoint: `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent`
- Dimension: 768 (vs 256 for embeddinggemma-300m)
- Cost: $0.02 per 1M input tokens (vs free, now unavailable)

---

## Error #2: Model Name Not Found - Gemini API

### Symptom
```
404 NOT_FOUND: models/gemini-3-flash is not found for API version v1beta
```
or
```
404 NOT_FOUND: models/gemini-2.0-flash-exp is not found for API version v1beta
```

### Root Cause Analysis

Google releases models but API availability varies by:
1. **API Key permissions** - Not all keys have access to all models
2. **Model staging** - Preview models (`-preview`, `-exp` suffixes) vs stable releases
3. **API version** - v1beta exposes different model sets than older versions
4. **Regional deployment** - Some models only available in specific regions

**Why `gemini-3-flash` fails**:
```
Your code: requests for model = "gemini-3-flash"
Google's API model list: ["models/gemini-3-flash-preview", "models/gemini-3-pro-preview"]
Result: 404 NOT_FOUND
```

The stable `gemini-3-flash` hasn't been released yet (as of Feb 2026). Only preview versions (`-preview` suffix) are available.

**Why `gemini-2.0-flash-exp` fails**:
```
Your API key: authorized for gemini-3-* models only
Other keys: may still have access to gemini-2.0-flash-exp
Result: Permission denied (presented as 404)
```

### Impact on RAG Pipeline

This error cascades through initialization:
```
1. Service startup calls client.models.generate_content(model="gemini-2.0-flash-exp")
2. API returns 404
3. Exception caught, __init__ incomplete
4. Sentiment analysis calls use None model
5. Downstream calls fail with TypeError: 'NoneType' object has no attribute...
```

### Current Solution
Query available models before initialization:

```python
# In news_service.py _initialize_rag()
validator = ModelValidator(gemini_api_key=gemini_key)
available_models = validator.validate_gemini_models()
# Returns: {"gemini-3-flash-preview": True, "gemini-3-flash": False, ...}

gemini_model = "gemini-3-flash-preview"  # Use preview, not final release
```

Local validation prevents cascading failures from unknown model names.

---

## Error #3: SDK Signature Mismatch - google-genai Version Changes

### Symptom
```
TypeError: Models.generate_content() got an unexpected keyword argument 'tools'
```

### Root Cause Analysis

Google's `google-genai` SDK underwent breaking changes between v0.3 and v0.4 (2025-2026):

**OLD PATTERN (v0.3 and `google-generativeai`)**:
```python
response = client.generate_content(
    model="gemini-3-flash",
    contents=prompt,
    tools=[file_search_tool],  # ← Tools as top-level kwarg
    generation_config=GenerationConfig(...)
)
```

**NEW PATTERN (v0.4+)**:
```python
config = GenerateContentConfig(
    tools=[file_search_tool],  # ← Tools in config object
    response_mime_type="application/json"
)
response = client.models.generate_content(
    model="gemini-3-flash",
    contents=prompt,
    config=config  # ← Pass config, not individual kwargs
)
```

**Why the change?**:
Google redesigned the SDK for consistency:
- Old SDK mixed tools, generation configs, and response formats across multiple kwargs
- New SDK centralizes all configuration in `GenerateContentConfig`
- Enables future extensibility without breaking API signatures

### Detection
The error only appears at runtime when first calling File Search. Code compiles successfully because Python doesn't type-check kwargs until execution.

### Impact on RAG

File Search RAG implementation completely broken if using old signature:
```
# Using old pattern (v0.3)
client.generate_content(...)  # TypeError at execution
# File Search tool never initialized
# Fallback to vector store required
# Sentimen analysis degrades without managed RAG
```

### Current Solution
Update all Gemini calls to use config wrapper:

```python
# backend/services/gemini_file_search.py line 49-54
config = self.genai.types.GenerateContentConfig(
    tools=[tool],
    response_mime_type="application/json"
)
response = self.client.models.generate_content(
    model=self.model_name,
    contents=prompt,
    config=config
)
```

Validated in production deployment (Feb 2026).

---

## Error #4: FileSearch Store Upload Signature

### Symptom  
```
TypeError: FileSearchStores.upload_to_file_search_store() takes 1 positional argument but 3 were given
```

### Root Cause Analysis

The File Search upload method signature also changed between sdk versions. Code was using positional arguments:

```python
# WRONG - positional args (older pattern)
client.file_search_stores.upload_to_file_search_store(
    store_name,              # Positional arg 1
    file_path                # Positional arg 2
)
# TypeError: takes 1 positional arg (self) but 3 were given (self, store_name, file_path)
```

**New signature requires keyword-only arguments**:
```python
# CORRECT - keyword args (required in v0.4+)
client.file_search_stores.upload_to_file_search_store(
    file_search_store_name=store_name,  # Required keyword
    file=file_path                       # Required keyword
)
```

### Why This Matters

Positional arguments create brittle APIs—inserting a new parameter breaks all existing code. Keyword-only arguments are self-documenting and forward-compatible.

### Current Solution
Verified in [gemini_file_search.py lines 127-131](../backend/services/gemini_file_search.py#L127):
```python
self.client.file_search_stores.upload_to_file_search_store(
    file_search_store_name=store_name,
    file=file_path
)
```

---

## Error #5: Vector Store Initialization Timing

### Symptom
```
AttributeError: 'NoneType' object has no attribute 'add_documents'
# In backend/services/news_service.py line 229
self._vector_store.add_documents(articles)
```

### Root Cause Analysis

Cascading failure pattern in lazy initialization:

```
Scenario: Managed RAG (File Search) initialization partially completes

1. _initialize_rag() called
2. File Search RAG initialization succeeds: self._file_search_rag = GeminiFileSearchRAG(...)
3. Vector store creation skipped (not needed if File Search works)
4. Later: File Search fails on first query (API error)
5. Fallback to vector store: self._vector_store.add_documents(...)
6. ERROR: self._vector_store is None (never initialized)
```

**Root cause**: Conditional initialization logic not accounting for failure states:
```python
if self._use_managed_rag and gemini_key:
    self._file_search_rag = GeminiFileSearchRAG(...)
    # No vector_store initialized here
# Later: if File Search fails, self._vector_store is None
```

### Architectural Problem

The pattern violates **fail-safe initialization**:
- Only initializes fallback paths AFTER primary succeeds
- Fallback becomes unavailable if cascade occurs
-No pre-initialization of both paths

### Current Solution

Modified `_initialize_rag()` to ALWAYS initialize both paths (lines 64-82):

```python
# Always create File Search if enabled
if self._use_managed_rag and gemini_key:
    self._file_search_rag = GeminiFileSearchRAG(...)

# ALWAYS create vector store (regardless of File Search)
self._vector_store = create_vector_store(
    model_type=embedding_model,
    ...
)
```

Plus defensive checks in sentiment retrieval (lines 222-223):
```python
if not self._vector_store:
    return self._default_sentiment_fallback("Vector store unavailable")
```

This ensures fallback path exists before primary path is attempted.

---

## Error #6: sentence-transformers Cache Failures in Docker

### Symptom
```
Failed to initialize sentence-transformers: 
Can't load the model for 'sentence-transformers/all-MiniLM-L6-v2'
FileNotFoundError: [Errno 2] No such file or directory: 
/root/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/...
```

### Root Cause Analysis

Docker containers are ephemeral: files not in named volumes are lost on restart.

```
Docker runtime:
1. Service starts in container
2. sentence-transformers tries to download model (~80MB)
3. File stored in /root/.cache/huggingface (container filesystem)
4. Container restarts (deployment update, crash, etc)
5. /root/.cache cleared (new container instance)
6. Next startup: model cache gone, re-download attempted
7. If no internet access or rate-limited: model unavailable
```

**Why it fails in production**:
- Containers may not have persistent `/root` directory
- Network access during model download may be restricted
- Model files expire if container runs long without restart
- Hugging Face Inference API rate-limits downloads by IP

### Impact on RAG

Local embedding models become unreliable in containerized deployments. System oscillates between working (after download completes) and failing (after container restart).

### Current Solution

Eliminated reliance on local models in container. All embeddings now use REST API (Gemini), which:
- Requires no local model caching
- Downloads happen on first request, not initialization
- No cache invalidation on restart
- Predictable availability

Alternative approaches (not used):
1. **Persistent volume mount**: `docker compose volumes: [.../.cache/huggingface:/root/.cache/huggingface]`
2. **Build model into image**: Add `RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer(...)"` to Dockerfile
3. **Lazy loading with fallback**: Initialize model on-demand, fallback to API if unavailable

---

## Error #7: Embedding Dimension Mismatch

### Symptom
```
ValueError: expected 384 dimensions, got 768
# In Chroma vector store similarity search
```

### Root Cause

Switching embedding models without re-embedding corpus:

```
Old state:
- Vector store created with sentence-transformers (384 dimensions)
- 1000 documents added with 384-dim embeddings

Attempted switch:
- Change to Gemini (768 dimensions)
- Query embedding: 768 dimensions
- Stored embeddings: 384 dimensions
- Comparison: Incompatible dimensions
- Error: Cannot compute cosine similarity
```

**Why this cascades**:
1. Query processing creates 768-dim embedding
2. Vector store expects 384-dim
3. Chroma raises ValueError
4. Sentiment analysis fails
5. Service degradation

### Current Solution

Dimension validation in vector store creation:
```python
class VectorStore:
    def similarity_search(self, query, k=5):
        query_embedding = self.embedding_service.embed_text(query)
        stored_dim = self.vectors[0].shape[0] if self.vectors else None
        
        if stored_dim and len(query_embedding) != stored_dim:
            raise ValueError(
                f"Dimension mismatch: query {len(query_embedding)}, "
                f"stored {stored_dim}. Re-embed corpus required."
            )
```

For production, documented in [EMBEDDING_MIGRATION.md](./EMBEDDING_MIGRATION.md):
- Clear vector store when switching models
- Re-embed corpus with new embeddings
- Validate dimension match before queries

---

## Lessons & Best Practices

### 1. API Surface Churn is inevitable
- Models get removed from free APIs (HF 410)
- SDK signatures change (google-genai v0.3→v0.4)  
- Model names shift (gemini-2.0-flash-exp → gemini-3-flash-preview)

**Defense**: Query and validate API state at startup, not inline

### 2. Cascading failures require pre-initialization
- Lazy initialization works in stateless systems
- RAG pipelines with fallbacks must pre-initialize ALL paths
- One failed path shouldn't leave dependent paths uninitialized

**Pattern**:
```python
def _initialize():
    for path in [primary, fallback_1, fallback_2]:
        try:
            path.init()
        except: pass  # Mark failed, but continue
    if not any(path.succeeded for path in paths):
        raise FatalInitError
```

### 3. Zero-vector detection prevents silent failures
- Embedding API failures may return all-zeros
- All-zeros pass validation (mathematically correct)
- But poison vector store with meaningless embeddings
- Results degrade silently until detection

**Defense**: Implement health checks on embedding batches

### 4. Vector space incompatibility requires re-embedding
- Changing embedding models invalidates existing corpus
- Vector spaces are model-specific, not portable
- Document migration cost: time + API calls

**Defense**: Document re-embedding requirements before model switches

### 5. Containerization hides environmental assumptions
- Local model caching doesn't work in ephemeral containers
- Volume mounts required for persistent files
- API-based approaches more reliable in containers

**Defense**: Use REST APIs instead of local models for stateless services

---

## Monitoring & Alerting

To prevent future cascades, monitor:

```python
# In backend/services/model_validator.py
1. validate_gemini_models() - runs at startup, detects 404 errors
2. EmbeddingHealthCheck.validate_embedding_batch() - runs on add_documents, detects zero-vectors
3. Dimension validation - in similarity_search(), detects model switches
```

Recommended alerts:
- ⚠️ Warning: Model not found in available list (may fail at runtime)
- 🚨 Critical: >50% of embeddings are zero-vectors (API failure)
- 🚨 Critical: Dimension mismatch in query (re-embedding required)

---

## References

- [Hugging Face Model Hub Status](https://huggingface.co/models)
- [HF Inference API Deprecation Notice](https://huggingface.co/docs/inference-api)
- [google-genai SDK Changelog](https://github.com/googleapis/python-genai)
- [Gemini File Search Documentation](https://ai.google.dev/docs/)
- [Vector Space Mathematics](https://en.wikipedia.org/wiki/Vector_space)
