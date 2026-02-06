#!/usr/bin/env python3
"""
Test script for Hugging Face embeddings integration.
Demonstrates how to use HF Inference API for generating embeddings.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.vector_store import create_vector_store 


def test_huggingface_embeddings():
    """Test Hugging Face embeddings with sample texts."""
    
    # Get HF token from environment
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("❌ Error: HUGGINGFACE_TOKEN or HF_TOKEN not found in environment")
        print("\n💡 Get your free token from: https://huggingface.co/settings/tokens")
        print("Then set it: export HUGGINGFACE_TOKEN=hf_your_token_here")
        return
    
    print("🚀 Testing Hugging Face Embeddings Integration\n")
    print(f"✓ HF Token found: {hf_token[:10]}...")
    
    # Create vector store with HF embeddings
    print("\n📦 Creating vector store with HF embeddings...")
    vector_store = create_vector_store(
        model_type="huggingface",
        huggingface_token=hf_token,
        huggingface_model="google/embeddinggemma-300m"
    )
    
    print(f"✓ Vector store created (embedding dim: {vector_store.embedding_service.embedding_dim})")
    
    # Sample Indian stock market news
    sample_articles = [
        {
            "title": "Reliance Industries Q3 results exceed expectations",
            "content": "Reliance Industries reported strong Q3 earnings with revenue growth of 25% YoY. The company's retail and digital segments showed robust performance.",
            "source": "Economic Times"
        },
        {
            "title": "RBI maintains repo rate at 6.5%, focuses on inflation control",
            "content": "The Reserve Bank of India kept interest rates unchanged in its latest policy review, citing concerns over inflation and global uncertainties.",
            "source": "Business Standard"
        },
        {
            "title": "TCS announces dividend, stock surges 5%",
            "content": "Tata Consultancy Services declared a special dividend after posting record quarterly profits. Investors responded positively.",
            "source": "Mint"
        },
        {
            "title": "HDFC Bank merger with HDFC Ltd completed successfully",
            "content": "The merger of HDFC Bank and HDFC Ltd is now complete, creating India's largest financial services conglomerate.",
            "source": "MoneyControl"
        },
        {
            "title": "Nifty 50 hits new all-time high on FII inflows",
            "content": "Indian stock market benchmarks reached record levels as foreign institutional investors increased their holdings in Indian equities.",
            "source": "Zee Business"
        }
    ]
    
    print(f"\n📰 Adding {len(sample_articles)} sample articles...")
    vector_store.add_documents(sample_articles, text_key='content')
    print(f"✓ Added {len(vector_store)} documents to vector store")
    
    # Test semantic search
    print("\n🔍 Testing semantic search...")
    
    test_queries = [
        "banking sector news",
        "interest rate impact on stocks",
        "technology companies earnings",
        "merger and acquisition updates",
        "market performance today"
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        results = vector_store.similarity_search(query, k=2, score_threshold=0.3)
        
        if results:
            for i, (doc, score) in enumerate(results, 1):
                print(f"    {i}. [{score:.2f}] {doc['title']}")
        else:
            print("    No relevant results found")
    
    print("\n" + "="*60)
    print("✅ Hugging Face Embeddings Test Completed Successfully!")
    print("="*60)
    
    # Print statistics
    print(f"\n📊 Statistics:")
    print(f"  Model: google/embeddinggemma-300m")
    print(f"  Embedding Dimension: 256")
    print(f"  Documents Indexed: {len(vector_store)}")
    print(f"  API: Hugging Face Inference API (Free Tier)")
    
    print("\n💡 Next Steps:")
    print("  1. Add HUGGINGFACE_TOKEN to your .env file")
    print("  2. Set EMBEDDING_MODEL_TYPE=huggingface in .env")
    print("  3. Restart backend: docker compose restart backend")
    print("  4. Test API: curl http://localhost:8000/news/RELIANCE")


if __name__ == "__main__":
    test_huggingface_embeddings()
