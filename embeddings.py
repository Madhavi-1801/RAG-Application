"""
embeddings.py
=============
Manages text embedding generation using local or cloud embedding models.

What Are Embeddings?
--------------------
An embedding converts a piece of text into a high-dimensional vector (a list of floating-point numbers).
Semantically similar texts (e.g. "dog" and "puppy") map to vectors that lie close together 
in vector space, allowing mathematical distance metrics (like Cosine Similarity) to calculate relevance.
"""

import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()


def get_embedding_model(
    provider: str = None, 
    model_name: str = None
) -> Any:
    """
    Instantiates and returns a LangChain embedding model instance.

    Args:
        provider (str): "huggingface" (default/free), "openai", or "gemini".
        model_name (str): Model identifier.

    Returns:
        Embeddings: A LangChain embedding object with embed_documents & embed_query methods.
    """
    provider = (provider or os.getenv("EMBEDDING_PROVIDER", "huggingface")).lower()

    if provider == "huggingface":
        model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            print(f"[Embeddings] Initializing local HuggingFace embedding model: '{model_name}'...")
            return HuggingFaceEmbeddings(model_name=model_name)
        except ImportError:
            # Fallback to community implementation if langchain-huggingface isn't installed
            from langchain_community.embeddings import HuggingFaceEmbeddings
            print(f"[Embeddings] Initializing local HuggingFace embedding model (community): '{model_name}'...")
            return HuggingFaceEmbeddings(model_name=model_name)

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError(
                "OPENAI_API_KEY missing in .env file. "
                "Set EMBEDDING_PROVIDER=huggingface to run free locally!"
            )
        from langchain_openai import OpenAIEmbeddings
        model_name = model_name or "text-embedding-3-small"
        print(f"[Embeddings] Initializing OpenAI embedding model: '{model_name}'...")
        return OpenAIEmbeddings(model=model_name, openai_api_key=api_key)

    elif provider in ["gemini", "google"]:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_google_gemini_api_key_here":
            raise ValueError(
                "GOOGLE_API_KEY missing in .env file. "
                "Set EMBEDDING_PROVIDER=huggingface to run free locally!"
            )
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        model_name = model_name or "models/embedding-001"
        print(f"[Embeddings] Initializing Google Gemini embedding model: '{model_name}'...")
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)

    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{provider}'. Supported: 'huggingface', 'openai', 'gemini'."
        )


if __name__ == "__main__":
    # Test embedding generator locally
    try:
        embeddings = get_embedding_model("huggingface")
        vector = embeddings.embed_query("Hello RAG world!")
        print(f"Embedding generated successfully! Vector dimension count: {len(vector)}")
        print(f"Sample vector slice (first 5 dimensions): {vector[:5]}")
    except Exception as e:
        print(f"Embedding test error: {e}")
