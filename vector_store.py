"""
vector_store.py
===============
Handles storing chunk embeddings in FAISS (Facebook AI Similarity Search) 
and executing similarity search queries.

How Vector Similarity Search Works:
----------------------------------
1. Every document chunk is converted into an embedding vector and stored in FAISS index space.
2. When a user asks a question, the question is also converted into an embedding vector.
3. FAISS performs vector distance calculations (e.g. L2 distance / Cosine similarity) 
   to find the 'k' stored document vectors closest to the question vector.
4. The document chunks corresponding to those closest vectors are returned as context.
"""

from typing import List, Tuple
from langchain_core.documents import Document
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS


class VectorStoreManager:
    """Manages creation and search operations on a FAISS vector index."""

    @staticmethod
    def create_vector_store(documents: List[Document], embedding_model) -> FAISS:
        """
        Creates a FAISS vector database from a list of Document chunks and an embedding model.

        Args:
            documents (List[Document]): List of chunked Document objects.
            embedding_model: Instantiated LangChain embedding object.

        Returns:
            FAISS: An indexed in-memory FAISS vector store instance.
        """
        if not documents:
            raise ValueError("Cannot build vector store from an empty list of documents.")

        print(f"[VectorStore] Indexing {len(documents)} document chunk(s) into FAISS...")
        vector_store = FAISS.from_documents(documents, embedding_model)
        print("[VectorStore] FAISS vector index created successfully.")
        return vector_store

    @staticmethod
    def search_similarity(
        vector_store: FAISS, 
        query: str, 
        k: int = 3
    ) -> List[Document]:
        """
        Performs vector similarity search to find top-k relevant document chunks.

        Args:
            vector_store (FAISS): Indexed FAISS vector database.
            query (str): User question or query string.
            k (int): Number of top relevant chunks to retrieve.

        Returns:
            List[Document]: Top-k relevant document chunks.
        """
        results = vector_store.similarity_search(query, k=k)
        return results

    @staticmethod
    def search_similarity_with_score(
        vector_store: FAISS, 
        query: str, 
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """
        Performs vector similarity search returning chunks alongside distance scores.

        Args:
            vector_store (FAISS): Indexed FAISS vector database.
            query (str): User question.
            k (int): Number of results.

        Returns:
            List[Tuple[Document, float]]: Pairs of (Document chunk, L2 distance score).
        """
        results_with_scores = vector_store.similarity_search_with_score(query, k=k)
        return results_with_scores


if __name__ == "__main__":
    from embeddings import get_embedding_model
    
    docs = [
        Document(page_content="Python is a popular programming language.", metadata={"id": 1}),
        Document(page_content="FAISS is an efficient library for vector search.", metadata={"id": 2}),
        Document(page_content="Retrieval-Augmented Generation enhances LLM answers.", metadata={"id": 3}),
    ]
    
    embed_model = get_embedding_model("huggingface")
    vstore = VectorStoreManager.create_vector_store(docs, embed_model)
    matches = VectorStoreManager.search_similarity(vstore, "What is FAISS?", k=1)
    
    print("\nSearch Result:")
    for doc in matches:
        print(f"Content: {doc.page_content} | Metadata: {doc.metadata}")
