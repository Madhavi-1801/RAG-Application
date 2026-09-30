"""
rag_pipeline.py
===============
Orchestrates the entire RAG workflow:
Document Loading -> Chunking -> Embedding -> Vector Store Indexing -> Retrieval -> LLM Answer Generation.
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv

from document_loader import DocumentLoader
from chunker import TextChunker
from embeddings import get_embedding_model
from vector_store import VectorStoreManager

# Load environment configuration
load_dotenv()


class MockLLM:
    """A zero-dependency local mock LLM for testing RAG pipeline without API keys."""
    def invoke(self, prompt: str):
        # Parse context and question from prompt
        context_part = ""
        question_part = ""
        if "RETRIEVED CONTEXT:" in prompt and "USER QUESTION:" in prompt:
            parts = prompt.split("RETRIEVED CONTEXT:")[1].split("USER QUESTION:")
            context_part = parts[0].strip()
            question_part = parts[1].split("ANSWER:")[0].strip()

        class MockResponse:
            def __init__(self, content_str):
                self.content = content_str

        # Check if question appears completely unrelated to retrieved context
        question_words = [w.lower() for w in question_part.replace("?", "").split() 
                          if len(w) > 3 and w.lower() not in ["what", "when", "where", "which", "about", "this", "document", "main", "with"]]
        
        has_match = any(w in context_part.lower() for w in question_words)

        if question_words and not has_match:
            return MockResponse("I couldn't find the answer in the uploaded document.")

        answer = (
            f"[Offline Demo Mode - No API Key Required]\n"
            f"Retrieved relevant content from document:\n\n"
            f"{context_part}"
        )
        return MockResponse(answer)


class RAGPipeline:
    """Core RAG pipeline class managing indexing and question-answering workflow."""

    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=int(os.getenv("CHUNK_SIZE", 500)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50))
        )
        self.top_k = int(os.getenv("TOP_K_RESULTS", 3))

        # Initialize Embedding Model
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")
        self.embedding_model = get_embedding_model(self.embedding_provider)

        # Initialize LLM
        self.llm = self._init_llm()

        # Vector store state (holds single loaded document index)
        self.vector_store = None
        self.current_filename = None

    def _init_llm(self):
        """Instantiates Chat LLM based on environment configuration."""
        llm_provider = os.getenv("LLM_PROVIDER", "mock").lower()
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

        if llm_provider in ["mock", "demo", "offline", "none"]:
            print("[LLM] Running in Offline Demo Mode (Zero API Key required)...")
            return MockLLM()

        elif llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key == "your_openai_api_key_here":
                print("\n[Notice] OPENAI_API_KEY missing in .env. Falling back to offline 'mock' LLM mode...")
                return MockLLM()
            from langchain_openai import ChatOpenAI
            print(f"[LLM] Initializing OpenAI Chat model: '{model_name}'...")
            return ChatOpenAI(model=model_name, temperature=0, openai_api_key=api_key)

        elif llm_provider in ["gemini", "google"]:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key or api_key == "your_google_gemini_api_key_here":
                print("\n[Notice] GOOGLE_API_KEY missing in .env. Falling back to offline 'mock' LLM mode...")
                return MockLLM()
            from langchain_google_genai import ChatGoogleGenerativeAI
            model_name = model_name if "gemini" in model_name else "gemini-1.5-flash"
            print(f"[LLM] Initializing Google Gemini Chat model: '{model_name}'...")
            return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)

        else:
            print(f"[Notice] Unknown LLM_PROVIDER '{llm_provider}'. Falling back to offline 'mock' LLM mode...")
            return MockLLM()

    def load_and_index_document(self, file_path: str) -> int:
        """
        Executes document ingestion pipeline:
        1. Text Extraction
        2. Text Chunking
        3. Vector Store Creation

        Returns:
            int: Number of chunk documents indexed into FAISS.
        """
        # Step 1: Load Document & Extract Text
        print(f"\n[1/3] Loading document from: '{file_path}'...")
        raw_text = self.loader.load(file_path)
        filename = os.path.basename(file_path)

        # Step 2: Split Text into Chunks
        print(f"[2/3] Splitting extracted text into chunks (size={self.chunker.chunk_size}, overlap={self.chunker.chunk_overlap})...")
        doc_chunks = self.chunker.split_text(raw_text, source_name=filename)
        print(f"      Generated {len(doc_chunks)} chunk(s).")

        # Step 3: Generate Embeddings and Build FAISS Vector Database
        print(f"[3/3] Generating embeddings and building FAISS vector database...")
        self.vector_store = VectorStoreManager.create_vector_store(
            documents=doc_chunks,
            embedding_model=self.embedding_model
        )
        self.current_filename = filename
        print(f"      Vector database successfully built for '{filename}'.")

        return len(doc_chunks)

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Executes question answering workflow:
        1. Perform Similarity Search to retrieve top-k chunks.
        2. Assemble Prompt with strict grounding rules.
        3. Invoke LLM and return formatted answer.

        Returns:
            Dict containing 'answer' string and 'retrieved_chunks' list.
        """
        if not self.vector_store:
            raise RuntimeError("No document loaded. Call load_and_index_document() first.")

        # Step 1: Similarity Search in FAISS
        retrieved_chunks = VectorStoreManager.search_similarity(
            vector_store=self.vector_store,
            query=question,
            k=self.top_k
        )

        # Build Context String from retrieved chunk contents
        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            context_blocks.append(f"--- Chunk {idx} (ID: {chunk.metadata.get('chunk_id')}) ---\n{chunk.page_content}")
        
        context_str = "\n\n".join(context_blocks)

        # Step 2: Build Strict Grounded System Prompt
        prompt = (
            "You are an accurate assistant for document question-answering tasks.\n"
            "Use ONLY the following retrieved context from the uploaded document to answer the user's question.\n\n"
            "STRICT RULES:\n"
            "1. Answer the question relying ONLY on the facts explicitly stated in the context below.\n"
            "2. If the answer cannot be found or deduced from the context, respond EXACTLY with:\n"
            '   "I couldn\'t find the answer in the uploaded document."\n'
            "3. Do NOT make assumptions, invent facts, or extrapolate beyond the provided text.\n\n"
            f"RETRIEVED CONTEXT:\n{context_str}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            "ANSWER:"
        )

        # Step 3: Pass Prompt to LLM
        response = self.llm.invoke(prompt)

        # Format output string
        answer_text = response.content if hasattr(response, 'content') else str(response)

        return {
            "answer": answer_text.strip(),
            "retrieved_chunks": retrieved_chunks
        }
