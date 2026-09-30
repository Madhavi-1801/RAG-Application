"""
chunker.py
==========
Splits raw extracted text into smaller, manageable, overlapping chunks.

Why Chunking Matters in RAG:
-----------------------------
1. Token Limits: LLMs have maximum context window limits.
2. Embedding Quality: Embeddings created from concise chunks carry much higher 
   semantic precision than embeddings created from entire books or long docs.
3. Overlap: Overlapping text between consecutive chunks ensures key sentences 
   or ideas spanning chunk boundaries are not cut in half.
"""

from typing import List, Dict, Any
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class TextChunker:
    """Splits text documents into smaller chunks using recursive boundary rules."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size (int): Target character count per chunk.
            chunk_overlap (int): Number of overlapping characters between adjacent chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # RecursiveCharacterTextSplitter attempts to split on natural boundaries:
        # Paragraphs ("\n\n"), Lines ("\n"), Words (" "), and Characters ("").
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_text(self, text: str, source_name: str = "document") -> List[Document]:
        """
        Splits a text string into a list of LangChain Document objects.
        
        Args:
            text (str): The plain text content to split.
            source_name (str): Metadata attribute indicating source file name.
            
        Returns:
            List[Document]: List of chunked Document objects with metadata.
        """
        raw_chunks = self.splitter.split_text(text)
        
        documents = []
        for idx, chunk in enumerate(raw_chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": idx,
                    "source": source_name,
                    "char_count": len(chunk)
                }
            )
            documents.append(doc)

        return documents


if __name__ == "__main__":
    sample_text = "Retrieval-Augmented Generation (RAG) is a powerful AI technique.\n\n" \
                  "It combines information retrieval with large language models.\n" \
                  "This ensures accurate, grounded, and up-to-date answers."
    
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split_text(sample_text, source_name="sample.txt")
    
    print(f"Created {len(chunks)} chunks:")
    for i, c in enumerate(chunks):
        print(f"--- Chunk {i} --- ({c.metadata})")
        print(c.page_content)
