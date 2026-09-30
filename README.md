# Console-based RAG Question Answering System 🤖📄

A beginner-friendly, zero-GUI, terminal-based **Retrieval-Augmented Generation (RAG)** Question Answering system built with Python, LangChain, FAISS, and LLM APIs (OpenAI & Google Gemini).

---

## 📌 Project Overview & Goal

The goal of this project is to demonstrate the complete **RAG workflow** in its purest, most accessible form. Users can upload a document (`.pdf`, `.txt`, `.docx`) through the command prompt and ask multiple questions about its content. 

The system extracts text from the document, splits it into overlapping chunks, generates high-dimensional vector embeddings, indexes them in an in-memory **FAISS** vector store, retrieves top relevant chunks for each user query, and prompts an LLM to generate grounded answers strictly based on the document.

---

## 🏗️ Project Architecture & Workflow

```text
User File Upload (.pdf / .txt / .docx)
         ↓
  Document Loader  (document_loader.py)
         ↓
  Text Extraction  (Raw text string)
         ↓
   Text Chunking   (chunker.py - Recursive character splitting with overlap)
         ↓
 Embedding Model   (embeddings.py - Local HuggingFace / OpenAI / Gemini)
         ↓
 Vector Database   (vector_store.py - FAISS vector index creation)
         ↓
 User Question     (Console CLI loop in main.py)
         ↓
Question Embedding (Same Embedding Model converts question to vector)
         ↓
Similarity Search  (FAISS top-k L2 / Cosine nearest neighbors search)
         ↓
 Relevant Chunks   (Extracted top matching document segments)
         ↓
 LLM Context Prompt(rag_pipeline.py - Grounded system prompt assembly)
         ↓
    LLM Answer     (Grounded answer or "I couldn't find the answer...")
```

---

## 📂 Project Structure

```text
rag-console/
├── main.py                # Command-line interactive loop and user interface
├── document_loader.py     # Extensible document loader (.pdf, .txt, .docx)
├── chunker.py             # Text splitting and chunking manager
├── embeddings.py          # Embedding model factory (Free local HF / OpenAI / Gemini)
├── vector_store.py        # FAISS vector store indexing and similarity retrieval
├── rag_pipeline.py        # Core RAG pipeline orchestrator
├── create_sample_docs.py  # Utility script to generate sample test files
├── requirements.txt       # Python dependencies specification
├── .env                   # Active environment configuration & API keys
├── .env.example           # Configuration template
└── README.md              # Project documentation and learning guide
```

---

## 📄 Explanation of Project Files

1. **`main.py`**:
   - The CLI entry point. Displays the welcome banner, asks the user for a file path, orchestrates document loading status messages, and manages the interactive question-answering terminal loop.
2. **`document_loader.py`**:
   - Handles text extraction across file formats (`.pdf` via `pypdf`, `.docx` via `python-docx`, `.txt` via built-in UTF-8 reading). Uses a registry dictionary mapping extensions to handler functions for easy extensibility.
3. **`chunker.py`**:
   - Splits extracted text into smaller segments using `RecursiveCharacterTextSplitter`. Attaches metadata (chunk ID, source name) to each chunk.
4. **`embeddings.py`**:
   - Instantiates vector embedding models. Supports free local CPU embeddings (`sentence-transformers/all-MiniLM-L6-v2`) via HuggingFace, as well as cloud models from OpenAI and Google Gemini.
5. **`vector_store.py`**:
   - Wraps the **FAISS** vector store library. Builds in-memory vector indices from document chunks and executes similarity searches to retrieve the top matching chunks for a question.
6. **`rag_pipeline.py`**:
   - Links all components together. Takes a user question, retrieves top-$k$ relevant chunks, formats a strict context-grounded prompt, calls the Chat LLM (`gpt-4o-mini` or `gemini-1.5-flash`), and returns the answer.
7. **`create_sample_docs.py`**:
   - Automatically generates sample `.txt`, `.pdf`, and `.docx` test files in a `sample_docs/` directory so you can test the system right away.

---

## 🔬 Core RAG Concepts Explained

### 1. How Document Chunks Are Created
Large documents cannot be sent to an embedding model or LLM as a single block due to token limits and reduced semantic precision. 

- **Recursive Character Splitting**: The text splitter attempts to divide text along natural sentence and paragraph boundaries (`"\n\n"`, `"\n"`, `" "`, `""`) to keep coherent paragraphs together.
- **Chunk Size (e.g., 500 chars)**: Sets the maximum length of each chunk.
- **Chunk Overlap (e.g., 50 chars)**: Ensures consecutive chunks share boundary characters so critical sentences bridging chunk divisions aren't cut in half.

### 2. What Embeddings Are
An **embedding** converts a text string into a fixed-length list of floating-point numbers (a numerical vector in high-dimensional space).

- **Semantic Proximity**: Words or sentences with similar meanings (e.g., *"How do I split text?"* and *"What is document chunking?"*) produce vectors that are geometrically close to each other in vector space.
- **Dimensionality**: A model like `all-MiniLM-L6-v2` produces a 384-dimensional vector for any text chunk.

### 3. How Vector Similarity Search Works
When a user asks a question:
1. The question string is converted into a vector using the **same** embedding model used for document chunks.
2. **FAISS** calculates geometric distance metrics (such as L2 Euclidean distance or Cosine Similarity) between the question vector and all stored document chunk vectors.
3. The $k$ chunks with the smallest distance (highest similarity) are selected as the context.

### 4. How Retrieved Chunks Are Passed to the LLM
The top-$k$ retrieved chunks are concatenated into a structured prompt context block:

```text
You are an assistant for document question-answering.
Use ONLY the following retrieved context to answer the question:

--- Chunk 1 ---
[Content of retrieved chunk 1]

--- Chunk 2 ---
[Content of retrieved chunk 2]

USER QUESTION: What is RAG?

STRICT RULE: If the answer cannot be found in the context above, 
respond with "I couldn't find the answer in the uploaded document."
```

### 5. Why RAG is Better Than Sending the Whole Document
- **Bypasses Context Limits**: Models have maximum token window limits. RAG works seamlessly on 1,000-page manuals.
- **Reduces Latency & Costs**: Sending 500 relevant tokens costs significantly less and runs faster than sending 100,000 document tokens every query.
- **Prevents Hallucinations ("Lost in the Middle")**: LLMs often ignore information buried in huge context windows. Precise retrieval focuses the LLM's attention on the exact relevant facts.
- **Instant Updates**: Knowledge can be updated by adding new documents to the vector store without costly model re-training or fine-tuning.

---

## 🛠️ Setup & Installation

### Step 1: Prerequisites
- Python **3.9+** installed on your system.
- An API Key for **OpenAI** or **Google Gemini**.

### Step 2: Clone / Open Project Folder
Navigate to the project root directory in your command prompt:

```bash
cd rag-console
```

### Step 3: Install Dependencies
Install all required libraries:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Environment Configuration (`.env`)

Create or edit the `.env` file in the project root:

```ini
# Choose LLM Provider: "openai" or "gemini"
LLM_PROVIDER=openai

# LLM Model Choice: "gpt-4o-mini" (OpenAI) or "gemini-1.5-flash" (Gemini)
LLM_MODEL=gpt-4o-mini

# API Keys
OPENAI_API_KEY=your_actual_openai_api_key_here
GOOGLE_API_KEY=your_actual_google_gemini_api_key_here

# Embedding Provider: "huggingface" (FREE local model, no API key needed!), "openai", or "gemini"
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Number of chunks to retrieve per question
TOP_K_RESULTS=3
```

> **💡 Pro Tip**: Using `EMBEDDING_PROVIDER=huggingface` runs embedding generation 100% free locally on your CPU!

---

## 🚀 Running the Project

### 1. Generate Sample Test Documents
Run the helper script to create sample `.txt`, `.pdf`, and `.docx` test files:

```bash
python create_sample_docs.py
```

### 2. Launch the Console Application
Start the RAG interactive CLI:

```bash
python main.py
```

---

## 🧪 Sample Testing Workflow

1. Start `main.py`.
2. When prompted for a file path, enter:
   ```text
   sample_docs/sample.pdf
   ```
3. Wait for the status messages to confirm document loading, chunking, embedding, and FAISS indexing.
4. Try asking valid questions based on the document:
   - **Question**: `What is the main purpose of RAG?`
   - **Question**: `What are the key advantages of RAG?`
   - **Question**: `What file formats are supported?`
5. Try asking an out-of-context question not mentioned in the document:
   - **Question**: `What is the capital of France?`
   - **Answer**: `I couldn't find the answer in the uploaded document.`
6. Try loading another document by typing `change` or type `exit` to quit.

---

## 🔍 Troubleshooting Section

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| `OPENAI_API_KEY is not configured` | `.env` file contains placeholder API key. | Open `.env` and paste your actual key or set `LLM_PROVIDER=gemini` with your Gemini key. |
| `ImportError: pypdf is required` | Dependency missing. | Run `pip install pypdf python-docx`. |
| `Unsupported file format '.xyz'` | Selected file extension is not `.pdf`, `.txt`, or `.docx`. | Provide a supported file format or register a handler in `document_loader.py`. |
| `FAISS installation error on Windows` | Compiler or arch mismatch. | Ensure you installed `faiss-cpu` (`pip install faiss-cpu`). |
| `UnicodeDecodeError on .txt file` | Special text encoding. | `document_loader.py` handles fallback encodings automatically (`utf-8`, `latin-1`). |

---

## 🚀 Future Improvements & Extensions

Once you master this single-document console RAG system, here is how it can be extended:

1. **Multi-Document Support**: Store vectors from multiple documents with `source` metadata to search across an entire repository.
2. **Persistent Vector Stores**: Save FAISS indexes to disk (`vector_store.save_local("faiss_index")`) so files don't need re-indexing on restart.
3. **Conversation Memory**: Track chat history so the LLM remembers previous follow-up questions.
4. **Source & Page Citations**: Include exact file names and page/paragraph numbers in the answer output.
5. **Hybrid Search**: Combine BM25 keyword matching with vector similarity search for higher precision.
6. **Reranking**: Use a Cross-Encoder reranker (e.g. Cohere Rerank) to re-order retrieved chunks before sending to the LLM.
7. **Local LLMs**: Replace cloud API models with local models using Ollama or Llama.cpp for 100% offline RAG.
8. **Streaming Responses**: Stream LLM tokens back to the console in real-time as they generate.
