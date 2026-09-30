"""
main.py
=======
Command-line entry point for the Console-based RAG Question Answering System.
Run this script using: python main.py
"""

import sys
import os
from rag_pipeline import RAGPipeline


def print_banner():
    banner = """
========================================
      Console RAG Question Answering
========================================
"""
    print(banner)


def main():
    print_banner()

    # Step 1: Initialize RAG Pipeline
    try:
        pipeline = RAGPipeline()
    except ValueError as val_err:
        print(f"\n[Initialization Error] {val_err}")
        print("Please check your '.env' file configuration and try again.\n")
        sys.exit(1)
    except Exception as err:
        print(f"\n[Unexpected Error] Could not initialize RAG Pipeline: {err}\n")
        sys.exit(1)

    # Step 2: Ask user for file path loop
    loaded = False
    while not loaded:
        print("Enter file path (or type 'exit' to quit):")
        file_path_input = input("> ").strip()

        if file_path_input.lower() in ["exit", "quit", "q"]:
            print("\nGoodbye!")
            sys.exit(0)

        if not file_path_input:
            print("Please enter a valid file path.\n")
            continue

        try:
            print("\nLoading document...")
            chunk_count = pipeline.load_and_index_document(file_path_input)
            loaded = True
            print(f"\nDocument loaded successfully! ({chunk_count} chunk(s) indexed)")
        except FileNotFoundError:
            print(f"\nError: File not found at '{file_path_input}'. Please check the path and try again.\n")
        except ValueError as val_err:
            print(f"\nDocument Loading Error: {val_err}\n")
        except Exception as e:
            print(f"\nError processing document: {e}\n")

    print("\nRAG system is ready!")
    print("\nAsk a question about the document.")
    print("Type 'exit' to quit, or 'change' to load a different document.\n")

    # Step 3: Interactive Question Answering Loop
    while True:
        try:
            question = input("\nQuestion: ").strip()

            if not question:
                continue

            if question.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break

            if question.lower() in ["change", "load", "new"]:
                print("\nChanging document...")
                # Reset loading loop
                loaded = False
                while not loaded:
                    print("Enter new file path (or type 'cancel' to stay with current document):")
                    new_path = input("> ").strip()
                    if new_path.lower() in ["cancel", "exit", "quit"]:
                        print(f"Continuing with current document: '{pipeline.current_filename}'")
                        loaded = True
                        break
                    try:
                        print("\nLoading document...")
                        pipeline.load_and_index_document(new_path)
                        loaded = True
                        print("\nNew document loaded successfully!")
                    except Exception as err:
                        print(f"\nError loading document: {err}\n")
                continue

            print("\nSearching vector database and generating answer...")
            result = pipeline.answer_question(question)
            
            print(f"\nAnswer:\n{result['answer']}")

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as err:
            print(f"\nAn error occurred while generating the answer: {err}")


if __name__ == "__main__":
    main()
