import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
# Path to the directory where your documents are stored
DATA_PATH = "data/"
# Path to the directory where ChromaDB will store its data
CHROMA_PERSIST_DIRECTORY = "data/chroma_db"
# Name of the HuggingFace embedding model to use
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Chunk size for splitting documents
CHUNK_SIZE = 1000
# Overlap between chunks
CHUNK_OVERLAP = 100

def load_documents(data_path):
    """Loads all documents from the specified data path."""
    documents = []
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            print(f"Loaded {len(loader.load())} pages from {filename}")
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
            print(f"Loaded {filename}")
        # Add more loaders here for other file types if needed (e.g., Docx2txtLoader for .docx)
    return documents

def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True, # Helpful for identifying chunk source
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks, embedding_model_name=EMBEDDING_MODEL_NAME, persist_directory=CHROMA_PERSIST_DIRECTORY):
    """Creates and persists a ChromaDB vector store from document chunks."""
    print(f"Initializing embeddings with model: {embedding_model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    print(f"Creating vector store and persisting to: {persist_directory}")
    # Ensure the persist directory exists
    os.makedirs(persist_directory, exist_ok=True)

    # Create a new Chroma vector store from the documents
    # If you run this multiple times with the same persist_directory,
    # new data will be added. For a fresh start, delete the directory.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("Vector store created and persisted successfully.")
    return vector_store

def main():
    """Main function to run the ingestion pipeline."""
    load_dotenv() # Load environment variables, not strictly needed for ingest but good practice

    print("Starting document ingestion process...")

    # 1. Load documents
    documents = load_documents(DATA_PATH)
    if not documents:
        print(f"No documents found in {DATA_PATH}. Please add some.")
        return

    # 2. Split documents
    chunks = split_documents(documents)
    if not chunks:
        print("No chunks were created from the documents.")
        return

    # 3. Create and persist vector store
    create_vector_store(chunks)

    print("Document ingestion process completed.")

if __name__ == "__main__":
    main()