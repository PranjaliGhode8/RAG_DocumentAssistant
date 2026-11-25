import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Load the embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# 2. Load an uploaded PDF
def load_uploaded_pdf(file_bytes, file_name):

    with tempfile.NamedTemporaryFile(
        suffix=".pdf", delete=False
    ) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        for document in documents:
            document.metadata["source"] = file_name

        return documents

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# 3. Split documents into chunks
def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


# 4. Create FAISS vector database
def create_vector_store(chunks):

    embedding_model = get_embedding_model()

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vector_store


# 5. Complete PDF processing pipeline
def process_pdf(file_bytes, file_name):

    documents = load_uploaded_pdf(file_bytes, file_name)

    if not documents:
        raise ValueError("No pages could be extracted from the PDF.")

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No readable text was found in the PDF.")

    vector_store = create_vector_store(chunks)

    return vector_store, len(documents), len(chunks), chunks

# 6. Process multiple uploaded PDFs
def process_multiple_pdfs(uploaded_files):

    all_documents = []

    for uploaded_file in uploaded_files:

        file_bytes = uploaded_file.getvalue()

        documents = load_uploaded_pdf(
            file_bytes,
            uploaded_file.name
        )

        all_documents.extend(documents)

    if not all_documents:
        raise ValueError("No readable PDF documents were uploaded.")

    # Split all extracted documents into chunks
    chunks = split_documents(all_documents)

    if not chunks:
        raise ValueError("No readable text was found in the uploaded PDFs.")

    # Create one FAISS database for all PDFs
    vector_store = create_vector_store(chunks)

    total_pages = len(all_documents)
    total_chunks = len(chunks)

    return vector_store, total_pages, total_chunks, chunks