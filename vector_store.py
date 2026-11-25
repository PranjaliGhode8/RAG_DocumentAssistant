
from ingestion import load_pdf, split_documents
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks):
    # Initialize the local embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Convert chunks into embeddings and store them in FAISS
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vector_store


if __name__ == "__main__":
    pdf_path = "data/sample_company_policy.pdf"

    # Load the PDF
    documents = load_pdf(pdf_path)

    # Split the PDF into chunks
    chunks = split_documents(documents)

    # Create the vector database
    vector_store = create_vector_store(chunks)

    # Save the FAISS database locally
    vector_store.save_local("faiss_index")

    print("FAISS vector database created successfully!")
    print(f"Total chunks stored: {len(chunks)}")
