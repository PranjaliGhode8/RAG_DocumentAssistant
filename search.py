
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_vector_store():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        "faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_store


if __name__ == "__main__":
    vector_store = load_vector_store()

    question = input("Ask a question about your document: ")

    results = vector_store.similarity_search(
        question,
        k=3
    )

    print("\n--- Relevant Document Chunks ---\n")

    for i, doc in enumerate(results):
        print(f"\nResult {i + 1}")
        print(doc.page_content)
        print("Metadata:", doc.metadata)
