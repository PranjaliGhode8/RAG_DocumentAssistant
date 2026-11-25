
from sentence_transformers import CrossEncoder


# 1. Load the reranker model
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# 2. Rerank retrieved documents
def rerank_documents(question, documents, top_n=3):

    if not documents:
        return []

    # Create question-document pairs
    pairs = [
        (question, doc.page_content)
        for doc in documents
    ]

    # Calculate relevance scores
    scores = reranker_model.predict(pairs)

    # Combine documents with their scores
    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    # Select the best documents
    selected_documents = [
        doc for doc, score in ranked_documents[:top_n]
    ]

    return selected_documents
