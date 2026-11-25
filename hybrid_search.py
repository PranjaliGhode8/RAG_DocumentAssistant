
from langchain_community.retrievers import BM25Retriever


def hybrid_search(question, chunks, vector_store, k=5):

    # 1. Create keyword retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    # 2. Create semantic retriever using FAISS
    faiss_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 15,
            "lambda_mult": 0.5
        }
    )

    # 3. Retrieve documents from both methods
    keyword_results = bm25_retriever.invoke(question)
    semantic_results = faiss_retriever.invoke(question)

    # 4. Combine results using Reciprocal Rank Fusion (RRF)
    scores = {}
    documents = {}

    def add_results(results, weight):
        for rank, doc in enumerate(results, start=1):

            # Use document content as a deduplication key
            key = doc.page_content

            documents[key] = doc

            # RRF gives higher scores to higher-ranked results
            scores[key] = scores.get(key, 0) + (
                weight / (60 + rank)
            )

    # Equal importance initially
    add_results(keyword_results, 0.5)
    add_results(semantic_results, 0.5)

    # 5. Sort combined results
    ranked_results = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [
        documents[key] for key in ranked_results[:k]
    ]
