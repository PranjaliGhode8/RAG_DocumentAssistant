
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


# 1. Initialize local Ollama model
ollama_model = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# 2. Initialize Hugging Face embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Wrap models for Ragas
evaluator_llm = LangchainLLMWrapper(ollama_model)

evaluator_embeddings = LangchainEmbeddingsWrapper(
    embedding_model
)

print("Ragas evaluator initialized successfully!")

