
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize a local embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Sample text
text = "Employees are entitled to 10 days of sick leave annually."

# Generate embedding
vector = embedding_model.embed_query(text)

# Display the result
print("Embedding successfully generated!")
print("Vector dimensions:", len(vector))
print("First 10 values:", vector[:10])
