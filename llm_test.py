from langchain_ollama import ChatOllama

# Connect to our local AI model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# Ask the AI a question
response = llm.invoke("Explain RAG in one simple sentence.")

# Print the AI's answer
print(response.content)