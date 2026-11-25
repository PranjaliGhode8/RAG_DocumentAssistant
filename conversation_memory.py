from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

standalone_prompt = ChatPromptTemplate.from_template("""
You are a question reformulation assistant.

Given the conversation history and the latest question,
rewrite the latest question as a standalone question.

Rules:
1. Resolve references such as "it", "they", "those".
2. Preserve the original meaning.
3. Do not answer the question.
4. If it is already standalone, return it unchanged.

Conversation history:
{chat_history}

Latest question:
{question}

Standalone question:
""")

def reformulate_question(question, chat_history):

    messages = standalone_prompt.invoke({
        "question": question,
        "chat_history": chat_history
    })

    response = llm.invoke(messages)

    return response.content.strip()