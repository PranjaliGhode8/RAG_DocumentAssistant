import os
import requests

API_KEY = os.getenv(
    "BEDROCK_API_KEY",
    os.getenv("AWS_ACCESS_KEY_ID")
)
DEFAULT_REGION = os.getenv("AWS_REGION", "us-east-1")
DEFAULT_MODEL = "us.amazon.nova-micro-v1:0"


def ask_question(
    prompt: str,
    model_id: str = DEFAULT_MODEL,
    system_prompt: str = "You are a helpful and concise AI assistant.",
    max_tokens: int = 512,
    temperature: float = 0.7,
    region: str = DEFAULT_REGION,
) -> str:
    """Asks a question to an experimental model on Amazon Bedrock."""
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/converse"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    payload = {
        "messages": [
            {"role": "user", "content": [{"text": prompt}]}
        ],
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    }

    if system_prompt:
        payload["system"] = [{"text": system_prompt}]

    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    result = response.json()
    return result["output"]["message"]["content"][0]["text"]


if __name__ == "__main__":
    question = "Explain quantum computing in 2 simple sentences."
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Question: {question}\n")
    answer = ask_question(question)
    print(f"Answer:\n{answer}")
