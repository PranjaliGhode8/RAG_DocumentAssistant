import base64
import json
import os
import requests

# Amazon Bedrock API Key
API_KEY = os.getenv(
    "BEDROCK_API_KEY",
    os.getenv("AWS_ACCESS_KEY_ID")
)
REGION = os.getenv("AWS_REGION", "us-east-1")


def get_multimodal_embedding(image_source: str, text: str, region: str = REGION) -> list:
    """Generates multimodal embeddings (text + image) using Amazon Titan Multimodal Embeddings G1."""
    if image_source.startswith(("http://", "https://")):
        print(f"Downloading image from {image_source}...")
        image_resp = requests.get(image_source)
        image_resp.raise_for_status()
        image_bytes = image_resp.content
    else:
        with open(image_source, "rb") as f:
            image_bytes = f.read()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/amazon.titan-embed-image-v1/invoke"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    payload = {
        "inputText": text,
        "inputImage": image_b64,
        "embeddingConfig": {
            "outputEmbeddingLength": 1024  # Options: 256, 384, 1024
        },
    }

    print(f"Invoking amazon.titan-embed-image-v1 in {region}...")
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    result = response.json()
    return result["embedding"]


def get_text_embedding(text: str, region: str = REGION) -> list:
    """Generates text embeddings using Amazon Titan Text Embeddings V2."""
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/amazon.titan-embed-text-v2:0/invoke"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    payload = {
        "inputText": text,
        "dimensions": 1024,
        "normalize": True,
    }

    print(f"Invoking amazon.titan-embed-text-v2:0 in {region}...")
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    result = response.json()
    return result["embedding"]


if __name__ == "__main__":
    image_url = "https://assets.ngc.nvidia.com/products/api-catalog/nemotron-rerank-vl/example1.jpg"
    text_query = "AI enables robots to perceive, plan, and act autonomously."

    print("=== Running Multimodal Embedding (Text + Image) on Amazon Bedrock ===")
    embedding = get_multimodal_embedding(image_source=image_url, text=text_query)

    print("\n[SUCCESS] Generated Embedding Vector:")
    print(f"- Total Dimensions: {len(embedding)}")
    print(f"- First 10 values:  {embedding[:10]}")
    print(f"- Full vector:      {embedding}")
