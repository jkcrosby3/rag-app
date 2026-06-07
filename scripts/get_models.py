"""Get the full list of available models."""
import os
from dotenv import load_dotenv
import httpx
import json

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")

http_client = httpx.Client(verify=False)

headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

response = http_client.get("https://api.anthropic.com/v1/models", headers=headers)
data = response.json()

print("Available models:\n")
for model in data.get("data", []):
    print(f"ID: {model['id']}")
    print(f"  Display Name: {model['display_name']}")
    print(f"  Max Tokens: {model.get('max_tokens', 'N/A')}")
    print()
