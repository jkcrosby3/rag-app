"""Try to get a list of available models from Anthropic API."""
import os
from dotenv import load_dotenv
import httpx

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")

# Try to hit the models endpoint
http_client = httpx.Client(verify=False)

headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

# Try different possible endpoints
endpoints = [
    "https://api.anthropic.com/v1/models",
    "https://api.anthropic.com/v1/complete",
]

print("Trying to find models endpoint...\n")

for endpoint in endpoints:
    try:
        print(f"Trying: {endpoint}")
        response = http_client.get(endpoint, headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}\n")
    except Exception as e:
        print(f"Error: {e}\n")

# Also try to see what the console says about your account
print("\nCheck your Anthropic console at: https://console.anthropic.com/settings/keys")
print("Look for:")
print("1. Any model access restrictions")
print("2. API tier (free vs paid)")
print("3. Try the 'Workbench' to test a model directly in the browser")
