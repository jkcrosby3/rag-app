#!/usr/bin/env python3
"""
Hugging Face Login Helper - Manual Token Save

Bypasses authentication API by directly saving token to file.
This works when SSL verification fails on corporate networks.
"""

import os
from pathlib import Path

print("🔑 Manually saving Hugging Face token...")
print("   Bypassing SSL issues by writing token directly to file.\n")

# Get token from environment variable
token = os.getenv("HUGGINGFACE_TOKEN")
if not token:
    print("❌ Error: HUGGINGFACE_TOKEN environment variable not set")
    print("   Please set it with: $env:HUGGINGFACE_TOKEN='your_token_here'")
    exit(1)

# Token file location
token_dir = Path.home() / ".huggingface"
token_file = token_dir / "token"

try:
    # Create directory if it doesn't exist
    token_dir.mkdir(parents=True, exist_ok=True)
    
    # Write token to file
    token_file.write_text(token)
    
    print(f"✅ Token saved successfully!")
    print(f"   Location: {token_file}")
    print(f"\n   You can now use Hugging Face datasets without authentication issues.")
    print(f"   The download script will automatically find this token.\n")
    
except Exception as e:
    print(f"\n❌ Failed to save token: {e}")
