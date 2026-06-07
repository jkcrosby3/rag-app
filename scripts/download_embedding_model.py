#!/usr/bin/env python3
"""
Download the embedding model manually to bypass SSL issues.
"""
import os
import ssl
import urllib.request
from pathlib import Path

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Model files to download
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BASE_URL = f"https://huggingface.co/{MODEL_NAME}/resolve/main"

FILES_TO_DOWNLOAD = [
    "config.json",
    "tokenizer_config.json",
    "vocab.txt",
    "tokenizer.json",
    "special_tokens_map.json",
    "modules.json",
    "config_sentence_transformers.json",
    "pytorch_model.bin",
    "1_Pooling/config.json",
]

# Cache directory
cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "main"
cache_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading model to: {cache_dir}")

for file_name in FILES_TO_DOWNLOAD:
    url = f"{BASE_URL}/{file_name}"
    output_path = cache_dir / file_name
    
    # Create subdirectories if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"✓ {file_name} (already exists)")
        continue
    
    try:
        print(f"Downloading {file_name}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, output_path)
        print("✓")
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n✅ Model download complete!")
print(f"Model location: {cache_dir}")
