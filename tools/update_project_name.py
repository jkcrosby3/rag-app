#!/usr/bin/env python3
"""
Update project name references from 'rag' to 'rag-app'.
"""
import os
from pathlib import Path


def update_file(file_path: Path):
    """Update references in a file."""
    with open(file_path, "r") as f:
        content = f.read()

    # Update specific patterns
    replacements = [
        ("cd rag\n", "cd rag-app\n"),
        ("cluster.name: rag-app-app-", "cluster.name: rag-app-app-app-"),
        ("node.name: rag-app-app-", "node.name: rag-app-app-app-"),
        ("ELASTICSEARCH_INDEX=rag-app-app-", "ELASTICSEARCH_INDEX=rag-app-app-app-"),
        ("/rag-app/", "/rag-app/"),
        ("rag-app/", "rag-app/"),
    ]

    new_content = content
    for old, new in replacements:
        new_content = new_content.replace(old, new)

    if new_content != content:
        print(f"Updating {file_path}")
        with open(file_path, "w") as f:
            f.write(new_content)


def main():
    """Update project name references."""
    root = Path("/Users/justinlawyer/Projects/rag-app")

    # Files to process
    extensions = {".md", ".py", ".yml", ".txt"}

    for ext in extensions:
        for file_path in root.rglob(f"*{ext}"):
            if not any(p.startswith(".") for p in file_path.parts):  # Skip hidden dirs
                update_file(file_path)


if __name__ == "__main__":
    main()
