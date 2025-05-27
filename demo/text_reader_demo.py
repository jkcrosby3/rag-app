"""Demo script for the text reader functionality using Alice in Wonderland."""
import logging
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import from src
from src.document_processing.reader import TextReader

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

 
def main():
    """Run the text reader demo using Alice in Wonderland."""
    # Path to Alice in Wonderland text file
    book_path = Path("data/books/alice_in_wonderland.txt")

    # Initialize reader
    reader = TextReader()

    # Read and process the file
    result = reader.read_file(book_path)

    # Display results
    print("\nBook Information:")
    print("----------------")
    print("Title: Alice's Adventures in Wonderland")
    print("Author: Lewis Carroll")
    print("Source: Project Gutenberg " "(https://gutenberg.org/cache/epub/11/pg11.txt)")

    print("\nFirst 500 characters of content:")
    print("------------------------------")
    # Remove the BOM character if present and handle encoding issues
    text = result["text"]
    if text.startswith('\ufeff'):
        text = text[1:]
    try:
        print(text[:500] + "...")
    except UnicodeEncodeError:
        print("[Content contains characters that cannot be displayed in the current console encoding]")
        print(text[:500].encode('ascii', 'replace').decode('ascii') + "...")

    print("\nMetadata:")
    print("--------")
    for key, value in result["metadata"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
