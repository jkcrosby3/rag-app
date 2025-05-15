#!/usr/bin/env python3
"""
Convert markdown files to Jupyter notebooks and vice versa.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_jupytext():
    """Check if jupytext is installed."""
    try:
        subprocess.run(["jupytext", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: jupytext not found. Installing...")
        subprocess.run(["pip", "install", "jupytext"], check=True)


def convert_to_notebook(md_file: Path, execute: bool = False):
    """Convert markdown to notebook."""
    nb_file = md_file.with_suffix(".ipynb")
    print(f"Converting {md_file} to {nb_file}")

    # Convert to notebook
    subprocess.run(["jupytext", "--to", "notebook", str(md_file)], check=True)

    if execute:
        print(f"Executing {nb_file}")
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--execute",
                "--to",
                "notebook",
                "--inplace",
                str(nb_file),
            ],
            check=True,
        )


def convert_to_markdown(nb_file: Path):
    """Convert notebook to markdown."""
    md_file = nb_file.with_suffix(".md")
    print(f"Converting {nb_file} to {md_file}")
    subprocess.run(["jupytext", "--to", "md", str(nb_file)], check=True)


def setup_paired_mode(file: Path):
    """Set up paired mode for a file."""
    print(f"Setting up paired mode for {file}")
    subprocess.run(
        ["jupytext", "--set-formats", "ipynb,md:myst", str(file)], check=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert between markdown and Jupyter notebooks"
    )
    parser.add_argument("files", nargs="+", type=Path, help="Files to convert")
    parser.add_argument(
        "--to", choices=["notebook", "markdown"], required=True, help="Target format"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Execute notebooks after conversion"
    )
    parser.add_argument(
        "--paired", action="store_true", help="Set up paired mode for automatic syncing"
    )

    args = parser.parse_args()

    # Check dependencies
    check_jupytext()

    # Process each file
    for file in args.files:
        if not file.exists():
            print(f"Error: {file} does not exist")
            continue

        try:
            if args.to == "notebook":
                convert_to_notebook(file, args.execute)
            else:
                convert_to_markdown(file)

            if args.paired:
                setup_paired_mode(file)

        except subprocess.CalledProcessError as e:
            print(f"Error converting {file}: {e}")
            continue


if __name__ == "__main__":
    main()
