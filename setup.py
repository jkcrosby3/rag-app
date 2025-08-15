"""Setup script for the rag-app package."""
from setuptools import setup, find_packages

setup(
    name="rag-app",
    version="0.1.0",
    description="Retrieval-Augmented Generation (RAG) application for document processing",
    author="RAG App Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "pdf": ["PyPDF2>=3.0.0"],
    },
)
