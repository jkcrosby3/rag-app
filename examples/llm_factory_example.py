"""
Example demonstrating how to use the LLM factory to switch between different LLM providers.
"""
import os
from dotenv import load_dotenv
from src.llm import get_llm_client, get_available_llm_types

def main():
    # Load environment variables
    load_dotenv()
    
    # Get available LLM types
    print(f"Available LLM types: {get_available_llm_types()}")
    
    # Example 1: Using Claude (default)
    print("\n--- Testing Claude LLM ---")
    claude_client = get_llm_client(
        "claude",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model_name="claude-3-5-sonnet-20241022"
    )
    
    response = claude_client.generate_response_direct(
        system_prompt="You are a helpful assistant.",
        user_message="Hello, how are you?"
    )
    print(f"Claude response: {response}")
    
    # Example 2: Using a different LLM (if available)
    # This is just an example - you would need to implement the OpenAI client
    if 'openai' in get_available_llm_types():
        print("\n--- Testing OpenAI LLM ---")
        try:
            openai_client = get_llm_client(
                "openai",
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="gpt-4"
            )
            response = openai_client.generate_response_direct(
                system_prompt="You are a helpful assistant.",
                user_message="Hello, how are you?"
            )
            print(f"OpenAI response: {response}")
        except Exception as e:
            print(f"Error using OpenAI client: {e}")
    
    # Example 3: Using the RAG system with a custom LLM
    print("\n--- Using RAG System with Custom LLM ---")
    from src.rag_system import RAGSystem
    
    # Initialize RAG system with Claude
    rag = RAGSystem(
        llm_type="claude",
        llm_api_key=os.getenv("ANTHROPIC_API_KEY"),
        llm_model_name="claude-3-5-sonnet-20241022"
    )
    
    # Query the RAG system
    result = rag.query("What is the capital of France?")
    print(f"RAG response: {result['response']}")

if __name__ == "__main__":
    main()
