"""
Conversation Manager for RAG System

This module provides conversation state tracking and management for multi-turn interactions,
enabling the system to ask clarifying questions and maintain context.
"""
import json
import time
from typing import Dict, List, Optional, Any, Tuple

class ConversationManager:
    """Manages conversation state and handles clarifying questions."""
    
    def __init__(self, llm_client, max_history_turns: int = 5):
        """Initialize the conversation manager.
        
        Args:
            llm_client: The LLM client used for generating responses
            max_history_turns: Maximum number of conversation turns to maintain in history
        """
        self.llm_client = llm_client
        self.max_history_turns = max_history_turns
    
    def analyze_query_ambiguity(self, query: str, conversation_history: List[Dict]) -> Tuple[bool, Optional[str]]:
        """Analyze if a query is ambiguous and needs clarification.
        
        Args:
            query: The user's query
            conversation_history: List of previous conversation turns
            
        Returns:
            Tuple containing:
            - Boolean indicating if clarification is needed
            - Optional clarifying question to ask the user
        """
        # Create a system prompt for ambiguity detection
        system_prompt = """
        You are an assistant that determines if a user query is ambiguous in the context of a conversation about the Great Depression.
        Analyze the query and determine if it needs clarification before a complete answer can be provided.
        If clarification is needed, provide a specific clarifying question to ask the user.
        If no clarification is needed, respond with "NO_CLARIFICATION_NEEDED".
        
        Consider the conversation history and current query context when making your determination.
        """
        
        # Format conversation history for the LLM
        formatted_history = self._format_conversation_history(conversation_history)
        
        # Create the user message with conversation context
        user_message = f"""
        Conversation history:
        {formatted_history}
        
        Current query: {query}
        
        Is this query ambiguous? If yes, what specific clarifying question should I ask?
        """
        
        # Get the LLM's analysis
        response = self.llm_client.generate_response_direct(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.3,  # Low temperature for more deterministic responses
            max_tokens=200
        )
        
        # Check if clarification is needed
        if "NO_CLARIFICATION_NEEDED" in response:
            return False, None
        else:
            # Extract the clarifying question from the response
            # The response should contain the clarifying question directly
            return True, response.strip()
    
    def generate_response_with_context(
        self,
        query: str,
        retrieved_documents: List[Dict],
        conversation_history: List[Dict],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate a response with conversation context.
        
        Args:
            query: The user's query
            retrieved_documents: Relevant documents for the query
            conversation_history: Previous conversation turns
            system_prompt: Optional system prompt
            temperature: Controls randomness in generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        # Format conversation history
        formatted_history = self._format_conversation_history(conversation_history)
        
        # Create a default system prompt if none provided
        if system_prompt is None:
            system_prompt = """
            You are an expert assistant specializing in the Great Depression.
            Provide accurate, comprehensive answers based on the retrieved documents.
            Use the conversation history to maintain context and provide coherent responses.
            """
        
        # Enhance the system prompt with conversation awareness
        enhanced_system_prompt = f"""
        {system_prompt}
        
        Conversation history:
        {formatted_history}
        """
        
        # Generate the response with context
        response = self.llm_client.generate_response(
            query=query,
            retrieved_documents=retrieved_documents,
            system_prompt=enhanced_system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response
    
    def _format_conversation_history(self, conversation_history: List[Dict]) -> str:
        """Format conversation history for inclusion in prompts.
        
        Args:
            conversation_history: List of conversation turns
            
        Returns:
            Formatted conversation history as a string
        """
        if not conversation_history:
            return "No previous conversation."
        
        # Limit history to the most recent turns
        recent_history = conversation_history[-self.max_history_turns:]
        
        formatted = []
        for i, turn in enumerate(recent_history):
            user_query = turn.get('query', '')
            system_response = turn.get('response', '')
            
            formatted.append(f"Turn {i+1}:")
            formatted.append(f"User: {user_query}")
            formatted.append(f"Assistant: {system_response}")
            formatted.append("")
        
        return "\n".join(formatted)
