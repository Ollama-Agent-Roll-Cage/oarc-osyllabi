"""
Agent integration for RAG-enhanced curriculum generation.

This module provides base classes and utilities for incorporating RAG
capabilities into curriculum generation agents.
"""
from typing import List, Dict, Any, Optional, Union

from osyllabi.utils.log import log
from osyllabi.rag.engine import RAGEngine
from osyllabi.rag.context import ContextAssembler
from osyllabi.rag.query import QueryFormulator


class RAGEnhancedAgent:
    """
    Base class for agents with RAG enhancement capabilities.
    
    This class provides common functionality for agents to use RAG
    for context enrichment in their generation tasks.
    """
    
    def __init__(
        self,
        name: str,
        rag_engine: Optional[RAGEngine] = None,
        context_assembler: Optional[ContextAssembler] = None,
        query_formulator: Optional[QueryFormulator] = None
    ):
        """
        Initialize the RAG-enhanced agent.
        
        Args:
            name: Agent name
            rag_engine: Optional RAG engine (can be set later)
            context_assembler: Optional context assembler (created if None)
            query_formulator: Optional query formulator (created if None)
        """
        self.name = name
        self.rag_engine = rag_engine
        self.context_assembler = context_assembler or ContextAssembler()
        self.query_formulator = query_formulator or QueryFormulator()
        self.results = {}
        
    def set_rag_engine(self, rag_engine: RAGEngine) -> None:
        """
        Set the RAG engine for this agent.
        
        Args:
            rag_engine: RAG engine to use
        """
        self.rag_engine = rag_engine
        
    def retrieve_context(
        self,
        topic: str,
        query_type: str,
        skill_level: str = "Beginner",
        additional_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        threshold: float = 0.0,
        deduplicate: bool = True
    ) -> str:
        """
        Retrieve and format context for generation.
        
        Args:
            topic: Main topic
            query_type: Type of query (learning_path, resources, etc.)
            skill_level: Target skill level
            additional_context: Optional additional context
            top_k: Maximum number of results
            threshold: Minimum similarity threshold
            deduplicate: Whether to deduplicate results
            
        Returns:
            str: Assembled context ready for prompt inclusion
            
        Raises:
            ValueError: If RAG engine is not set
        """
        if not self.rag_engine:
            log.warning("No RAG engine set for agent, returning empty context")
            return ""
            
        # Formulate the query
        query = self.query_formulator.formulate_query(
            topic=topic,
            query_type=query_type,
            skill_level=skill_level,
            additional_context=additional_context
        )
        
        # Retrieve relevant chunks
        log.debug(f"Agent {self.name} retrieving context for: {query}")
        results = self.rag_engine.retrieve(query, top_k=top_k, threshold=threshold)
        
        # Assemble context
        context = self.context_assembler.assemble_context(
            retrieved_chunks=results,
            query=query,
            deduplicate=deduplicate
        )
        
        log.debug(f"Agent {self.name} retrieved {len(results)} chunks, context length: {len(context)}")
        return context
    
    def create_enhanced_prompt(
        self,
        base_prompt: str,
        topic: str,
        query_type: str,
        skill_level: str = "Beginner",
        top_k: int = 5
    ) -> str:
        """
        Create a RAG-enhanced prompt by adding retrieved context.
        
        Args:
            base_prompt: The original prompt template
            topic: Main topic
            query_type: Type of query
            skill_level: Target skill level
            top_k: Maximum number of results to include
            
        Returns:
            str: Enhanced prompt with context
        """
        # Get context
        context = self.retrieve_context(
            topic=topic,
            query_type=query_type,
            skill_level=skill_level,
            top_k=top_k
        )
        
        # If we got meaningful context, add it to the prompt
        if context:
            enhanced_prompt = (
                f"{base_prompt}\n\n"
                f"Use the following relevant information to improve your response:\n\n"
                f"{context}"
            )
            return enhanced_prompt
        
        # If no context, return the original prompt
        return base_prompt
