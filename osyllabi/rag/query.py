"""
Query formulation for effective RAG retrieval.

This module provides utilities to generate and optimize queries for
the RAG system to improve retrieval quality.
"""
from typing import List, Dict, Any, Optional, Union

from osyllabi.utils.log import log


class QueryFormulator:
    """
    Formulates effective queries for RAG retrieval.
    
    This class provides methods to generate queries that will
    yield the most relevant results from the vector database.
    """
    
    def __init__(self, default_modifiers: Optional[Dict[str, str]] = None):
        """
        Initialize the query formulator.
        
        Args:
            default_modifiers: Optional dict of topic type to query modifier templates
        """
        self.default_modifiers = default_modifiers or {
            "learning_path": "detailed curriculum structure for {topic} at {skill_level} level",
            "resources": "recommended learning resources for {topic} at {skill_level} level",
            "projects": "practical projects for learning {topic} at {skill_level} level",
            "overview": "comprehensive introduction to {topic} for {skill_level}s"
        }
    
    def formulate_query(
        self,
        topic: str,
        query_type: str,
        skill_level: str = "Beginner",
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Formulate a query for effective RAG retrieval.
        
        Args:
            topic: The main topic of interest
            query_type: Type of query (learning_path, resources, projects, overview)
            skill_level: Target skill level
            additional_context: Optional additional context to include
            
        Returns:
            str: Formulated query
        """
        # Get base template
        template = self.default_modifiers.get(
            query_type.lower(), 
            "information about {topic} for {skill_level}s"
        )
        
        # Format with basic information
        query = template.format(topic=topic, skill_level=skill_level)
        
        # Add additional context if provided
        if additional_context:
            context_parts = []
            for key, value in additional_context.items():
                if isinstance(value, str) and value.strip():
                    context_parts.append(f"{key}: {value}")
            
            if context_parts:
                query += " " + " ".join(context_parts)
        
        log.debug(f"Formulated query: {query}")
        return query
    
    def formulate_multi_queries(
        self,
        topic: str,
        skill_level: str = "Beginner",
        query_types: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Formulate multiple queries for different aspects of the curriculum.
        
        Args:
            topic: The main topic of interest
            skill_level: Target skill level
            query_types: List of query types to generate (defaults to all)
            
        Returns:
            Dict[str, str]: Dictionary of query_type to formulated query
        """
        query_types = query_types or list(self.default_modifiers.keys())
        
        result = {}
        for query_type in query_types:
            query = self.formulate_query(topic, query_type, skill_level)
            result[query_type] = query
            
        return result
    
    def expand_query(self, query: str, subtopics: List[str]) -> List[str]:
        """
        Expand a query into multiple related queries.
        
        Args:
            query: Base query
            subtopics: List of subtopics to include
            
        Returns:
            List[str]: List of expanded queries
        """
        expanded = [query]  # Start with the base query
        
        # Add a query for each subtopic
        for subtopic in subtopics:
            expanded.append(f"{query} focused on {subtopic}")
            
        return expanded
