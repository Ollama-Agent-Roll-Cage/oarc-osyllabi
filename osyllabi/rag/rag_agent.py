"""
Agent integration for RAG-enhanced curriculum generation.

This module provides base classes and utilities for incorporating RAG
capabilities into curriculum generation agents.
"""
from typing import Dict, Any, Optional, List, Union, Tuple
import time

from osyllabi.utils.log import log
from osyllabi.ai.agents.agent import Agent
from osyllabi.rag.engine import RAGEngine
from osyllabi.rag.context import ContextAssembler
from osyllabi.rag.query import QueryFormulator
from osyllabi.ai.client import OllamaClient
from osyllabi.rag.monitor import RAGMonitor


class RAGAgent(Agent):
    """
    Agent with RAG enhancement capabilities.
    
    This class extends the base Agent with RAG capabilities
    for context enrichment in generation tasks.
    """
    
    def __init__(
        self,
        name: str,
        model: str = "llama3.1:latest",
        rag_engine: Optional[RAGEngine] = None,
        context_assembler: Optional[ContextAssembler] = None,
        query_formulator: Optional[QueryFormulator] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        monitor_performance: bool = True
    ):
        """
        Initialize the RAG-enhanced agent.
        
        Args:
            name: Agent name
            model: Model to use for generation
            rag_engine: Optional RAG engine (can be set later)
            context_assembler: Optional context assembler (created if None)
            query_formulator: Optional query formulator (created if None)
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
            monitor_performance: Whether to track performance metrics
        """
        # Initialize base agent
        super().__init__(name, model, temperature, max_tokens)
        
        # RAG-specific components
        self.rag_engine = rag_engine
        self.context_assembler = context_assembler or ContextAssembler()
        self.query_formulator = query_formulator or QueryFormulator()
        self.client = OllamaClient()
        
        # Performance monitoring
        self.monitor_performance = monitor_performance
        if monitor_performance:
            self.monitor = RAGMonitor()
        
        # Context management
        self.context_cache = {}
        self.retrieval_stats = {
            "total_retrievals": 0,
            "total_chunks": 0,
            "hit_rate": 0.0,
            "avg_context_length": 0
        }
        
    def set_rag_engine(self, rag_engine: RAGEngine) -> None:
        """
        Set the RAG engine for this agent.
        
        Args:
            rag_engine: RAG engine to use
        """
        self.rag_engine = rag_engine
        self.log_activity(f"RAG engine set: {rag_engine.run_id}")
        
    def retrieve_context(
        self,
        topic: str,
        query_type: str,
        skill_level: str = "Beginner",
        additional_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        threshold: float = 0.0,
        deduplicate: bool = True,
        use_cache: bool = True,
        expand_subtopics: bool = False
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
            use_cache: Whether to use cached context when available
            expand_subtopics: Whether to expand query with subtopics
            
        Returns:
            str: Assembled context ready for prompt inclusion
            
        Raises:
            ValueError: If RAG engine is not set
        """
        start_time = time.time()
        
        # Check for RAG engine
        if not self.rag_engine:
            self.log_activity("No RAG engine set, returning empty context", "warning")
            return ""
            
        # Check cache if enabled
        cache_key = f"{topic}:{query_type}:{skill_level}"
        if use_cache and cache_key in self.context_cache:
            self.log_activity(f"Using cached context for {cache_key}", "debug")
            return self.context_cache[cache_key]
            
        # Start retrieval monitoring if enabled
        retrieval_id = None
        if self.monitor_performance:
            retrieval_id = self.monitor.start_retrieval()
            
        # Formulate the query
        query = self.query_formulator.formulate_query(
            topic=topic,
            query_type=query_type,
            skill_level=skill_level,
            additional_context=additional_context
        )
        
        # Get results from multiple queries if expanding subtopics
        all_results = []
        if expand_subtopics:
            # Extract potential subtopics
            subtopics = self.query_formulator._extract_potential_subtopics(topic)[:3]  # Limit to top 3
            
            # Get base results
            self.log_activity(f"Retrieving context for base query: {query}", "debug")
            base_results = self.rag_engine.retrieve(
                query, 
                top_k=max(2, top_k // 2),  # Allocate portion to base query
                threshold=threshold
            )
            all_results.extend(base_results)
            
            # Get subtopic results
            remaining_results = max(1, top_k - len(base_results))
            per_subtopic = max(1, remaining_results // len(subtopics))
            
            for subtopic in subtopics:
                subtopic_query = f"{query} focused on {subtopic}"
                self.log_activity(f"Retrieving context for subtopic: {subtopic}", "debug")
                
                subtopic_results = self.rag_engine.retrieve(
                    subtopic_query,
                    top_k=per_subtopic,
                    threshold=threshold
                )
                all_results.extend(subtopic_results)
        else:
            # Standard single query retrieval
            self.log_activity(f"Retrieving context for: {query}", "debug")
            all_results = self.rag_engine.retrieve(
                query, 
                top_k=top_k,
                threshold=threshold
            )
        
        # Assemble context
        context = self.context_assembler.assemble_context(
            retrieved_chunks=all_results,
            query=query,
            deduplicate=deduplicate
        )
        
        # Update cache
        if use_cache:
            self.context_cache[cache_key] = context
            
        # Update retrieval stats
        self.retrieval_stats["total_retrievals"] += 1
        self.retrieval_stats["total_chunks"] += len(all_results)
        if len(all_results) > 0:
            self.retrieval_stats["hit_rate"] = (
                (self.retrieval_stats["hit_rate"] * (self.retrieval_stats["total_retrievals"] - 1) + 1) /
                self.retrieval_stats["total_retrievals"]
            )
        self.retrieval_stats["avg_context_length"] = (
            (self.retrieval_stats["avg_context_length"] * (self.retrieval_stats["total_retrievals"] - 1) + len(context)) /
            self.retrieval_stats["total_retrievals"]
        )
        
        # Record retrieval performance if monitoring is enabled
        if self.monitor_performance and retrieval_id is not None:
            duration = time.time() - start_time
            self.monitor.record_retrieval(retrieval_id, query, all_results, duration)
            
        self.log_activity(f"Retrieved {len(all_results)} chunks, context length: {len(context)}", "debug")
        return context
    
    def create_enhanced_prompt(
        self,
        base_prompt: str,
        topic: str,
        query_type: str,
        skill_level: str = "Beginner",
        top_k: int = 5,
        context_strategy: str = "standard",
        max_context_tokens: int = 1500
    ) -> str:
        """
        Create a RAG-enhanced prompt by adding retrieved context.
        
        Args:
            base_prompt: The original prompt template
            topic: Main topic
            query_type: Type of query
            skill_level: Target skill level
            top_k: Maximum number of results to include
            context_strategy: Strategy for context integration 
                              ("standard", "chain_of_thought", "vectorized")
            max_context_tokens: Maximum tokens to allocate for context
            
        Returns:
            str: Enhanced prompt with context
        """
        # Determine if we should expand subtopics based on query type
        expand_subtopics = query_type in ["learning_path", "overview", "resources"]
        
        # Get context with appropriate settings
        context = self.retrieve_context(
            topic=topic,
            query_type=query_type,
            skill_level=skill_level,
            top_k=top_k,
            expand_subtopics=expand_subtopics
        )
        
        # If we got meaningful context, add it to the prompt with the specified strategy
        if context:
            if context_strategy == "chain_of_thought":
                # Chain of thought prompting encourages stepwise reasoning
                enhanced_prompt = (
                    f"{base_prompt}\n\n"
                    f"Use the following relevant information to improve your response:\n\n"
                    f"{context}\n\n"
                    f"Now, step by step:\n"
                    f"1. Analyze the context information provided\n"
                    f"2. Extract the most relevant concepts for {topic}\n"
                    f"3. Organize these concepts in a logical learning progression\n"
                    f"4. Create a comprehensive {query_type} for {topic} at {skill_level} level"
                )
                
            elif context_strategy == "vectorized":
                # Vectorized approach organizes context into distinct sections
                enhanced_prompt = (
                    f"{base_prompt}\n\n"
                    f"RETRIEVED CONTEXT:\n{context}\n\n"
                    f"TASK DESCRIPTION:\n"
                    f"Create a comprehensive {query_type} for {topic} at {skill_level} level "
                    f"by integrating the most relevant information from the retrieved context.\n\n"
                    f"RESPONSE FORMAT:\n"
                    f"1. Introduction to {topic}\n"
                    f"2. Main components/concepts\n"
                    f"3. Detailed {query_type}\n"
                    f"4. Summary and recommendations"
                )
                
            else:  # standard
                # Standard approach simply adds context
                enhanced_prompt = (
                    f"{base_prompt}\n\n"
                    f"Use the following relevant information to improve your response:\n\n"
                    f"{context}"
                )
                
            self.log_activity(f"Enhanced prompt using {context_strategy} strategy", "debug")
            return enhanced_prompt
        
        # If no context, return the original prompt
        return base_prompt
        
    def generate(self, prompt: str) -> str:
        """
        Generate content based on a prompt using the specified model.
        
        Args:
            prompt: Prompt to generate from
            
        Returns:
            Generated content
            
        Raises:
            RuntimeError: If generation fails
        """
        self.log_activity(f"Generating content with model: {self.model}", "debug")
        
        generation_start = time.time()
        response = self.client.generate(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        generation_time = time.time() - generation_start
        
        # Store generation metrics
        self.store_metadata("last_generation_time", generation_time)
        self.store_metadata("last_prompt_length", len(prompt))
        self.store_metadata("last_response_length", len(response))
        
        self.log_activity(f"Generated {len(response)} chars in {generation_time:.2f}s", "debug")
        return response
        
    def process(self, input_data: Dict[str, Any]) -> str:
        """
        Process input data and produce output using RAG enhancement.
        
        Args:
            input_data: Dictionary containing at minimum:
                - 'topic': Main topic
                - 'query_type': Type of content to generate
                - 'base_prompt': Base prompt to enhance
                
        Returns:
            Generated content
            
        Raises:
            ValueError: If required fields are missing
        """
        process_start = time.time()
        
        # Validate input
        for field in ['topic', 'query_type', 'base_prompt']:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
                
        # Extract input
        topic = input_data['topic']
        query_type = input_data['query_type']
        base_prompt = input_data['base_prompt']
        skill_level = input_data.get('skill_level', 'Beginner')
        
        # Get optional parameters
        context_strategy = input_data.get('context_strategy', 'standard')
        top_k = input_data.get('top_k', 5)
        
        # Create enhanced prompt
        enhanced_prompt = self.create_enhanced_prompt(
            base_prompt=base_prompt,
            topic=topic,
            query_type=query_type,
            skill_level=skill_level,
            top_k=top_k,
            context_strategy=context_strategy
        )
        
        # Generate content
        self.log_activity(f"Processing {query_type} for {topic}", "info")
        generated_content = self.generate(enhanced_prompt)
        
        # Store the result with metadata
        self.store_result(query_type, generated_content)
        self.store_metadata("total_process_time", time.time() - process_start)
        
        return generated_content
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this agent.
        
        Returns:
            Dict[str, Any]: Performance metrics dictionary
        """
        metrics = {
            "retrieval_stats": self.retrieval_stats.copy(),
            "generation_metrics": {
                "last_generation_time": self.get_metadata("last_generation_time", 0),
                "last_prompt_length": self.get_metadata("last_prompt_length", 0),
                "last_response_length": self.get_metadata("last_response_length", 0),
                "total_process_time": self.get_metadata("total_process_time", 0)
            }
        }
        
        # Add monitor metrics if available
        if self.monitor_performance:
            metrics["rag_monitor"] = self.monitor.get_metrics()
            
        return metrics
    
    def clear_cache(self) -> None:
        """Clear the context cache to free memory and force fresh retrievals."""
        cache_size = len(self.context_cache)
        self.context_cache = {}
        self.log_activity(f"Cleared context cache ({cache_size} entries)", "debug")
