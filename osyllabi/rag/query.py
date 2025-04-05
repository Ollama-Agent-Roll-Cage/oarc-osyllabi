"""
Query formulation for effective RAG retrieval.

This module provides utilities to generate and optimize queries for
the RAG system to improve retrieval quality.
"""
import re
from typing import List, Dict, Any, Optional

from osyllabi.utils.log import log
from llama_index.core.prompts import PromptTemplate as LlamaPromptTemplate


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
            "overview": "comprehensive introduction to {topic} for {skill_level}s",
            # New query types
            "fundamentals": "core concepts and foundational knowledge in {topic} for {skill_level}s",
            "advanced_topics": "advanced topics and concepts in {topic} for {skill_level}s",
            "best_practices": "best practices and standards for {topic} development",
            "industry_applications": "real-world industry applications of {topic}",
            "case_studies": "case studies and examples demonstrating {topic} principles",
            "learning_objectives": "key learning objectives and outcomes for {topic} at {skill_level} level"
        }
        
        # Topic-specific template overrides
        self.topic_specific_templates = {
            "programming": {
                "projects": "coding projects and exercises for {topic} at {skill_level} level",
                "resources": "programming resources, libraries, and documentation for {topic}"
            },
            "data_science": {
                "projects": "data analysis projects using {topic} techniques at {skill_level} level",
                "resources": "datasets, tools, and libraries for {topic} at {skill_level} level"
            },
            "design": {
                "projects": "design challenges and portfolio projects for {topic} at {skill_level} level",
                "resources": "design tools, inspiration sources, and guidelines for {topic}"
            }
        }
        
        # Common domain-specific terms by field to enhance queries
        self.domain_terms = {
            "programming": ["code", "syntax", "functions", "libraries", "debugging", "algorithms"],
            "machine_learning": ["models", "features", "training", "inference", "datasets", "accuracy"],
            "web_development": ["frontend", "backend", "API", "responsive", "frameworks", "deployment"],
            "data_science": ["analysis", "visualization", "statistics", "hypothesis", "correlation"],
            "design": ["user experience", "wireframes", "prototypes", "user interface", "accessibility"]
        }
        
        # Initialize LlamaIndex custom prompt templates
        self.llama_prompt_templates = {}
        self._setup_llama_templates()
    
    def _setup_llama_templates(self):
        """Set up LlamaIndex custom prompt templates for advanced querying."""
        try:
            # Create custom templates for different query types
            curriculum_template = """
            Given a pandas DataFrame about {topic} for {skill_level} level, 
            analyze the content and provide a structured curriculum with key 
            concepts, learning path, and recommendations.
            
            DataFrame schema: {schema}
            
            Query: {query_str}
            """
            
            resources_template = """
            Given a pandas DataFrame about {topic} for {skill_level} level,
            identify and list the most relevant learning resources, tools,
            and references.
            
            DataFrame schema: {schema}
            
            Query: {query_str}
            """
            
            self.llama_prompt_templates = {
                "curriculum": LlamaPromptTemplate(curriculum_template),
                "resources": LlamaPromptTemplate(resources_template)
            }
            
            log.info("Initialized LlamaIndex prompt templates")
        except Exception as e:
            log.warning(f"Failed to initialize LlamaIndex templates: {e}")
    
    def get_llama_prompt_template(self, query_type: str):
        """
        Get LlamaIndex prompt template for a specific query type.
        
        Args:
            query_type: Type of query
            
        Returns:
            LlamaPromptTemplate or None if not available
        """
        # Map our query types to template types
        template_map = {
            "learning_path": "curriculum",
            "resources": "resources",
            "projects": "curriculum",
            "overview": "curriculum"
        }
        
        template_type = template_map.get(query_type, "curriculum")
        return self.llama_prompt_templates.get(template_type)
    
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
        # Determine domain area for topic-specific templates
        domain = self._detect_domain(topic)
        
        # Get domain-specific template if available, otherwise use default
        if (domain and domain in self.topic_specific_templates and 
            query_type in self.topic_specific_templates[domain]):
            template = self.topic_specific_templates[domain][query_type]
        else:
            # Get base template
            template = self.default_modifiers.get(
                query_type.lower(), 
                "information about {topic} for {skill_level}s"
            )
        
        # Format with basic information
        query = template.format(topic=topic, skill_level=skill_level)
        
        # Add domain-specific terms to enhance query relevance
        if domain and domain in self.domain_terms:
            domain_keywords = " ".join(f"{term}" for term in self.domain_terms[domain][:3])
            query += f" including {domain_keywords}"
        
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
        
    def generate_subtopic_queries(self, topic: str, query_type: str, subtopics: List[str], skill_level: str = "Beginner") -> List[str]:
        """
        Generate queries for multiple subtopics of a main topic.
        
        Args:
            topic: Main topic
            query_type: Type of query
            subtopics: List of subtopics
            skill_level: Target skill level
            
        Returns:
            List[str]: Queries for each subtopic
        """
        queries = []
        base_query = self.formulate_query(topic, query_type, skill_level)
        
        # Generate a query for each subtopic
        for subtopic in subtopics:
            subtopic_query = f"{base_query} with focus on {subtopic}"
            queries.append(subtopic_query)
            
        return queries
        
    def create_hierarchical_queries(self, topic: str, skill_level: str = "Beginner") -> Dict[str, List[str]]:
        """
        Create hierarchically organized queries from general to specific.
        
        Args:
            topic: Main topic
            skill_level: Target skill level
            
        Returns:
            Dict[str, List[str]]: Hierarchy of queries
        """
        result = {
            "overview": [self.formulate_query(topic, "overview", skill_level)],
            "fundamentals": [self.formulate_query(topic, "fundamentals", skill_level)],
            "specific_topics": [],
            "applications": []
        }
        
        # Extract potential subtopics based on common patterns
        subtopics = self._extract_potential_subtopics(topic)
        
        # Add subtopic queries
        for subtopic in subtopics:
            result["specific_topics"].append(
                self.formulate_query(f"{subtopic} in {topic}", "learning_path", skill_level)
            )
            
        # Add application queries
        result["applications"].append(
            self.formulate_query(topic, "industry_applications", skill_level)
        )
        
        return result
    
    def _detect_domain(self, topic: str) -> Optional[str]:
        """
        Detect the domain area of a topic.
        
        Args:
            topic: Topic to analyze
            
        Returns:
            Optional[str]: Detected domain or None
        """
        topic_lower = topic.lower()
        
        # Simple keyword-based domain detection
        domain_keywords = {
            "programming": ["programming", "coding", "development", "software", "python", "java", "javascript"],
            "data_science": ["data science", "machine learning", "statistics", "analytics", "data analysis"],
            "web_development": ["web", "html", "css", "frontend", "backend", "fullstack"],
            "design": ["design", "ui", "ux", "user experience", "graphic", "visual"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                return domain
                
        return None
        
    def _extract_potential_subtopics(self, topic: str) -> List[str]:
        """
        Extract potential subtopics based on the main topic.
        
        Args:
            topic: Main topic
            
        Returns:
            List[str]: List of potential subtopics
        """
        domain = self._detect_domain(topic)
        
        if not domain:
            # Generic subtopics for unknown domains
            return ["fundamentals", "advanced concepts", "practical applications"]
            
        # Domain-specific subtopics
        domain_subtopics = {
            "programming": ["syntax", "data structures", "algorithms", "libraries", "frameworks", "testing"],
            "data_science": ["data cleaning", "visualization", "modeling", "evaluation", "deployment"],
            "web_development": ["frontend", "backend", "databases", "security", "deployment"],
            "design": ["research", "wireframing", "prototyping", "testing", "evaluation"]
        }
        
        return domain_subtopics.get(domain, ["fundamentals", "advanced concepts", "practical applications"])
        
    def optimize_query_for_retrieval(
        self, 
        query: str, 
        query_purpose: str = "semantic", 
        max_length: int = 100
    ) -> str:
        """
        Optimize a query for specific retrieval purposes.
        
        Args:
            query: Original query
            query_purpose: Purpose of optimization ("semantic", "keyword", "hybrid")
            max_length: Maximum length of optimized query
            
        Returns:
            str: Optimized query
        """
        if query_purpose == "keyword":
            # Extract and focus on keywords for keyword-based retrieval
            keywords = self._extract_keywords(query)
            return " ".join(keywords)[:max_length]
            
        elif query_purpose == "hybrid":
            # Combine original query with extracted keywords
            keywords = self._extract_keywords(query)
            keyword_str = " ".join(keywords)
            
            # Build hybrid query but respect max length
            hybrid = f"{query} {keyword_str}"
            if len(hybrid) > max_length:
                return hybrid[:max_length]
            return hybrid
            
        else:  # semantic - default
            # Just trim if too long
            if len(query) > max_length:
                return query[:max_length]
            return query
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text for query optimization.
        
        Args:
            text: Input text
            
        Returns:
            List[str]: Extracted keywords
        """
        # Simple keyword extraction - remove stopwords and keep content words
        stopwords = {'the', 'and', 'is', 'in', 'at', 'of', 'for', 'with', 'by', 'to', 'a', 'an'}
        words = re.findall(r'\b\w+\b', text.lower())
        content_words = [word for word in words if word not in stopwords and len(word) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_words = [word for word in content_words if not (word in seen or seen.add(word))]
        
        return unique_words
