"""
Curriculum generation workflow management.

This module provides the workflow for curriculum generation, including
resource collection, content generation, and assembly.
"""
import time
import importlib.util
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from osyllabi.utils.log import log
from osyllabi.utils.utils import check_for_ollama
from osyllabi.ai.client import OllamaClient
from osyllabi.ai.prompts import PromptTemplate
from osyllabi.generator.resources import ResourceCollectionManager


class CurriculumWorkflow:
    """
    Orchestrates the curriculum generation process.
    
    This class manages the end-to-end workflow for generating
    curriculum content, including resource collection,
    content generation, and assembly.
    """
    
    def __init__(
        self,
        topic: str,
        skill_level: str = "Beginner",
        client: Optional[OllamaClient] = None,
        model: str = "llama3",
        temperature: float = 0.7,
        use_langchain: bool = True
    ):
        """
        Initialize the curriculum workflow.
        
        Args:
            topic: Main topic for curriculum
            skill_level: Target skill level
            client: Optional Ollama client (created if None)
            model: Model to use for generation
            temperature: Temperature for generation
            use_langchain: Whether to use LangChain for enhanced processing if available
            
        Raises:
            RuntimeError: If Ollama is not available
        """
        self.topic = topic
        self.skill_level = skill_level
        
        # Verify Ollama is available
        check_for_ollama(raise_error=True)
        
        # Initialize the AI client
        self.client = client or OllamaClient()
        self.model = model
        self.temperature = temperature
        
        # Check if LangChain is available for enhanced processing
        self.langchain_available = False
        if use_langchain:
            self.langchain_available = importlib.util.find_spec("langchain") is not None
            if self.langchain_available:
                log.info("LangChain is available for enhanced content processing")
        
        # Initialize resource collection manager
        self.resource_manager = ResourceCollectionManager()
        
        # Templates for different curriculum sections - use the exact preset names from PromptTemplate
        self.templates = {
            "overview": PromptTemplate.from_preset("overview"),
            "learning_path": PromptTemplate.from_preset("learning_path"),
            "resources": PromptTemplate.from_preset("resources"),
            "projects": PromptTemplate.from_preset("projects")
        }
        
        # Start time for performance tracking
        self.start_time = time.time()
        
        log.info(f"Initialized curriculum workflow for topic: {topic}, level: {skill_level}")
    
    def collect_resources(
        self, 
        links: List[str], 
        source_paths: List[str]
    ) -> Dict[str, Any]:
        """
        Collect resources for curriculum generation.
        
        Args:
            links: List of URLs to use as resources
            source_paths: List of file/directory paths to use
            
        Returns:
            Collected resources
        """
        log.info(f"Collecting resources for {self.topic} curriculum")
        
        # Use resource manager to collect from various sources
        resources = self.resource_manager.collect_resources(
            urls=links,
            paths=source_paths
        )
        
        # Log collection statistics
        stats = resources.get("stats", {})
        log.info(f"Collected {len(resources.get('urls', {}))} web resources and " 
                f"{len(resources.get('files', {}))} file resources")
        
        return resources
    
    def generate_overview(self) -> str:
        """
        Generate curriculum overview.
        
        Returns:
            Generated overview
        """
        log.info(f"Generating curriculum overview for {self.topic}")
        
        # Format the prompt
        prompt = self.templates["overview"].format(
            topic=self.topic, 
            skill_level=self.skill_level
        )
        
        # Generate with Ollama
        start_time = time.time()
        overview = self.client.generate(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature
        )
        
        log.info(f"Generated overview in {time.time() - start_time:.2f}s")
        return overview
    
    def generate_learning_path(self, resources: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate learning path.
        
        Args:
            resources: Optional collected resources to enhance generation
            
        Returns:
            Generated learning path
        """
        log.info(f"Generating learning path for {self.topic}")
        
        # Format the prompt
        prompt = self.templates["learning_path"].format(
            topic=self.topic, 
            skill_level=self.skill_level
        )
        
        # Add context from resources if available
        context = ""
        if resources:
            if self.langchain_available and len(resources.get("urls", {})) + len(resources.get("files", {})) > 5:
                # Use LangChain for more sophisticated processing when many resources
                try:
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                    from langchain.chains.summarize import load_summarize_chain
                    
                    # Extract all content
                    all_texts = []
                    for url_data in resources.get("urls", {}).values():
                        all_texts.append(url_data.get("content", ""))
                    for file_data in resources.get("files", {}).values():
                        all_texts.append(file_data.get("content", ""))
                        
                    # Combine and chunk content
                    combined = "\n\n".join(all_texts)
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=100
                    )
                    chunks = text_splitter.split_text(combined)
                    
                    log.debug(f"Using LangChain to process {len(chunks)} resource chunks")
                    # Just use the first few chunks as context directly
                    context = "\n\n".join(chunks[:5])
                except Exception as e:
                    log.warning(f"Failed to use LangChain for processing: {e}")
                    # Fall back to standard processing
                    context = self.resource_manager.extract_context_for_prompt(resources, self.topic)
            else:
                context = self.resource_manager.extract_context_for_prompt(resources, self.topic)
                
            if context:
                prompt += f"\n\nUse the following context to enhance your learning path:\n{context}"
        
        # Generate with Ollama
        start_time = time.time()
        learning_path = self.client.generate(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature
        )
        
        log.info(f"Generated learning path in {time.time() - start_time:.2f}s")
        return learning_path
    
    def generate_resources_section(self, links: List[str]) -> str:
        """
        Generate resources section.
        
        Args:
            links: List of URLs to include
            
        Returns:
            Generated resources section
        """
        log.info(f"Generating resources section for {self.topic}")
        
        # Format the prompt
        prompt = self.templates["resources"].format(
            topic=self.topic, 
            skill_level=self.skill_level
        )
        
        # Add provided links
        if links:
            prompt += "\n\nInclude these specific resources in your section:\n"
            for link in links:
                prompt += f"- {link}\n"
        
        # Generate with Ollama
        start_time = time.time()
        resources_section = self.client.generate(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature
        )
        
        log.info(f"Generated resources section in {time.time() - start_time:.2f}s")
        return resources_section
    
    def generate_projects(self, number_of_projects: int = 3) -> str:
        """
        Generate projects or exercises.
        
        Args:
            number_of_projects: Number of projects to generate
            
        Returns:
            Generated projects section
        """
        log.info(f"Generating {number_of_projects} projects for {self.topic}")
        
        # Format the prompt
        prompt = self.templates["projects"].format(
            topic=self.topic, 
            skill_level=self.skill_level,
            number_of_projects=number_of_projects
        )
        
        # Generate with Ollama
        start_time = time.time()
        projects = self.client.generate(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature
        )
        
        log.info(f"Generated projects in {time.time() - start_time:.2f}s")
        return projects
    
    def generate_full_curriculum(
        self, 
        links: List[str], 
        source_paths: List[str]
    ) -> str:
        """
        Generate a complete curriculum.
        
        Args:
            links: List of URLs to include as resources
            source_paths: List of source files/directories
            
        Returns:
            Complete curriculum content
        """
        log.info(f"Generating full curriculum for {self.topic} at {self.skill_level} level")
        
        # Start building the curriculum
        curriculum = f"# {self.topic} Curriculum ({self.skill_level} Level)\n\n"
        
        # Collect resources
        resources = self.collect_resources(links, source_paths)
        
        # Generate each section
        overview = self.generate_overview()
        curriculum += f"## Overview\n\n{overview}\n\n"
        
        learning_path = self.generate_learning_path(resources)
        curriculum += f"## Learning Path\n\n{learning_path}\n\n"
        
        resources_section = self.generate_resources_section(links)
        curriculum += f"## Resources\n\n{resources_section}\n\n"
        
        projects = self.generate_projects()
        curriculum += f"## Projects\n\n{projects}\n\n"
        
        # Add a metadata section
        total_time = time.time() - self.start_time
        curriculum += (
            f"## Metadata\n\n"
            f"- Topic: {self.topic}\n"
            f"- Skill Level: {self.skill_level}\n"
            f"- Generation Time: {total_time:.2f} seconds\n"
            f"- Model: {self.model}\n"
            f"- Resources Used: {len(resources.get('urls', {})) + len(resources.get('files', {}))}\n"
        )
        
        log.info(f"Completed full curriculum generation in {total_time:.2f}s")
        return curriculum
