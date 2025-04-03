"""
Workflow for curriculum generation.
"""
from typing import List, Dict, Any, Optional

from osyllabi.utils.log import log, with_context
from osyllabi.ai.client import OllamaClient
from osyllabi.ai.prompts import PromptTemplate
from osyllabi.generator.resources import ResourceCollector


class CurriculumWorkflow:
    """
    Orchestrates the curriculum generation workflow.
    
    This class coordinates the different components needed to generate a complete
    curriculum, including resource collection, content generation, and assembly.
    """
    
    def __init__(
        self,
        topic: str,
        skill_level: str = "Beginner",
        client: Optional[OllamaClient] = None,
        model: str = "llama2",
        temperature: float = 0.7
    ):
        """
        Initialize the workflow.
        
        Args:
            topic: Main curriculum topic
            skill_level: Target skill level
            client: Optional OllamaClient instance (creates one if None)
            model: Model to use for generation
            temperature: Temperature for generation
        """
        self.topic = topic
        self.skill_level = skill_level
        self.client = client if client else OllamaClient()
        self.model = model
        self.temperature = temperature
        self.resource_collector = ResourceCollector()
        
        # Store generated components
        self.components = {
            "overview": "",
            "learning_path": "",
            "resources": "",
            "projects": ""
        }
        
        log.info(f"Initialized curriculum workflow for topic: {topic}")
    
    def collect_resources(self, links: List[str], source_paths: List[str]) -> Dict[str, Any]:
        """
        Collect resources for curriculum generation.
        
        Args:
            links: List of URLs to use as resources
            source_paths: List of local file/directory paths
            
        Returns:
            Dict of collected resources
        """
        with with_context(topic=self.topic):
            log.info(f"Collecting resources for topic: {self.topic}")
            resources = self.resource_collector.collect(links, source_paths)
            log.info(f"Collected {len(resources)} resources")
            return resources
            
    def generate_overview(self) -> str:
        """
        Generate curriculum overview.
        
        Returns:
            Generated overview content
        """
        log.info(f"Generating curriculum overview for topic: {self.topic}")
        
        prompt_template = PromptTemplate.from_preset("overview")
        prompt = prompt_template.format(topic=self.topic, skill_level=self.skill_level)
        
        overview = self.client.generate(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature
        )
        
        self.components["overview"] = overview
        return overview
    
    def generate_learning_path(self) -> str:
        """
        Generate learning path.
        
        Returns:
            Generated learning path content
        """
        log.info(f"Generating learning path for topic: {self.topic}")
        
        prompt_template = PromptTemplate.from_preset("learning_path")
        prompt = prompt_template.format(topic=self.topic, skill_level=self.skill_level)
        
        learning_path = self.client.generate(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature
        )
        
        self.components["learning_path"] = learning_path
        return learning_path
    
    def generate_resources_section(self, links: List[str]) -> str:
        """
        Generate resources section.
        
        Args:
            links: List of URLs to include
            
        Returns:
            Generated resources content
        """
        log.info(f"Generating resources section for topic: {self.topic}")
        
        links_str = "\n".join([f"- {link}" for link in links]) if links else "None provided"
        
        prompt_template = PromptTemplate.from_preset("resources")
        prompt = prompt_template.format(
            topic=self.topic, 
            skill_level=self.skill_level,
            links=links_str
        )
        
        resources = self.client.generate(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature
        )
        
        self.components["resources"] = resources
        return resources
    
    def generate_projects(self, number_of_projects: int = 3) -> str:
        """
        Generate projects or exercises.
        
        Args:
            number_of_projects: Number of projects to generate
            
        Returns:
            Generated projects content
        """
        log.info(f"Generating {number_of_projects} projects for topic: {self.topic}")
        
        prompt_template = PromptTemplate.from_preset("projects")
        prompt = prompt_template.format(
            topic=self.topic,
            skill_level=self.skill_level,
            number_of_projects=number_of_projects
        )
        
        projects = self.client.generate(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature
        )
        
        self.components["projects"] = projects
        return projects
    
    def generate_full_curriculum(self, links: List[str], source_paths: List[str]) -> str:
        """
        Generate a complete curriculum.
        
        Args:
            links: List of URLs to include as resources
            source_paths: List of local file/directory paths
            
        Returns:
            Complete generated curriculum content
        """
        with with_context(topic=self.topic, skill_level=self.skill_level):
            log.info(f"Generating full curriculum for: {self.topic} ({self.skill_level})")
            
            # Step 1: Collect resources
            resources = self.collect_resources(links, source_paths)
            
            # Step 2: Generate each component
            overview = self.generate_overview()
            learning_path = self.generate_learning_path()
            resources_section = self.generate_resources_section(links)
            projects = self.generate_projects()
            
            # Step 3: Assemble the complete curriculum
            curriculum_content = f"""
# {self.topic} Curriculum

## Overview
{overview}

## Learning Path
{learning_path}

## Resources
{resources_section}

## Projects and Exercises
{projects}
"""
            
            log.info(f"Generated complete curriculum for: {self.topic}")
            return curriculum_content.strip()
