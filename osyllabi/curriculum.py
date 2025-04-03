"""
Curriculum generation functionality.
"""
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


class Curriculum:
    """Represents a generated curriculum."""
    
    def __init__(self, topic: str, title: Optional[str] = None):
        self.topic = topic
        self.title = title or f"{topic} Curriculum"
        self.content = ""
    
    def export(self, path: str, fmt: str = 'md') -> None:
        """
        Export the curriculum to the specified path in the given format.
        
        Args:
            path: The file path to export to
            fmt: The format to export as (md, pdf, html, docx)
        """
        # Convert path to Path object if it's a string
        if isinstance(path, str):
            path = Path(path)
            
        # Ensure the directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle different export formats
        if fmt == 'md':
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.content)
        else:
            # For other formats, additional dependencies might be needed
            raise NotImplementedError(f"Export to {fmt} is not yet implemented")


class CurriculumGenerator:
    """Generates curriculum content based on various inputs."""
    
    def __init__(
        self, 
        topic: str, 
        title: Optional[str] = None,
        skill_level: str = "Beginner",  # Keeping parameter name the same for backward compatibility
        links: List[str] = None,
        source: List[str] = None
    ):
        """
        Initialize a new curriculum generator.
        
        Args:
            topic: The main topic for the curriculum
            title: Optional title (defaults to "{topic} Curriculum")
            skill_level: Target skill level (Beginner, Intermediate, Advanced, Expert)
            links: List of URLs to include as resources
            source: Source files or directories to include
        """
        self.topic = topic
        self.title = title
        self.skill_level = skill_level  # Internal variable name remains the same
        self.links = links or []
        self.source = source or ["."]
        
    def create(self) -> Curriculum:
        """
        Generate a curriculum based on the configured parameters.
        
        Returns:
            Curriculum: The generated curriculum
        """
        curriculum = Curriculum(self.topic, self.title)
        
        # Generate basic structure (placeholder implementation)
        curriculum.content = f"""# {curriculum.title}

## Overview
A curriculum for learning about {self.topic} at the {self.skill_level} level.

## Resources
"""
        
        # Add links
        if self.links:
            curriculum.content += "\n### External Links\n"
            for link in self.links:
                curriculum.content += f"- {link}\n"
        
        return curriculum
