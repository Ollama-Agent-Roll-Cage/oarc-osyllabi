"""
Curriculum generation functionality.
"""
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Union
from pathlib import Path
from datetime import datetime


class Curriculum:
    """Represents a generated curriculum."""
    
    def __init__(self, topic: str, title: Optional[str] = None):
        """
        Initialize a new curriculum.
        
        Args:
            topic: The main subject of the curriculum
            title: Optional title (defaults to "{topic} Curriculum")
        """
        self.topic = topic
        self.title = title or f"{topic} Curriculum"
        self.content = ""
        self.created_at = datetime.now()
    
    def export(self, path: Union[str, Path], fmt: str = 'md') -> Path:
        """
        Export the curriculum to the specified path in the given format.
        
        Args:
            path: The file path to export to (directory or file)
            fmt: The format to export as (md, pdf, html, docx)
            
        Returns:
            Path: The full path to the exported file
            
        Raises:
            NotImplementedError: If export format is not supported yet
            IOError: For file system related errors
        """
        # Convert path to Path object if it's a string
        if isinstance(path, str):
            path = Path(path)
            
        # If path is a directory, create a filename based on the curriculum title
        if path.is_dir() or not path.suffix:
            # Convert title to valid filename
            safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in self.title)
            safe_title = safe_title.replace(" ", "_")
            
            # Create full path including the file name
            path = path / f"{safe_title}.{fmt}"
        
        # Ensure the directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Handle different export formats
            if fmt == 'md':
                with open(path, 'w', encoding='utf-8') as f:
                    # Add metadata at the top
                    metadata = f"---\ntitle: {self.title}\ntopic: {self.topic}\ncreated: {self.created_at.isoformat()}\n---\n\n"
                    f.write(metadata + self.content)
                    
                return path
            else:
                # For other formats, additional dependencies might be needed
                raise NotImplementedError(f"Export to {fmt} is not yet implemented")
        except IOError as e:
            print(f"Error exporting curriculum: {e}", file=sys.stderr)
            raise
            

class CurriculumGenerator:
    """Generates curriculum content based on various inputs."""
    
    def __init__(
        self, 
        topic: str, 
        title: Optional[str] = None,
        skill_level: str = "Beginner",
        links: Optional[List[str]] = None,
        source: Optional[List[str]] = None
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
        self.skill_level = skill_level
        self.links = links or []
        self.source = source or ["."]
        
    def create(self) -> Curriculum:
        """
        Generate a curriculum based on the configured parameters.
        
        Returns:
            Curriculum: The generated curriculum
        """
        curriculum = Curriculum(self.topic, self.title)
        
        # Generate basic structure
        curriculum.content = f"""# {curriculum.title}

## Overview
A curriculum for learning about {self.topic} at the {self.skill_level} level.

## Resources
"""
        
        # Add links
        if self.links:
            curriculum.content += "\n### External Links\n"
            for link in self.links:
                curriculum.content += f"- [{link}]({link})\n"
        
        # Add placeholder for content sections that would be filled by more advanced implementation
        curriculum.content += "\n## Learning Path\n\n*Curriculum content generation in progress...*\n"
        
        return curriculum
