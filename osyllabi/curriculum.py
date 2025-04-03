"""
Curriculum generation functionality.
"""
import json
import argparse
from typing import List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime

from osyllabi.utils.log import log, with_context
from osyllabi.utils.const import SUCCESS, FAILURE
from osyllabi.utils.paths import get_output_directory, create_unique_file_path
from osyllabi.utils.decorators.factory import factory


@factory
class Curriculum:
    """Represents a generated curriculum."""
    
    def __init__(
        self, 
        topic: str = None,
        title: Optional[str] = None,
        skill_level: str = "Beginner",
        links: Optional[List[str]] = None,
        source: Optional[List[str]] = None,
        args: Optional[argparse.Namespace] = None
    ):
        """
        Initialize a new curriculum.
        
        Args:
            topic: The main topic/subject of the curriculum (required normally, but optional when args is provided)
            title: Optional title (defaults to "{topic} Curriculum")
            skill_level: Target skill level (Beginner, Intermediate, Advanced, Expert)
            links: List of URLs to include as resources
            source: Source files or directories to include
            args: Optional command-line arguments
            
        Raises:
            ValueError: If topic is empty or only whitespace and args is not provided
        """
        # Store initialization result to be returned by factory
        self._result = None
        
        # Process command-line args if provided
        if args is not None:
            self._result = self._process_args(args)
            return
            
        # Normal initialization
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty")
            
        self.topic = topic.strip()
        self.title = title.strip() if title and title.strip() else f"{self.topic} Curriculum"
        self.skill_level = skill_level
        self.links = links or []
        self.source = source or ["."]
        self.content = ""
        self.created_at = datetime.now()
        
        log.debug(f"Created new curriculum: {self.title}")
    
    def _process_args(self, args: argparse.Namespace) -> Tuple[int, Optional[Path]]:
        """
        Process command-line arguments and create/export a curriculum.
        
        Args:
            args: The parsed command line arguments
            
        Returns:
            Tuple containing (exit_code, output_path)
        """
        topic = args.topic.strip()
        if not topic:
            log.error("Error: Topic is required and cannot be empty")
            log.error(f"Run 'osyllabi create --help' for more information.")
            return FAILURE, None
            
        log.info(f"Creating curriculum on topic: {topic}")
        
        try:
            self.topic = topic
            self.title = args.title.strip() if args.title and args.title.strip() else f"{topic} Curriculum"
            self.skill_level = args.level
            self.links = args.links or []
            self.source = args.source or ["."]
            self.content = ""
            self.created_at = datetime.now()
            
            self.generate_content()
            
            if args.export_path:
                export_path = args.export_path
            else:
                output_dir = get_output_directory()
                filename = self.title
                export_path = create_unique_file_path(output_dir, filename, args.format)
            
            result_path = self.export(export_path, fmt=args.format)
            log.info(f"Curriculum exported to {result_path}")
            return SUCCESS, result_path
            
        except Exception as e:
            raise RuntimeError(f"Error creating curriculum: {e}")
        
    def generate_content(self) -> None:
        """Generate the content for the curriculum."""
        log.info(f"Generating curriculum content for topic: {self.topic}")
        
        with with_context(topic=self.topic, skill_level=self.skill_level):
            self.content = f"""
# {self.title}

## Overview
A curriculum for learning about {self.topic} at the {self.skill_level} level.

## Resources
"""
            
            # Add links
            if self.links:
                log.debug(f"Adding {len(self.links)} external links to curriculum")
                self.content += "\n### External Links\n"
                for link in self.links:
                    self.content += f"- [{link}]({link})\n"
            
            # Add placeholder for content sections that would be filled by more advanced implementation
            self.content += "\n## Learning Path\n\n*Curriculum content generation in progress...*\n"
            
            log.info(f"Generated curriculum content: {self.title}")
    
    def export(self, path: Union[str, Path], fmt: str = 'md') -> Path:
        """
        Export the curriculum to the specified path in the given format.
        
        Args:
            path: The file path to export to (directory or file)
            fmt: The format to export as (md, pdf, html, docx, json)
            
        Returns:
            Path: The full path to the exported file
            
        Raises:
            NotImplementedError: If export format is not supported yet
            IOError: For file system related errors
        """
        # Generate content if it hasn't been generated yet
        if not self.content:
            self.generate_content()
            
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
                    
                log.info(f"Exported curriculum to {path} in {fmt} format")
                return path
            elif fmt == 'json':
                data = {
                    "meta": {
                        "title": self.title,
                        "topic": self.topic,
                        "skill_level": self.skill_level,
                        "created": self.created_at.isoformat()
                    },
                    "links": self.links,
                    "sources": [str(s) for s in self.source],
                    "content": self.content
                }
                
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                log.info(f"Exported curriculum to {path} in {fmt} format")
                return path
            else:
                raise NotImplementedError(f"Unknown export {fmt} format")
        except IOError as e:
            raise IOError(f"Error writing to file {path}: {e}")
