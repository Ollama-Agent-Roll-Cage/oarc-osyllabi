"""
Prompt templates for curriculum generation.

This module provides advanced templating capabilities for AI prompts,
supporting template loading, formatting, validation, and versioning.
"""
import os
import json
from pathlib import Path
from string import Template
from typing import Dict, Any, Optional, Union, List, Tuple
import re

from osyllabi.utils.log import log
from osyllabi.utils.decorators.singleton import singleton


@singleton
class PromptTemplate:
    """
    Handles template-based prompts for AI requests.
    
    This class provides functionality to create, format, validate, and track
    prompt templates for generating different parts of a curriculum.
    """
    
    # Template for creating a curriculum overview
    OVERVIEW_TEMPLATE = """
    Create a comprehensive overview for a curriculum on ${topic} for ${skill_level} level students.
    
    The overview should include:
    1. A brief introduction to ${topic} explaining its importance and scope
    2. Why learning ${topic} is valuable (real-world applications, career opportunities)
    3. Who this curriculum is designed for (prerequisites, background knowledge)
    4. What learners will achieve by the end (skills, knowledge, capabilities)
    5. How this curriculum is structured to support effective learning
    
    Your overview should be engaging, well-structured, and targeted at ${skill_level} level learners.
    Format the overview in markdown with appropriate headings and sections.
    """
    
    # Template for generating a learning path
    LEARNING_PATH_TEMPLATE = """
    Create a structured learning path for ${topic} at ${skill_level} level.
    
    The learning path should include:
    1. 5-8 sequential modules or units, each with clear learning objectives
    2. A logical progression from fundamentals to more advanced concepts
    3. Estimated time commitment for each module (hours/weeks)
    4. Key concepts, skills, and knowledge areas covered in each module
    5. Dependencies between modules (prerequisites)
    6. Milestones and checkpoints to assess progress
    
    Consider the typical background of ${skill_level} level learners when determining
    the appropriate depth, pace, and complexity of the material.
    
    Format the learning path in markdown with numbered modules, clear headings, 
    and bullet points for the content of each module.
    """
    
    # Template for generating resources
    RESOURCES_TEMPLATE = """
    Suggest comprehensive learning resources for a curriculum on ${topic} at ${skill_level} level.
    
    For each resource category, provide 3-5 specific recommendations:
    
    1. BOOKS AND TEXTBOOKS:
       - Include both foundational texts and practical guides
       - Note which are best for beginners vs. more advanced learners
    
    2. ONLINE COURSES:
       - Include free and paid options from platforms like Coursera, edX, Udemy
       - Mention course duration and key topics covered
    
    3. VIDEO TUTORIALS:
       - Include high-quality YouTube channels, playlists, or other video resources
    
    4. INTERACTIVE TOOLS:
       - Software, websites, or platforms that provide hands-on practice
    
    5. COMMUNITIES AND FORUMS:
       - Places where learners can ask questions and connect with others
    
    If there are specific URLs provided (${links}), incorporate them into your recommendations 
    with explanations of how they fit into the learning path.
    
    For each resource, include:
    - Title/name with link if available
    - Brief description of what it covers (2-3 sentences)
    - Why it's particularly valuable for ${skill_level} level learners
    - Any prerequisites needed
    
    Format the resources in markdown with clear headings and bullet points.
    """
    
    # Template for generating projects or exercises
    PROJECTS_TEMPLATE = """
    Design ${number_of_projects} practical projects or exercises for a curriculum on ${topic} at ${skill_level} level.
    
    For each project/exercise:
    1. Create a descriptive title that reflects the learning objectives
    2. Provide a concise problem statement or project overview (2-3 sentences)
    3. List 3-5 specific learning objectives that will be achieved
    4. Outline step-by-step instructions or requirements (5-10 steps)
    5. Suggest approximate time needed to complete (hours/days)
    6. Define clear criteria for success and evaluation metrics
    7. Include tips for overcoming common challenges
    8. Suggest ways to extend the project for additional learning
    
    Projects should progressively increase in complexity and build upon concepts from the learning path.
    Each project should be practical, engaging, and relevant to real-world applications of ${topic}.
    
    Format the projects in markdown with appropriate headings, numbered lists for steps,
    and bullet points for objectives and evaluation criteria.
    """
    
    # System template for setting up the agent persona
    SYSTEM_TEMPLATE = """
    You are an expert curriculum designer with extensive knowledge in ${topic} and education.
    Your goal is to create clear, structured, and effective learning materials.
    
    When designing curriculum components:
    - Be comprehensive but concise
    - Use clear, precise language appropriate for ${skill_level} level learners
    - Structure content logically with meaningful progression
    - Focus on practical, applicable knowledge
    - Consider diverse learning styles and backgrounds
    
    Format your response with markdown to improve readability.
    """
    
    # Version tracking for templates
    TEMPLATE_VERSIONS = {
        "overview": "2.1",
        "learning_path": "2.2",
        "resources": "2.3",
        "projects": "2.0",
        "system": "1.0"
    }
    
    def __init__(self, template_text: Optional[str] = None, template_name: Optional[str] = None):
        """
        Initialize a prompt template.
        
        Args:
            template_text: Custom template text, or None to create an empty template
            template_name: Optional name for the template for tracking purposes
        """
        self.template = Template(template_text) if template_text else None
        self.template_name = template_name
        self.template_text = template_text
        self.variables = self._extract_variables(template_text) if template_text else set()
        self.version = "custom-1.0"
        self.usage_count = 0
        self.successful_uses = 0
        
    def _extract_variables(self, template_text: str) -> set:
        """
        Extract template variables from the template text.
        
        Args:
            template_text: The template text to analyze
            
        Returns:
            Set of variable names used in the template
        """
        if not template_text:
            return set()
            
        # Find all ${variable} patterns in the template
        matches = re.findall(r'\${([^}]+)}', template_text)
        return set(matches)
            
    @classmethod
    def from_preset(cls, template_name: str) -> 'PromptTemplate':
        """
        Create a template from a predefined preset.
        
        Args:
            template_name: Name of the preset template to use
                (overview, learning_path, resources, projects, or system)
                
        Returns:
            PromptTemplate: Initialized with the selected preset
            
        Raises:
            ValueError: If template_name is not a valid preset
        """
        template_map = {
            'overview': cls.OVERVIEW_TEMPLATE,
            'learning_path': cls.LEARNING_PATH_TEMPLATE,
            'resources': cls.RESOURCES_TEMPLATE,
            'projects': cls.PROJECTS_TEMPLATE,
            'system': cls.SYSTEM_TEMPLATE
        }
        
        if template_name not in template_map:
            valid_options = ", ".join(template_map.keys())
            raise ValueError(f"Unknown template preset: '{template_name}'. "
                            f"Available presets: {valid_options}")
            
        template = cls(template_map[template_name], template_name)
        template.version = cls.TEMPLATE_VERSIONS.get(template_name, "1.0")
        
        log.debug(f"Created template from preset '{template_name}' (v{template.version})")
        return template
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'PromptTemplate':
        """
        Load a template from a file.
        
        Args:
            file_path: Path to the template file (txt or json)
            
        Returns:
            PromptTemplate: Initialized with the loaded template
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {file_path}")
            
        try:
            # Determine file type and load accordingly
            if path.suffix.lower() in ['.json']:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                template_text = data.get('template')
                template_name = data.get('name', path.stem)
                version = data.get('version', "file-1.0")
                
                if not template_text:
                    raise ValueError(f"JSON template file must contain a 'template' field")
                    
                template = cls(template_text, template_name)
                template.version = version
                
            else:  # Default to raw text file
                with open(path, 'r', encoding='utf-8') as f:
                    template_text = f.read()
                    
                template = cls(template_text, path.stem)
                template.version = "file-1.0"
                
            log.info(f"Loaded template '{template.template_name}' from {path.name}")
            return template
            
        except Exception as e:
            log.error(f"Failed to load template from {path}: {e}")
            raise ValueError(f"Failed to load template from {file_path}: {e}")
    
    def save_to_file(self, file_path: Union[str, Path], format: str = 'json') -> None:
        """
        Save the template to a file.
        
        Args:
            file_path: Path where to save the template
            format: Format to save as ('json' or 'txt')
            
        Raises:
            ValueError: If the template is not initialized or format is invalid
        """
        if not self.template_text:
            raise ValueError("Cannot save empty template")
            
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format.lower() == 'json':
                data = {
                    'name': self.template_name or 'unnamed_template',
                    'version': self.version,
                    'template': self.template_text,
                    'variables': list(self.variables),
                    'stats': {
                        'usage_count': self.usage_count,
                        'successful_uses': self.successful_uses
                    }
                }
                
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                    
            else:  # Default to raw text
                with open(path, 'w', encoding='utf-8') as f:
                    if self.template_name:
                        f.write(f"# Template: {self.template_name} (v{self.version})\n\n")
                    f.write(self.template_text)
                
            log.info(f"Saved template to {path}")
            
        except Exception as e:
            log.error(f"Failed to save template to {path}: {e}")
            raise ValueError(f"Failed to save template: {e}")
    
    def format(self, **kwargs) -> str:
        """
        Format the template with the provided values.
        
        Args:
            **kwargs: Key-value pairs to substitute in the template
            
        Returns:
            str: The formatted prompt string
            
        Raises:
            ValueError: If template is not initialized
            KeyError: If required template variables are missing
        """
        if not self.template:
            raise ValueError("Template not initialized")
        
        # Track usage stats
        self.usage_count += 1
            
        try:
            # Check for missing variables
            missing_vars = self.variables - set(kwargs.keys())
            if missing_vars:
                missing_list = ", ".join(missing_vars)
                raise KeyError(f"Missing required template variables: {missing_list}")
                
            # Format template
            result = self.template.substitute(**kwargs)
            
            # Track successful formatting
            self.successful_uses += 1
            return result
            
        except KeyError as e:
            log.error(f"Missing template variable: {e}")
            raise ValueError(f"Missing required template variable: {e}")
        
    def add_examples(self, examples: List[Dict[str, str]]) -> 'PromptTemplate':
        """
        Add few-shot learning examples to a template.
        
        Args:
            examples: List of example dictionaries with keys matching template variables
            
        Returns:
            PromptTemplate: Updated template with examples
            
        Raises:
            ValueError: If template is not initialized or examples are invalid
        """
        if not self.template_text:
            raise ValueError("Cannot add examples to empty template")
            
        if not examples:
            return self
            
        # Format example section
        example_section = "\n\nEXAMPLES:\n"
        
        for i, example in enumerate(examples, 1):
            example_section += f"\nExample {i}:\n"
            
            # Format each example using the available variables
            try:
                # Extract input and output from example
                example_inputs = {k: v for k, v in example.items() if k != 'output'}
                example_output = example.get('output', '')
                
                # Format example with available inputs
                example_prompt = self.template.safe_substitute(**example_inputs)
                
                # Add formatted example
                example_section += f"Input:\n```\n{example_prompt}\n```\n"
                if example_output:
                    example_section += f"Output:\n```\n{example_output}\n```\n"
                    
            except Exception as e:
                log.warning(f"Skipped invalid example {i}: {e}")
        
        # Create new template with examples appended
        new_template_text = self.template_text + example_section
        new_template = PromptTemplate(new_template_text, f"{self.template_name}_with_examples")
        new_template.version = f"{self.version}+examples"
        
        return new_template
    
    def create_chat_messages(
        self, 
        system_template: Optional['PromptTemplate'] = None,
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Create a list of chat messages from the template.
        
        Args:
            system_template: Optional system template to use
            **kwargs: Key-value pairs to substitute in the templates
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
            
        Raises:
            ValueError: If user template is not initialized
        """
        messages = []
        
        # Add system message if provided
        if system_template:
            try:
                system_content = system_template.format(**kwargs)
                messages.append({
                    "role": "system",
                    "content": system_content
                })
            except Exception as e:
                log.warning(f"Failed to format system template: {e}")
                # Continue with just the user message
        
        # Add user message from this template
        user_content = self.format(**kwargs)
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate the template structure and variables.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.template_text:
            return False, "Template is empty"
            
        # Check for balanced braces in variable placeholders
        open_count = self.template_text.count('${')
        close_count = self.template_text.count('}')
        
        if open_count != close_count:
            return False, f"Unbalanced variable placeholders: ${open_count} opening vs {close_count} closing"
            
        # Check template size
        if len(self.template_text) > 10000:
            return False, f"Template too large: {len(self.template_text)} chars (max 10000)"
            
        return True, None
