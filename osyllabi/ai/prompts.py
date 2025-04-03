"""
Prompt templates for curriculum generation.
"""
from string import Template

from osyllabi.utils.log import log


class PromptTemplate:
    """
    Handles template-based prompts for AI requests.
    
    This class provides functionality to create and format prompt templates
    for generating different parts of a curriculum.
    """
    
    # Template for creating a curriculum overview
    OVERVIEW_TEMPLATE = """
    Create a comprehensive overview for a curriculum on ${topic} for ${skill_level} level students.
    The overview should include:
    1. A brief introduction to ${topic}
    2. Why it's important to learn
    3. Who this curriculum is suitable for
    4. What learners will achieve by the end
    
    Format the overview in markdown with appropriate headings.
    """
    
    # Template for generating a learning path
    LEARNING_PATH_TEMPLATE = """
    Create a structured learning path for ${topic} at ${skill_level} level.
    
    The learning path should include:
    1. 5-8 modules or units, each with clear learning objectives
    2. A logical progression from fundamentals to more advanced concepts
    3. Suggested time investment for each module
    4. Key concepts that will be covered in each module
    
    Format the learning path in markdown with appropriate headings and bullet points.
    """
    
    # Template for generating resources
    RESOURCES_TEMPLATE = """
    Suggest comprehensive learning resources for a curriculum on ${topic} at ${skill_level} level.
    
    For each resource, include:
    1. Title/name
    2. Type (book, online course, video, etc.)
    3. Brief description of what it covers
    4. Why it's valuable for this curriculum
    
    If there are specific URLs provided (${links}), incorporate them into your recommendations 
    with explanations of how they fit into the learning path.
    
    Format the resources in markdown with appropriate headings and bullet points.
    """
    
    # Template for generating projects or exercises
    PROJECTS_TEMPLATE = """
    Design ${number_of_projects} practical projects or exercises for a curriculum on ${topic} at ${skill_level} level.
    
    For each project/exercise:
    1. Create a clear title
    2. Provide a concise description of the task
    3. List specific learning objectives
    4. Suggest approximate time needed
    5. Define criteria for success
    
    Projects should increase in complexity and help reinforce concepts from the learning path.
    Format the projects in markdown with appropriate headings.
    """
    
    def __init__(self, template_text: str = None):
        """
        Initialize a prompt template.
        
        Args:
            template_text: Custom template text, or None to create an empty template
        """
        self.template = Template(template_text) if template_text else None
        
    @classmethod
    def from_preset(cls, template_name: str) -> 'PromptTemplate':
        """
        Create a template from a predefined preset.
        
        Args:
            template_name: Name of the preset template to use
                (overview, learning_path, resources, or projects)
                
        Returns:
            PromptTemplate: Initialized with the selected preset
            
        Raises:
            ValueError: If template_name is not a valid preset
        """
        template_map = {
            'overview': cls.OVERVIEW_TEMPLATE,
            'learning_path': cls.LEARNING_PATH_TEMPLATE,
            'resources': cls.RESOURCES_TEMPLATE,
            'projects': cls.PROJECTS_TEMPLATE
        }
        
        if template_name not in template_map:
            raise ValueError(f"Unknown template preset: {template_name}. "
                            f"Available presets: {', '.join(template_map.keys())}")
            
        return cls(template_map[template_name])
    
    def format(self, **kwargs) -> str:
        """
        Format the template with the provided values.
        
        Args:
            **kwargs: Key-value pairs to substitute in the template
            
        Returns:
            str: The formatted prompt string
            
        Raises:
            ValueError: If template is not initialized
        """
        if not self.template:
            raise ValueError("Template not initialized")
            
        try:
            return self.template.substitute(**kwargs)
        except KeyError as e:
            log.error(f"Missing required template variable: {e}")
            raise ValueError(f"Missing required template variable: {e}")
