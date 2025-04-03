"""
Code file extraction for curriculum generation.

This module provides classes for extracting and analyzing content from code files
of various programming languages.
"""
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from osyllabi.utils.log import log
from osyllabi.generator.resource.extractor import ContentExtractorABC


class CodeFileExtractor(ContentExtractorABC):
    """
    Extracts and analyzes content from code files.
    
    This class provides functionality to extract meaningful information from
    code files, including language detection, structure analysis, and formatting.
    """
    
    def __init__(self):
        """Initialize the code file extractor."""
        # Language mapping by extension
        self.language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.jsx': 'JavaScript (React)',
            '.ts': 'TypeScript',
            '.tsx': 'TypeScript (React)',
            '.java': 'Java',
            '.c': 'C',
            '.cpp': 'C++',
            '.h': 'C/C++ Header',
            '.cs': 'C#',
            '.go': 'Go',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin'
        }
        
        # Language-specific patterns
        self.patterns = {
            'Python': {
                'class': r'class\s+([A-Za-z0-9_]+)',
                'function': r'def\s+([A-Za-z0-9_]+)',
                'import': r'(?:import|from)\s+([A-Za-z0-9_.]+)',
                'comment': r'#.*$'
            },
            'JavaScript': {
                'class': r'class\s+([A-Za-z0-9_]+)',
                'function': r'(?:function\s+([A-Za-z0-9_]+)|(?:const|let|var)\s+([A-Za-z0-9_]+)\s*=\s*(?:async\s*)?\()',
                'import': r'(?:import|require)\s+.*?(?:from\s+)?[\'"]([^\'"]*)[\'"]+',
                'comment': r'(?://.*$|/\*[\s\S]*?\*/)'
            },
            'Java': {
                'class': r'class\s+([A-Za-z0-9_]+)',
                'function': r'(?:public|private|protected)?\s+(?:static\s+)?[A-Za-z0-9_<>]+\s+([A-Za-z0-9_]+)\s*\(',
                'import': r'import\s+([A-Za-z0-9_.]+)',
                'comment': r'(?://.*$|/\*[\s\S]*?\*/)'
            },
            'C': {
                'function': r'[A-Za-z0-9_]+\s+([A-Za-z0-9_]+)\s*\([^;]*\)\s*\{',
                'struct': r'struct\s+([A-Za-z0-9_]+)',
                'include': r'#include\s+[<"]([^>"]+)[>"]',
                'comment': r'(?://.*$|/\*[\s\S]*?\*/)'
            },
            'C++': {
                'class': r'class\s+([A-Za-z0-9_]+)',
                'function': r'[A-Za-z0-9_]+\s+([A-Za-z0-9_]+)\s*\([^;]*\)\s*(?:const)?\s*\{',
                'namespace': r'namespace\s+([A-Za-z0-9_]+)',
                'include': r'#include\s+[<"]([^>"]+)[>"]',
                'comment': r'(?://.*$|/\*[\s\S]*?\*/)'
            }
        }
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract and analyze content from a code file.
        
        Args:
            file_path: Path to the code file
            
        Returns:
            Dictionary with extracted content and analysis
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            # Determine the programming language
            extension = file_path.suffix.lower()
            language = self.language_map.get(extension, 'Unknown')
            
            # Analyze the code structure
            analysis = self._analyze_code(content, language)
            
            # Format result
            summary = self._create_summary(analysis, language)
            
            return {
                "title": file_path.name,
                "content": f"{summary}\n\n{content}",
                "content_type": "code",
                "extension": extension,
                "language": language,
                "metadata": {
                    "size_bytes": file_path.stat().st_size,
                    "language": language,
                    "analysis": analysis
                }
            }
        except Exception as e:
            log.warning(f"Error analyzing code file {file_path}: {e}")
            
            # Fall back to basic extraction
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            return {
                "title": file_path.name,
                "content": content,
                "content_type": "text",
                "extension": file_path.suffix,
                "metadata": {
                    "size_bytes": file_path.stat().st_size
                }
            }
    
    def supports(self, resource: Any) -> bool:
        """Check if this extractor supports the resource."""
        if isinstance(resource, Path) and resource.is_file():
            extension = resource.suffix.lower()
            return extension in self.language_map
        return False
    
    def _analyze_code(self, content: str, language: str) -> Dict[str, Any]:
        """
        Analyze code structure.
        
        Args:
            content: Code content
            language: Programming language
            
        Returns:
            Dictionary with analysis results
        """
        # Basic statistics
        line_count = content.count('\n') + 1
        char_count = len(content)
        
        # Find language-specific elements
        classes = []
        functions = []
        imports = []
        
        # Get patterns for this language
        patterns = self.patterns.get(language, {})
        
        # Extract classes
        if 'class' in patterns:
            classes = re.findall(patterns['class'], content, re.MULTILINE)
            
        # Extract functions
        if 'function' in patterns:
            functions = re.findall(patterns['function'], content, re.MULTILINE)
            # Some patterns may have multiple capture groups - flatten the results
            flattened_functions = []
            for func in functions:
                if isinstance(func, tuple):
                    flattened_functions.extend([f for f in func if f])
                else:
                    flattened_functions.append(func)
            functions = flattened_functions
            
        # Extract imports/includes
        if 'import' in patterns:
            imports = re.findall(patterns['import'], content, re.MULTILINE)
        elif 'include' in patterns:
            imports = re.findall(patterns['include'], content, re.MULTILINE)
            
        # Count comments
        comment_count = 0
        if 'comment' in patterns:
            comments = re.findall(patterns['comment'], content, re.MULTILINE)
            comment_count = len(comments)
            
        # Return analysis results
        return {
            "line_count": line_count,
            "char_count": char_count,
            "class_count": len(classes),
            "classes": classes[:10],  # Limit to 10 for brevity
            "function_count": len(functions),
            "functions": functions[:10],  # Limit to 10 for brevity
            "import_count": len(imports),
            "imports": imports[:10],  # Limit to 10 for brevity
            "comment_count": comment_count
        }
    
    def _create_summary(self, analysis: Dict[str, Any], language: str) -> str:
        """
        Create a summary of the code analysis.
        
        Args:
            analysis: Analysis results
            language: Programming language
            
        Returns:
            Formatted summary string
        """
        summary = [f"# {language} Code Analysis"]
        
        # Add basic stats
        summary.append(f"\n## Statistics")
        summary.append(f"- Lines of code: {analysis['line_count']}")
        summary.append(f"- Characters: {analysis['char_count']}")
        
        # Add classes if any
        if analysis['class_count'] > 0:
            summary.append(f"\n## Classes ({analysis['class_count']} total)")
            for cls in analysis['classes']:
                summary.append(f"- `{cls}`")
                
        # Add functions if any
        if analysis['function_count'] > 0:
            summary.append(f"\n## Functions ({analysis['function_count']} total)")
            for func in analysis['functions']:
                summary.append(f"- `{func}`")
                
        # Add imports/includes if any
        if analysis['import_count'] > 0:
            summary.append(f"\n## Imports/Includes ({analysis['import_count']} total)")
            for imp in analysis['imports']:
                summary.append(f"- `{imp}`")
                
        # Add note about comments
        if analysis['comment_count'] > 0:
            summary.append(f"\n## Comments")
            summary.append(f"- {analysis['comment_count']} comment sections")
            
        return "\n".join(summary)
