"""
Data file extraction for curriculum generation.

This module provides classes for extracting and analyzing content from data files
such as CSV, JSON, and other structured data formats.
"""
import json
import csv
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

from osyllabi.utils.log import log
from osyllabi.generator.resource.extractor import ContentExtractorABC


class DataFileExtractor(ContentExtractorABC):
    """
    Extracts and analyzes content from data files.
    
    This class provides functionality to extract meaningful information from
    data files, including format detection, structure analysis, and summarization.
    """
    
    def __init__(self):
        """Initialize the data file extractor."""
        # Format handlers
        self.format_handlers = {
            '.json': self._handle_json,
            '.csv': self._handle_csv,
            '.tsv': self._handle_csv,
            '.yml': self._handle_yaml,
            '.yaml': self._handle_yaml,
            '.docx': self._handle_docx
        }
        
        # Check for available libraries
        self.yaml_available = importlib.util.find_spec("yaml") is not None
        self.pandas_available = importlib.util.find_spec("pandas") is not None
        self.docx_available = importlib.util.find_spec("docx") is not None
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract and analyze content from a data file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Dictionary with extracted content and analysis
        """
        extension = file_path.suffix.lower()
        
        # Use specific handler if available
        if extension in self.format_handlers:
            try:
                return self.format_handlers[extension](file_path)
            except Exception as e:
                log.warning(f"Error processing {extension} file {file_path}: {e}")
        
        # Fall back to basic text extraction
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            return {
                "title": file_path.name,
                "content": content,
                "content_type": "text",
                "extension": extension,
                "metadata": {
                    "size_bytes": file_path.stat().st_size
                }
            }
        except Exception as e:
            log.warning(f"Error reading file {file_path}: {e}")
            return {
                "title": file_path.name,
                "content": f"[Error reading file: {e}]",
                "content_type": "error",
                "metadata": {
                    "size_bytes": file_path.stat().st_size,
                    "error": str(e)
                }
            }
    
    def supports(self, resource: Any) -> bool:
        """Check if this extractor supports the resource."""
        if isinstance(resource, Path) and resource.is_file():
            extension = resource.suffix.lower()
            return extension in self.format_handlers
        return False
    
    def _handle_json(self, file_path: Path) -> Dict[str, Any]:
        """Handle JSON file extraction."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                data = json.load(f)
                
            # Analyze JSON structure
            structure = self._analyze_json(data)
            summary = self._create_json_summary(data, structure)
            
            # Format for display
            formatted_json = json.dumps(data, indent=2, ensure_ascii=False)
            
            return {
                "title": file_path.name,
                "content": f"{summary}\n\n```json\n{formatted_json}\n```",
                "content_type": "json",
                "extension": file_path.suffix,
                "metadata": {
                    "size_bytes": file_path.stat().st_size,
                    "structure": structure
                }
            }
        except json.JSONDecodeError as e:
            # Handle invalid JSON
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            return {
                "title": file_path.name,
                "content": f"Invalid JSON content:\n{content}",
                "content_type": "text",
                "extension": file_path.suffix,
                "metadata": {
                    "size_bytes": file_path.stat().st_size,
                    "error": f"JSON parse error: {e}"
                }
            }
    
    def _handle_csv(self, file_path: Path) -> Dict[str, Any]:
        """Handle CSV/TSV file extraction."""
        try:
            # Use pandas if available for better CSV handling
            if self.pandas_available:
                import pandas as pd
                
                # Determine delimiter
                delimiter = ',' if file_path.suffix.lower() == '.csv' else '\t'
                
                # Read data with pandas
                df = pd.read_csv(file_path, delimiter=delimiter, nrows=100)  # Limit to 100 rows
                
                # Get basic info
                row_count = len(df)
                col_count = len(df.columns)
                header = df.columns.tolist()
                
                # Basic statistics for numeric columns
                stats = {}
                for col in df.select_dtypes(include=['number']).columns:
                    stats[col] = {
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean()),
                        "median": float(df[col].median())
                    }
                
                # Generate summary
                summary = [f"# CSV Data Analysis: {file_path.name}"]
                summary.append(f"\n## Structure")
                summary.append(f"- Rows: {row_count}")
                summary.append(f"- Columns: {col_count}")
                
                if header:
                    summary.append(f"\n## Headers")
                    for i, col in enumerate(header):
                        summary.append(f"- Column {i+1}: {col}")
                
                # Add statistics if available
                if stats:
                    summary.append(f"\n## Statistics")
                    for col, col_stats in stats.items():
                        summary.append(f"- {col}: min={col_stats['min']:.2f}, max={col_stats['max']:.2f}, mean={col_stats['mean']:.2f}")
                
                # Show sample data
                sample_rows = min(5, row_count)
                if sample_rows > 0:
                    summary.append(f"\n## Sample Data (first {sample_rows} rows)")
                    summary.append(f"\n```\n{df.head(sample_rows).to_string()}\n```")
                
                return {
                    "title": file_path.name,
                    "content": "\n".join(summary),
                    "content_type": "csv",
                    "extension": file_path.suffix,
                    "metadata": {
                        "size_bytes": file_path.stat().st_size,
                        "row_count": row_count,
                        "column_count": col_count,
                        "headers": header,
                        "statistics": stats
                    }
                }
                
            # Fall back to standard library if pandas not available
            else:
                delimiter = ',' if file_path.suffix.lower() == '.csv' else '\t'
                rows = []
                
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    reader = csv.reader(f, delimiter=delimiter)
                    for row in reader:
                        rows.append(row)
                        if len(rows) >= 100:  # Limit to 100 rows for performance
                            break
                
                # Create summary
                row_count = len(rows)
                col_count = len(rows[0]) if row_count > 0 else 0
                header = rows[0] if row_count > 0 else []
                
                summary = [f"# CSV Data Analysis: {file_path.name}"]
                summary.append(f"\n## Structure")
                summary.append(f"- Rows: {row_count}")
                summary.append(f"- Columns: {col_count}")
                
                if header:
                    summary.append(f"\n## Headers")
                    for i, col in enumerate(header):
                        summary.append(f"- Column {i+1}: {col}")
                
                # Show sample data
                sample_rows = min(5, row_count)
                if sample_rows > 0:
                    summary.append(f"\n## Sample Data (first {sample_rows} rows)")
                    
                    # Format as table
                    table_rows = []
                    if row_count > 1:  # If we have data rows
                        for i in range(1, sample_rows + 1):
                            if i < row_count:
                                table_rows.append("| " + " | ".join(rows[i]) + " |")
                    
                    if header and table_rows:
                        header_row = "| " + " | ".join(header) + " |"
                        separator = "| " + " | ".join(["---"] * len(header)) + " |"
                        table = [header_row, separator] + table_rows
                        summary.append("\n" + "\n".join(table))
                
                return {
                    "title": file_path.name,
                    "content": "\n".join(summary),
                    "content_type": "csv",
                    "extension": file_path.suffix,
                    "metadata": {
                        "size_bytes": file_path.stat().st_size,
                        "row_count": row_count,
                        "column_count": col_count,
                        "headers": header if header else []
                    }
                }
                
        except Exception as e:
            log.warning(f"Error processing CSV {file_path}: {e}")
            
            # Fall back to basic text
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            return {
                "title": file_path.name,
                "content": content,
                "content_type": "text",
                "extension": file_path.suffix,
                "metadata": {
                    "size_bytes": file_path.stat().st_size,
                    "error": str(e)
                }
            }
    
    def _handle_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Handle YAML file extraction."""
        try:
            # Use PyYAML if available
            if self.yaml_available:
                import yaml
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    data = yaml.safe_load(f)
                    
                # Re-format as JSON for display (more universal)
                return {
                    "title": file_path.name,
                    "content": f"YAML content (converted to JSON):\n```json\n{json.dumps(data, indent=2)}\n```",
                    "content_type": "yaml",
                    "extension": file_path.suffix,
                    "metadata": {
                        "size_bytes": file_path.stat().st_size
                    }
                }
            else:
                # PyYAML not available, just show raw content
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    
                return {
                    "title": file_path.name,
                    "content": f"YAML content (raw):\n```yaml\n{content}\n```",
                    "content_type": "yaml",
                    "extension": file_path.suffix,
                    "metadata": {
                        "size_bytes": file_path.stat().st_size
                    }
                }
        except Exception as e:
            log.warning(f"Error processing YAML {file_path}: {e}")
            
            # Fall back to basic text
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            return {
                "title": file_path.name,
                "content": content,
                "content_type": "text",
                "extension": file_path.suffix,
                "metadata": {
                    "size_bytes": file_path.stat().st_size,
                    "error": str(e)
                }
            }
    
    def _handle_docx(self, file_path: Path) -> Dict[str, Any]:
        """Handle DOCX file extraction using python-docx if available."""
        if not self.docx_available:
            return None
            
        try:
            from docx import Document
            
            # Open the document
            doc = Document(file_path)
            
            # Extract text
            paragraphs = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    paragraphs.append(paragraph.text)
            
            # Extract metadata
            metadata = {
                "size_bytes": file_path.stat().st_size,
                "paragraph_count": len(paragraphs),
                "section_count": len(doc.sections),
            }
            
            # Get headings/structure
            structure = []
            for paragraph in doc.paragraphs:
                if paragraph.style.name.startswith('Heading'):
                    heading_level = int(paragraph.style.name.replace('Heading ', ''))
                    if paragraph.text.strip():
                        structure.append({
                            "level": heading_level,
                            "title": paragraph.text
                        })
            
            if structure:
                metadata["structure"] = structure
            
            # Combine all paragraphs into content
            content = "\n\n".join(paragraphs)
            
            return {
                "title": file_path.name,
                "content": content,
                "content_type": "docx",
                "extension": file_path.suffix,
                "metadata": metadata
            }
        except Exception as e:
            log.warning(f"Error processing DOCX {file_path}: {e}")
            return None
            
    def _analyze_json(self, data: Any) -> Dict[str, Any]:
        """
        Analyze JSON structure.
        
        Args:
            data: Parsed JSON data
            
        Returns:
            Dictionary with analysis information
        """
        if isinstance(data, dict):
            # Handle dictionary
            keys = list(data.keys())
            key_types = {k: type(v).__name__ for k, v in data.items()}
            has_nested = any(isinstance(v, (dict, list)) for v in data.values())
            
            return {
                "type": "object",
                "keys_count": len(keys),
                "keys": keys[:20],  # Limit to first 20 keys
                "key_types": key_types,
                "has_nested": has_nested
            }
            
        elif isinstance(data, list):
            # Handle list
            item_count = len(data)
            sample_types = set()
            has_nested = False
            
            # Analyze sample items
            for item in data[:10]:  # Check first 10 items
                sample_types.add(type(item).__name__)
                if isinstance(item, (dict, list)):
                    has_nested = True
                    
            return {
                "type": "array",
                "item_count": item_count,
                "sample_types": list(sample_types),
                "has_nested": has_nested
            }
            
        else:
            # Handle primitive
            return {
                "type": type(data).__name__,
                "value": str(data)
            }
    
    def _create_json_summary(self, data: Any, structure: Dict[str, Any]) -> str:
        """
        Create a summary of JSON data based on structure analysis.
        
        Args:
            data: JSON data
            structure: Structure analysis from _analyze_json
            
        Returns:
            Formatted summary string
        """
        summary = [f"# JSON Data Analysis"]
        
        if structure.get("type") == "object":
            keys = structure.get("keys", [])
            key_count = structure.get("keys_count", 0)
            
            summary.append(f"\n## Structure")
            summary.append(f"- Type: JSON Object")
            summary.append(f"- Keys: {key_count}")
            
            if keys:
                summary.append(f"\n## Keys")
                for i, key in enumerate(keys[:15]):  # Show first 15 keys max
                    key_type = structure.get("key_types", {}).get(key, "unknown")
                    summary.append(f"- `{key}`: {key_type}")
                    
                if len(keys) > 15:
                    summary.append(f"- ... ({len(keys) - 15} more keys)")
                    
            if structure.get("has_nested"):
                summary.append(f"\n## Notes")
                summary.append(f"- Contains nested objects or arrays")
                
        elif structure.get("type") == "array":
            item_count = structure.get("item_count", 0)
            sample_types = structure.get("sample_types", [])
            
            summary.append(f"\n## Structure")
            summary.append(f"- Type: JSON Array")
            summary.append(f"- Items: {item_count}")
            
            if sample_types:
                summary.append(f"\n## Content Types")
                for type_name in sample_types:
                    summary.append(f"- {type_name}")
                    
            if structure.get("has_nested"):
                summary.append(f"\n## Notes")
                summary.append(f"- Contains nested objects or arrays")
                
            # Add sample items if array is not too large
            if item_count > 0 and item_count <= 5:
                summary.append(f"\n## Sample Items")
                for i, item in enumerate(data):
                    summary.append(f"- Item {i+1}: {str(item)[:100]}")
        else:
            # Simple primitive value
            summary.append(f"\n## Content")
            summary.append(f"- Type: {structure.get('type', 'unknown')}")
            summary.append(f"- Value: {structure.get('value', '')}")
            
        return "\n".join(summary)
