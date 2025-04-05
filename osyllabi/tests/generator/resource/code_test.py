"""
Tests for code file extraction functionality.
"""
import unittest
from unittest.mock import patch, MagicMock, Mock, mock_open
from pathlib import Path

from osyllabi.generator.resource.code import CodeFileExtractor


class TestCodeFileExtractor(unittest.TestCase):
    """Test cases for the CodeFileExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = CodeFileExtractor()
        
    def test_initialization(self):
        """Test initialization of CodeFileExtractor."""
        self.assertIn('.py', self.extractor.language_map)
        self.assertIn('.js', self.extractor.language_map)
        self.assertIn('.java', self.extractor.language_map)
        self.assertEqual(self.extractor.language_map['.py'], 'Python')
        self.assertEqual(self.extractor.language_map['.js'], 'JavaScript')
        self.assertEqual(self.extractor.language_map['.java'], 'Java')
        
        # Verify patterns for different languages
        self.assertIn('Python', self.extractor.patterns)
        self.assertIn('JavaScript', self.extractor.patterns)
        self.assertIn('Java', self.extractor.patterns)
        self.assertIn('class', self.extractor.patterns['Python'])
        self.assertIn('function', self.extractor.patterns['JavaScript'])
        
    def test_supports_method(self):
        """Test supports method for different file types."""
        mock_path = MagicMock(spec=Path)
        
        # Test Python file support
        mock_path.is_file.return_value = True
        mock_path.suffix.lower.return_value = '.py'
        self.assertTrue(self.extractor.supports(mock_path))
        
        # Test JavaScript file support
        mock_path.suffix.lower.return_value = '.js'
        self.assertTrue(self.extractor.supports(mock_path))
        
        # Test Java file support
        mock_path.suffix.lower.return_value = '.java'
        self.assertTrue(self.extractor.supports(mock_path))
        
        # Test unsupported file type
        mock_path.suffix.lower.return_value = '.txt'
        self.assertFalse(self.extractor.supports(mock_path))
        
        # Test non-file resource
        mock_path.is_file.return_value = False
        mock_path.suffix.lower.return_value = '.py'
        self.assertFalse(self.extractor.supports(mock_path))
        
    @patch('builtins.open', new_callable=mock_open, read_data="""
class TestClass:
    def __init__(self):
        self.value = 42
    
    def test_method(self, param):
        return self.value + param
        
def standalone_function():
    return "Hello World"
    
# This is a comment
import os
from datetime import datetime
""")
    @patch('pathlib.Path')
    def test_extract_python_code(self, mock_path_class, mock_file):
        """Test extracting content from Python code file."""
        # Setup mock
        mock_path = mock_path_class.return_value
        mock_path.suffix.lower.return_value = '.py'
        mock_path.name = 'test_file.py'
        mock_path.stat.return_value.st_size = 500
        
        # Extract content
        result = self.extractor.extract(mock_path)
        
        # Verify extraction results
        self.assertEqual(result["title"], 'test_file.py')
        self.assertEqual(result["content_type"], "code")
        self.assertEqual(result["language"], "Python")
        
        # Verify code analysis results
        analysis = result["metadata"]["analysis"]
        self.assertGreaterEqual(analysis["line_count"], 10)
        self.assertEqual(analysis["class_count"], 1)
        self.assertEqual(analysis["function_count"], 2)
        self.assertEqual(analysis["comment_count"], 1)
        
        # Check extracted classes, functions, and imports
        self.assertIn("TestClass", analysis["classes"])
        self.assertIn("test_method", analysis["functions"])
        self.assertIn("standalone_function", analysis["functions"])
        self.assertIn("os", analysis["imports"])
        self.assertIn("datetime", analysis["imports"])
        
        # Verify summary content - get the full summary part
        summary_parts = result["content"].split("class TestClass")[0].strip()
        
        # Check for various sections in the summary
        self.assertIn("# Python Code Analysis", summary_parts)
        self.assertIn("## Statistics", summary_parts)
        self.assertIn("## Classes", summary_parts)
        self.assertIn("## Functions", summary_parts)
        self.assertIn("## Imports", summary_parts)
        
    @patch('builtins.open', new_callable=mock_open, read_data="""
class Rectangle {
    constructor(height, width) {
        this.height = height;
        this.width = width;
    }
    
    get area() {
        return this.height * this.width;
    }
}

function calculateArea(shape) {
    return shape.area;
}

// This is a comment
import { Component } from '@angular/core';
""")
    @patch('pathlib.Path')
    def test_extract_javascript_code(self, mock_path_class, mock_file):
        """Test extracting content from JavaScript code file."""
        # Setup mock
        mock_path = mock_path_class.return_value
        mock_path.suffix.lower.return_value = '.js'
        mock_path.name = 'test_file.js'
        mock_path.stat.return_value.st_size = 500
        
        # Extract content
        result = self.extractor.extract(mock_path)
        
        # Verify extraction results
        self.assertEqual(result["title"], 'test_file.js')
        self.assertEqual(result["content_type"], "code")
        self.assertEqual(result["language"], "JavaScript")
        
        # Verify code analysis results
        analysis = result["metadata"]["analysis"]
        self.assertGreaterEqual(analysis["line_count"], 10)
        self.assertEqual(analysis["class_count"], 1)
        self.assertGreaterEqual(analysis["function_count"], 1)
        self.assertEqual(analysis["comment_count"], 1)
        
        # Check extracted classes, functions, and imports
        self.assertIn("Rectangle", analysis["classes"])
        self.assertIn("calculateArea", analysis["functions"])
        self.assertIn("@angular/core", analysis["imports"])
        
    @patch('builtins.open', new_callable=mock_open, read_data="""
public class HelloWorld {
    private int value;
    
    public HelloWorld(int initialValue) {
        this.value = initialValue;
    }
    
    public int getValue() {
        return this.value;
    }
    
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}

// This is a comment
import java.util.List;
import java.util.ArrayList;
""")
    @patch('pathlib.Path')
    def test_extract_java_code(self, mock_path_class, mock_file):
        """Test extracting content from Java code file."""
        # Setup mock
        mock_path = mock_path_class.return_value
        mock_path.suffix.lower.return_value = '.java'
        mock_path.name = 'HelloWorld.java'
        mock_path.stat.return_value.st_size = 500
        
        # Extract content
        result = self.extractor.extract(mock_path)
        
        # Verify extraction results
        self.assertEqual(result["title"], 'HelloWorld.java')
        self.assertEqual(result["content_type"], "code")
        self.assertEqual(result["language"], "Java")
        
        # Verify code analysis results
        analysis = result["metadata"]["analysis"]
        self.assertGreaterEqual(analysis["line_count"], 10)
        self.assertEqual(analysis["class_count"], 1)
        self.assertGreaterEqual(analysis["function_count"], 2)
        self.assertEqual(analysis["comment_count"], 1)
        
        # Check extracted classes, functions, and imports
        self.assertIn("HelloWorld", analysis["classes"])
        self.assertIn("getValue", analysis["functions"])
        self.assertIn("main", analysis["functions"])
        self.assertIn("java.util.List", analysis["imports"])
        self.assertIn("java.util.ArrayList", analysis["imports"])
        
    @patch('builtins.open', new_callable=mock_open, read_data="""
#include <iostream>
#include <vector>

struct Point {
    int x;
    int y;
};

int calculateDistance(const Point& p1, const Point& p2) {
    return abs(p1.x - p2.x) + abs(p1.y - p2.y);
}

int main() {
    std::cout << "Hello C++!" << std::endl;
    return 0;
}

// This is a comment
""")
    @patch('pathlib.Path')
    def test_extract_cpp_code(self, mock_path_class, mock_file):
        """Test extracting content from C++ code file."""
        # Setup mock
        mock_path = mock_path_class.return_value
        mock_path.suffix.lower.return_value = '.cpp'
        mock_path.name = 'test_file.cpp'
        mock_path.stat.return_value.st_size = 500
        
        # Extract content
        result = self.extractor.extract(mock_path)
        
        # Verify extraction results
        self.assertEqual(result["title"], 'test_file.cpp')
        self.assertEqual(result["content_type"], "code")
        self.assertEqual(result["language"], "C++")
        
        # Verify code analysis results
        analysis = result["metadata"]["analysis"]
        self.assertGreaterEqual(analysis["line_count"], 10)
        self.assertGreaterEqual(analysis["function_count"], 2)
        self.assertEqual(analysis["comment_count"], 1)
        
        # Check extracted functions and includes
        self.assertIn("calculateDistance", analysis["functions"])
        self.assertIn("main", analysis["functions"])
        self.assertIn("iostream", analysis["imports"])
        self.assertIn("vector", analysis["imports"])
        
    def test_analyze_code_python(self):
        """Test analyzing Python code structure."""
        python_code = """
class TestClass:
    def __init__(self):
        self.value = 42
    
    def test_method(self, param):
        return self.value + param
        
def standalone_function():
    return "Hello World"
    
# This is a comment
import os
from datetime import datetime
"""
        
        analysis = self.extractor._analyze_code(python_code, "Python")
        
        self.assertGreaterEqual(analysis["line_count"], 10)
        self.assertEqual(analysis["class_count"], 1)
        self.assertEqual(analysis["function_count"], 2)
        self.assertEqual(analysis["comment_count"], 1)
        self.assertIn("TestClass", analysis["classes"])
        self.assertIn("test_method", analysis["functions"])
        self.assertIn("standalone_function", analysis["functions"])
        self.assertIn("os", analysis["imports"])
        self.assertIn("datetime", analysis["imports"])
        
    def test_analyze_code_javascript(self):
        """Test analyzing JavaScript code structure."""
        js_code = """
class Rectangle {
    constructor(height, width) {
        this.height = height;
        this.width = width;
    }
    
    get area() {
        return this.height * this.width;
    }
}

function calculateArea(shape) {
    return shape.area;
}

const arrowFunc = () => {
    return 'Arrow function';
};

// This is a comment
import { Component } from '@angular/core';
"""
        
        analysis = self.extractor._analyze_code(js_code, "JavaScript")
        
        self.assertGreaterEqual(analysis["line_count"], 10)
        self.assertEqual(analysis["class_count"], 1)
        self.assertGreaterEqual(analysis["function_count"], 2)
        self.assertEqual(analysis["comment_count"], 1)
        self.assertIn("Rectangle", analysis["classes"])
        self.assertIn("calculateArea", analysis["functions"])
        self.assertIn("@angular/core", analysis["imports"])
        
    def test_create_summary(self):
        """Test creating code analysis summary."""
        analysis = {
            "line_count": 25,
            "char_count": 500,
            "class_count": 2,
            "classes": ["TestClass", "AnotherClass"],
            "function_count": 3,
            "functions": ["func1", "func2", "func3"],
            "import_count": 4,
            "imports": ["module1", "module2", "module3", "module4"],
            "comment_count": 5
        }
        
        summary = self.extractor._create_summary(analysis, "Python")
        
        self.assertIn("# Python Code Analysis", summary)
        self.assertIn("## Statistics", summary)
        self.assertIn("Lines of code: 25", summary)
        self.assertIn("Characters: 500", summary)
        self.assertIn("## Classes (2 total)", summary)
        self.assertIn("- `TestClass`", summary)
        self.assertIn("- `AnotherClass`", summary)
        self.assertIn("## Functions (3 total)", summary)
        self.assertIn("- `func1`", summary)
        self.assertIn("- `func2`", summary)
        self.assertIn("- `func3`", summary)
        self.assertIn("## Imports/Includes (4 total)", summary)
        self.assertIn("- `module1`", summary)
        self.assertIn("- `module2`", summary)
        self.assertIn("## Comments", summary)
        self.assertIn("5 comment sections", summary)


if __name__ == '__main__':
    unittest.main()
