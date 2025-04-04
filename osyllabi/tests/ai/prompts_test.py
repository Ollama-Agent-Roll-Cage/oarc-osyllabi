"""
Unit tests for the PromptTemplate class.
"""
import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from osyllabi.ai.prompts import PromptTemplate


class TestPromptTemplate(unittest.TestCase):
    """Test cases for the PromptTemplate class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_template_text = "This is a test template for ${topic} at ${skill_level} level."
        self.template = PromptTemplate(self.test_template_text, "test_template")
        
    def test_initialization(self):
        """Test template initialization and attributes."""
        self.assertEqual(self.template.template_text, self.test_template_text)
        self.assertEqual(self.template.template_name, "test_template")
        self.assertEqual(self.template.version, "custom-1.0")
        self.assertEqual(self.template.usage_count, 0)
        self.assertEqual(self.template.successful_uses, 0)
        
        # Check extracted variables
        self.assertSetEqual(self.template.variables, {"topic", "skill_level"})
        
    def test_format(self):
        """Test template formatting with variables."""
        formatted = self.template.format(
            topic="Python Programming",
            skill_level="Intermediate"
        )
        
        expected = "This is a test template for Python Programming at Intermediate level."
        self.assertEqual(formatted, expected)
        
        # Check usage tracking
        self.assertEqual(self.template.usage_count, 1)
        self.assertEqual(self.template.successful_uses, 1)
        
    def test_format_missing_variables(self):
        """Test formatting with missing variables raises error."""
        with self.assertRaises(ValueError):
            self.template.format(topic="Python")
            
        # Usage count increases but successful uses doesn't
        self.assertEqual(self.template.usage_count, 1)
        self.assertEqual(self.template.successful_uses, 0)
        
    def test_from_preset(self):
        """Test creating template from preset."""
        for preset_name in ["overview", "learning_path", "resources", "projects", "system"]:
            template = PromptTemplate.from_preset(preset_name)
            self.assertEqual(template.template_name, preset_name)
            self.assertIsNotNone(template.template_text)
            self.assertIn(preset_name, PromptTemplate.TEMPLATE_VERSIONS)
            self.assertEqual(template.version, PromptTemplate.TEMPLATE_VERSIONS[preset_name])
            
        # Test invalid preset name
        with self.assertRaises(ValueError):
            PromptTemplate.from_preset("invalid_preset")
            
    def test_add_examples(self):
        """Test adding examples to a template."""
        examples = [
            {"topic": "Python", "skill_level": "Beginner", "output": "Example output for Python beginners"},
            {"topic": "Machine Learning", "skill_level": "Advanced", "output": "Example output for ML experts"}
        ]
        
        new_template = self.template.add_examples(examples)
        
        # Check that it's a new instance
        self.assertIsNot(new_template, self.template)
        self.assertIn("EXAMPLES:", new_template.template_text)
        self.assertIn("Example 1:", new_template.template_text)
        self.assertIn("Example 2:", new_template.template_text)
        self.assertIn("Python", new_template.template_text)
        self.assertIn("Machine Learning", new_template.template_text)
        
        # Original template should be unchanged
        self.assertNotIn("EXAMPLES:", self.template.template_text)
        
    def test_create_chat_messages(self):
        """Test creating chat messages from template."""
        # Create a system template
        system_template = PromptTemplate(
            "You are an expert in ${topic}. Help with ${skill_level} level content.",
            "system_template"
        )
        
        # Create chat messages
        messages = self.template.create_chat_messages(
            system_template=system_template,
            topic="Python",
            skill_level="Intermediate"
        )
        
        # Check structure and content
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("expert in Python", messages[0]["content"])
        self.assertIn("template for Python at Intermediate level", messages[1]["content"])
        
    def test_validate(self):
        """Test template validation."""
        # Valid template
        is_valid, error = self.template.validate()
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        # Invalid template with unbalanced braces
        invalid_template = PromptTemplate("This has ${unbalanced braces", "invalid")
        is_valid, error = invalid_template.validate()
        self.assertFalse(is_valid)
        self.assertIn("Unbalanced variable", error)
        
        # Empty template
        empty_template = PromptTemplate(None, "empty")
        is_valid, error = empty_template.validate()
        self.assertFalse(is_valid)
        self.assertEqual(error, "Template is empty")
        
    def test_save_and_load_json(self):
        """Test saving and loading template to/from JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save template to JSON file
            file_path = Path(temp_dir) / "test_template.json"
            self.template.save_to_file(file_path, format='json')
            
            # Check that file exists
            self.assertTrue(file_path.exists())
            
            # Load template from file
            loaded_template = PromptTemplate.from_file(file_path)
            
            # Check loaded template attributes
            self.assertEqual(loaded_template.template_text, self.test_template_text)
            self.assertEqual(loaded_template.template_name, "test_template")
            self.assertSetEqual(loaded_template.variables, {"topic", "skill_level"})
            
    def test_save_and_load_text(self):
        """Test saving and loading template to/from text file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save template to text file
            file_path = Path(temp_dir) / "test_template.txt"
            self.template.save_to_file(file_path, format='txt')
            
            # Check that file exists
            self.assertTrue(file_path.exists())
            
            # Load template from file
            loaded_template = PromptTemplate.from_file(file_path)
            
            # Check loaded template attributes - use assertIn instead of assertEqual
            # since save_to_file adds a header with the template name and version
            self.assertIn(self.test_template_text, loaded_template.template_text)
            self.assertEqual(loaded_template.template_name, "test_template")
            self.assertSetEqual(loaded_template.variables, {"topic", "skill_level"})
            
    def test_extract_variables(self):
        """Test variable extraction from template text."""
        template_text = "Template with ${var1}, ${var2}, and ${var3} variables."
        variables = self.template._extract_variables(template_text)
        self.assertSetEqual(variables, {"var1", "var2", "var3"})
        
        # Test with duplicate variables
        template_text = "Template with ${var1} and ${var1} again."
        variables = self.template._extract_variables(template_text)
        self.assertSetEqual(variables, {"var1"})
        
        # Test with no variables
        template_text = "Template with no variables."
        variables = self.template._extract_variables(template_text)
        self.assertSetEqual(variables, set())
        
        # Test with empty template
        variables = self.template._extract_variables("")
        self.assertSetEqual(variables, set())
        
        # Test with None template
        variables = self.template._extract_variables(None)
        self.assertSetEqual(variables, set())
        
    def test_empty_template(self):
        """Test behavior with empty template."""
        empty_template = PromptTemplate(None)
        
        # Should not be able to format empty template
        with self.assertRaises(ValueError):
            empty_template.format(topic="something")
            
        # Should not be able to save empty template
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "empty.json"
            with self.assertRaises(ValueError):
                empty_template.save_to_file(file_path)


if __name__ == '__main__':
    unittest.main()
