"""
Tests for curriculum workflow functionality.
"""
import unittest
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, Any, List

from osyllabi.generator.workflow import CurriculumWorkflow
from osyllabi.ai.client import OllamaClient
from osyllabi.generator.resource import ResourceCollectionManager


class TestCurriculumWorkflow(unittest.TestCase):
    """Test cases for the CurriculumWorkflow class."""
    
    @patch('osyllabi.generator.workflow.check_for_ollama')
    @patch('osyllabi.generator.workflow.OllamaClient')
    @patch('osyllabi.generator.workflow.ResourceCollectionManager')
    def setUp(self, mock_resource_manager_class, mock_client_class, mock_check_ollama):
        """Set up test fixtures with mocked dependencies."""
        # Configure mocks
        self.mock_client = mock_client_class.return_value
        self.mock_resource_manager = mock_resource_manager_class.return_value
        
        # Initialize CurriculumWorkflow with mocked dependencies
        self.workflow = CurriculumWorkflow(
            topic="Python Programming",
            skill_level="Beginner",
            client=self.mock_client,
            model="llama3:latest",
            temperature=0.5
        )
        
        # Replace resource manager with our mock
        self.workflow.resource_manager = self.mock_resource_manager
        
    def test_initialization(self):
        """Test initialization of CurriculumWorkflow."""
        self.assertEqual(self.workflow.topic, "Python Programming")
        self.assertEqual(self.workflow.skill_level, "Beginner")
        self.assertEqual(self.workflow.model, "llama3:latest")
        self.assertEqual(self.workflow.temperature, 0.5)
        self.assertIs(self.workflow.client, self.mock_client)
        
        # Verify templates are loaded
        self.assertIn("overview", self.workflow.templates)
        self.assertIn("learning_path", self.workflow.templates)
        self.assertIn("resources", self.workflow.templates)
        self.assertIn("projects", self.workflow.templates)
        
    def test_collect_resources(self):
        """Test resource collection."""
        # Configure mock resource manager
        self.mock_resource_manager.collect_resources.return_value = {
            "urls": {"http://example.com": {"content": "Example content"}},
            "files": {"test.py": {"content": "Test content"}},
            "stats": {"total_content_size": 25}
        }
        
        # Call collect_resources
        links = ["http://example.com"]
        source_paths = ["test.py"]
        result = self.workflow.collect_resources(links, source_paths)
        
        # Verify collect_resources was called with correct arguments
        self.mock_resource_manager.collect_resources.assert_called_once_with(
            urls=links,
            paths=source_paths
        )
        
        # Verify result is passed through unchanged
        self.assertEqual(len(result["urls"]), 1)
        self.assertEqual(len(result["files"]), 1)
        self.assertEqual(result["stats"]["total_content_size"], 25)
        
    def test_generate_overview(self):
        """Test overview generation."""
        # Configure mock client to return sample overview
        self.mock_client.generate.return_value = "This is an overview of Python Programming."
        
        # Generate overview
        overview = self.workflow.generate_overview()
        
        # Verify client.generate was called with correct template and parameters
        self.mock_client.generate.assert_called_once()
        args, kwargs = self.mock_client.generate.call_args
        
        self.assertEqual(kwargs["model"], "llama3:latest")
        self.assertEqual(kwargs["temperature"], 0.5)
        self.assertIn("Python Programming", args[0])  # topic should be in prompt
        self.assertIn("Beginner", args[0])  # skill level should be in prompt
        
        # Verify result
        self.assertEqual(overview, "This is an overview of Python Programming.")
        
    def test_generate_learning_path_without_resources(self):
        """Test learning path generation without resources."""
        # Configure mock client to return sample learning path
        self.mock_client.generate.return_value = "This is a learning path for Python Programming."
        
        # Generate learning path without resources
        learning_path = self.workflow.generate_learning_path()
        
        # Verify client.generate was called with correct template and parameters
        self.mock_client.generate.assert_called_once()
        args, kwargs = self.mock_client.generate.call_args
        
        # Verify no context was added to prompt (since no resources provided)
        self.assertNotIn("context", args[0].lower())
        
        # Verify result
        self.assertEqual(learning_path, "This is a learning path for Python Programming.")
        
    def test_generate_learning_path_with_resources(self):
        """Test learning path generation with resources."""
        # Configure mock client to return sample learning path
        self.mock_client.generate.return_value = "This is a learning path for Python Programming with resources."
        
        # Configure mock resource manager to return sample context
        self.mock_resource_manager.extract_context_for_prompt.return_value = "Sample resource context"
        
        # Mock resources
        resources = {
            "urls": {"http://example.com": {"content": "Example content"}},
            "files": {"test.py": {"content": "Test content"}}
        }
        
        # Generate learning path with resources
        learning_path = self.workflow.generate_learning_path(resources)
        
        # Verify extract_context_for_prompt was called
        self.mock_resource_manager.extract_context_for_prompt.assert_not_called()  # This is called only with LangChain
        
        # Verify client.generate was called with correct template and parameters
        self.mock_client.generate.assert_called_once()
        args, kwargs = self.mock_client.generate.call_args
        
        # Verify context was added to prompt 
        self.assertIn("context", args[0].lower())
        
        # Verify result
        self.assertEqual(learning_path, "This is a learning path for Python Programming with resources.")
        
    @patch('osyllabi.generator.workflow.importlib.util.find_spec')
    def test_generate_learning_path_with_langchain(self, mock_find_spec):
        """Test learning path generation using LangChain."""
        # Configure mock to indicate LangChain is available
        mock_find_spec.return_value = True
        self.workflow.langchain_available = True
        
        # Mock LangChain
        with patch('osyllabi.generator.workflow.RecursiveCharacterTextSplitter') as mock_splitter_class:
            mock_splitter = mock_splitter_class.return_value
            mock_splitter.split_text.return_value = ["Chunk 1", "Chunk 2", "Chunk 3"]
            
            # Configure mock client to return sample learning path
            self.mock_client.generate.return_value = "This is a learning path with LangChain."
            
            # Mock resources with many URLs and files to trigger LangChain processing
            resources = {
                "urls": {
                    "http://example1.com": {"content": "Example content 1"},
                    "http://example2.com": {"content": "Example content 2"},
                    "http://example3.com": {"content": "Example content 3"},
                    "http://example4.com": {"content": "Example content 4"},
                    "http://example5.com": {"content": "Example content 5"},
                    "http://example6.com": {"content": "Example content 6"},
                },
                "files": {}
            }
            
            # Generate learning path with resources and LangChain
            learning_path = self.workflow.generate_learning_path(resources)
            
            # Verify splitter was used
            mock_splitter.split_text.assert_called_once()
            
            # Verify client.generate was called with chunks in the prompt
            self.mock_client.generate.assert_called_once()
            args, kwargs = self.mock_client.generate.call_args
            self.assertIn("context", args[0].lower())
            
            # Verify result
            self.assertEqual(learning_path, "This is a learning path with LangChain.")
            
    def test_generate_resources_section(self):
        """Test resources section generation."""
        # Configure mock client to return sample resources section
        self.mock_client.generate.return_value = "This is a resources section for Python Programming."
        
        # Generate resources section with specific links
        links = ["http://python.org", "http://tutorial.com"]
        resources_section = self.workflow.generate_resources_section(links)
        
        # Verify client.generate was called with correct template and parameters
        self.mock_client.generate.assert_called_once()
        args, kwargs = self.mock_client.generate.call_args
        
        # Verify links were added to prompt
        self.assertIn("http://python.org", args[0])
        self.assertIn("http://tutorial.com", args[0])
        
        # Verify result
        self.assertEqual(resources_section, "This is a resources section for Python Programming.")
        
    def test_generate_projects(self):
        """Test projects section generation."""
        # Configure mock client to return sample projects
        self.mock_client.generate.return_value = "This is a projects section for Python Programming."
        
        # Generate projects section with specific number
        projects = self.workflow.generate_projects(number_of_projects=5)
        
        # Verify client.generate was called with correct template and parameters
        self.mock_client.generate.assert_called_once()
        args, kwargs = self.mock_client.generate.call_args
        
        # Verify number of projects was included in prompt
        self.assertIn("5", args[0])  # number_of_projects should be in prompt
        
        # Verify result
        self.assertEqual(projects, "This is a projects section for Python Programming.")
        
    def test_generate_full_curriculum(self):
        """Test full curriculum generation."""
        # Configure mock client responses for each section
        self.mock_client.generate.side_effect = [
            "This is an overview.",
            "This is a learning path.",
            "These are resources.",
            "These are projects."
        ]
        
        # Configure mock resource collection
        self.mock_resource_manager.collect_resources.return_value = {
            "urls": {"http://example.com": {"content": "Example content"}},
            "files": {"test.py": {"content": "Test content"}},
            "stats": {}
        }
        
        # Generate full curriculum
        links = ["http://example.com"]
        paths = ["test.py"]
        curriculum = self.workflow.generate_full_curriculum(links, paths)
        
        # Verify collect_resources was called - use named parameters to match actual call
        self.mock_resource_manager.collect_resources.assert_called_once_with(urls=links, paths=paths)
        
        # Verify client.generate was called 4 times (overview, learning path, resources, projects)
        self.assertEqual(self.mock_client.generate.call_count, 4)
        
        # Verify curriculum contains all sections
        self.assertIn("# Python Programming Curriculum (Beginner Level)", curriculum)
        self.assertIn("## Overview", curriculum)
        self.assertIn("This is an overview.", curriculum)
        self.assertIn("## Learning Path", curriculum)
        self.assertIn("This is a learning path.", curriculum)
        self.assertIn("## Resources", curriculum)
        self.assertIn("These are resources.", curriculum)
        self.assertIn("## Projects", curriculum)
        self.assertIn("These are projects.", curriculum)
        self.assertIn("## Metadata", curriculum)  # Should include metadata section
        
    def test_get_model_name(self):
        """Test retrieving the model name."""
        self.assertEqual(self.workflow.get_model_name(), "llama3:latest")


if __name__ == '__main__':
    unittest.main()
