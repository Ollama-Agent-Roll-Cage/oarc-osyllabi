"""
Tests for the curriculum generator functionality.
"""
import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import os
import json
import argparse
import time
from pathlib import Path

from osyllabi.core.curriculum import Curriculum
from osyllabi.utils.const import SUCCESS, FAILURE


class TestCurriculum(unittest.TestCase):
    """Test cases for Curriculum class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test directory for outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = Path(self.temp_dir.name)
        
        # Mock arguments
        self.test_topic = "Python Programming"
        self.test_skill_level = "Beginner"
        self.test_links = ["https://python.org"]
        self.test_source = ["./docs"]
        
        # Patch check_for_ollama to avoid real verification
        self.ollama_patcher = patch('osyllabi.core.curriculum.check_for_ollama')
        self.mock_check_ollama = self.ollama_patcher.start()
        
        # Patch CurriculumWorkflow to avoid real generation
        self.workflow_patcher = patch('osyllabi.core.curriculum.CurriculumWorkflow')
        self.mock_workflow_class = self.workflow_patcher.start()
        self.mock_workflow = self.mock_workflow_class.return_value
        self.mock_workflow.generate_full_curriculum.return_value = "# Test Curriculum\n\nTest content"
        
        # Set model attribute on workflow mock for generate_with_llama_index test
        self.mock_workflow.model = "llama3.1:latest"
        
        # Patch file operations
        self.open_patcher = patch('builtins.open', mock_open())
        self.mock_open = self.open_patcher.start()
        
        # Patch RAGEngine for generate_with_llama_index test
        self.rag_engine_patcher = patch('osyllabi.rag.engine.RAGEngine')  # Fix: Use correct module path
        self.mock_rag_engine_class = self.rag_engine_patcher.start()
        self.mock_rag_engine = self.mock_rag_engine_class.return_value
        
        # Patch time import for generate_with_llama_index test
        self.time_patcher = patch('osyllabi.core.curriculum.time')
        self.mock_time = self.time_patcher.start()
        self.mock_time.time.return_value = 12345
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.ollama_patcher.stop()
        self.workflow_patcher.stop()
        self.open_patcher.stop()
        self.rag_engine_patcher.stop()
        self.time_patcher.stop()
        self.temp_dir.cleanup()
    
    def test_curriculum_initialization(self):
        """Test basic curriculum initialization."""
        curriculum = Curriculum(
            topic=self.test_topic,
            skill_level=self.test_skill_level,
            links=self.test_links,
            source=self.test_source
        )
        
        self.assertEqual(curriculum.topic, self.test_topic)
        self.assertEqual(curriculum.title, f"{self.test_topic} Curriculum")
        self.assertEqual(curriculum.skill_level, self.test_skill_level)
        self.assertEqual(curriculum.links, self.test_links)
        self.assertEqual(curriculum.source, self.test_source)
        self.assertEqual(curriculum.content, "")
        self.assertTrue(hasattr(curriculum, 'created_at'))
        
        # Verify Ollama check was called
        self.mock_check_ollama.assert_called_once_with(raise_error=True)
    
    def test_initialization_with_empty_topic(self):
        """Test initialization with empty topic raises ValueError."""
        with self.assertRaises(ValueError):
            Curriculum(topic="", skill_level=self.test_skill_level)
            
        with self.assertRaises(ValueError):
            Curriculum(topic=None, skill_level=self.test_skill_level)
    
    def test_initialization_with_custom_title(self):
        """Test initialization with a custom title."""
        custom_title = "Advanced Python Masterclass"
        curriculum = Curriculum(
            topic=self.test_topic,
            title=custom_title,
            skill_level="Advanced"
        )
        
        self.assertEqual(curriculum.title, custom_title)
    
    def test_factory_method(self):
        """Test the factory method for creating curriculum instances."""
        curriculum = Curriculum.create(
            topic=self.test_topic,
            skill_level=self.test_skill_level
        )
        
        self.assertIsInstance(curriculum, Curriculum)
        self.assertEqual(curriculum.topic, self.test_topic)
    
    @patch('osyllabi.core.curriculum.get_output_directory')
    @patch('osyllabi.core.curriculum.create_unique_file_path')
    def test_process_args(self, mock_create_path, mock_get_output):
        """Test processing command line arguments."""
        # Setup mocks
        mock_get_output.return_value = self.output_path
        mock_create_path.return_value = self.output_path / "test_curriculum.md"
        
        # Create mock args
        args = argparse.Namespace()
        args.topic = self.test_topic
        args.title = "Test Curriculum"
        args.level = self.test_skill_level
        args.links = self.test_links
        args.source = self.test_source
        args.format = "md"
        args.export_path = None
        
        # Create the original curriculum object to access _process_args directly
        curriculum = Curriculum(
            topic=self.test_topic,
            skill_level=self.test_skill_level
        )
        
        # Call _process_args directly to test its behavior
        result = curriculum._process_args(args)
        
        # Now test the result tuple directly
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], SUCCESS)
        self.assertIsInstance(result[1], Path)
    
    def test_process_args_empty_topic(self):
        """Test processing args with empty topic returns failure."""
        args = argparse.Namespace()
        args.topic = ""
        
        # Create the original curriculum object to access _process_args directly
        curriculum = Curriculum(
            topic=self.test_topic,
            skill_level=self.test_skill_level
        )
        
        # Call _process_args directly to test its behavior
        result = curriculum._process_args(args)
        
        # Now test the result tuple directly
        self.assertEqual(result[0], FAILURE)
        self.assertIsNone(result[1])
    
    def test_process_args_ollama_error(self):
        """Test processing args with Ollama error returns failure."""
        # Setup the args
        args = argparse.Namespace()
        args.topic = self.test_topic
        args.title = ""
        args.level = self.test_skill_level
        args.links = []
        args.source = []
        args.format = "md"
        args.export_path = None
        
        # Create the curriculum object first
        curriculum = Curriculum(
            topic=self.test_topic,
            skill_level=self.test_skill_level
        )
        
        # Now patch within the specific context of _process_args
        with patch('osyllabi.core.curriculum.check_for_ollama', 
                  side_effect=RuntimeError("Ollama not available")):
            # Call _process_args directly
            result = curriculum._process_args(args)
            
            # Test the result tuple
            self.assertEqual(result[0], FAILURE)
            self.assertIsNone(result[1])
    
    def test_generate_content(self):
        """Test content generation."""
        curriculum = Curriculum(
            topic=self.test_topic,
            skill_level=self.test_skill_level
        )
        
        # Generate should be empty initially
        self.assertEqual(curriculum.content, "")
        
        # Generate content
        curriculum.generate_content()
        
        # Should now have content from the mocked workflow
        self.assertEqual(curriculum.content, "# Test Curriculum\n\nTest content")
        self.mock_workflow.generate_full_curriculum.assert_called_once()
    
    def test_export_markdown(self):
        """Test exporting curriculum to markdown format."""
        curriculum = Curriculum(
            topic=self.test_topic,
            skill_level=self.test_skill_level
        )
        curriculum.content = "# Test Curriculum\n\nTest content"
        
        export_path = curriculum.export(self.output_path / "test.md")
        
        self.mock_open.assert_called_once()
        self.assertEqual(export_path, self.output_path / "test.md")
    
    def test_export_json(self):
        """Test exporting curriculum to JSON format."""
        curriculum = Curriculum(
            topic=self.test_topic,
            skill_level=self.test_skill_level
        )
        curriculum.content = "# Test Curriculum\n\nTest content"
        
        with patch('json.dump') as mock_json_dump:
            export_path = curriculum.export(self.output_path / "test.json", fmt="json")
        
        self.mock_open.assert_called_once()
        mock_json_dump.assert_called_once()
        self.assertEqual(export_path, self.output_path / "test.json")
    
    def test_export_invalid_format(self):
        """Test exporting to an invalid format raises NotImplementedError."""
        curriculum = Curriculum(
            topic=self.test_topic,
            skill_level=self.test_skill_level
        )
        curriculum.content = "# Test Curriculum\n\nTest content"
        
        with self.assertRaises(NotImplementedError):
            curriculum.export(self.output_path / "test.pdf", fmt="pdf")
    
    def test_export_to_directory(self):
        """Test exporting to a directory path."""
        curriculum = Curriculum(
            topic=self.test_topic,
            skill_level=self.test_skill_level
        )
        curriculum.content = "# Test Curriculum\n\nTest content"
        
        # Mock the directory exists
        with patch('pathlib.Path.is_dir', return_value=True):
            export_path = curriculum.export(self.output_path, fmt="md")
        
        self.assertIsInstance(export_path, Path)
        # Should create a filename based on the title
        expected_filename = f"{self.test_topic.replace(' ', '_')}_Curriculum.md"
        self.assertTrue(str(export_path).endswith(expected_filename))
    
    @patch('osyllabi.rag.rag_agent.RAGAgent')
    def test_generate_with_llama_index(self, mock_rag_agent_class):
        """Test generating curriculum with LlamaIndex capabilities."""
        # Setup mocks
        mock_rag_agent = mock_rag_agent_class.return_value
        mock_rag_agent.retrieve_context.return_value = "Test context"
        mock_rag_agent.create_enhanced_prompt.return_value = "Test enhanced prompt"
        mock_rag_agent.generate.return_value = "Test content from RAG"
        
        # Create curriculum
        curriculum = Curriculum(
            topic=self.test_topic,
            skill_level=self.test_skill_level
        )
        
        # Run the method - import time here to prevent issues
        with patch('osyllabi.core.curriculum.time') as mock_time:
            mock_time.time.return_value = 12345
            result = curriculum.generate_with_llama_index(self.test_topic, self.test_skill_level)
        
        # RAGEngine should be created
        self.mock_rag_engine_class.assert_called_once_with(run_id=f"llama_index_12345")
        
        # Agent should be created with correct params
        mock_rag_agent_class.assert_called_once_with(
            name="LlamaIndex_Curriculum",
            model="llama3.1:latest",  # Uses the mocked workflow.model
            rag_engine=self.mock_rag_engine  # Uses the mocked engine
        )
        
        # Verify RAG operations were called
        self.assertEqual(mock_rag_agent.retrieve_context.call_count, 4)  # Called for each section
        self.assertEqual(mock_rag_agent.create_enhanced_prompt.call_count, 4)
        self.assertEqual(mock_rag_agent.generate.call_count, 4)
        
        # Verify content contains the expected text
        self.assertTrue("Test content from RAG" in curriculum.content)
        
        # Verify the method returns the content
        self.assertEqual(result, curriculum.content)


if __name__ == '__main__':
    unittest.main()
