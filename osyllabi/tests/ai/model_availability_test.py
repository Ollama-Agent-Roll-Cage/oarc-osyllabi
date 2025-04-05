"""
Tests for Ollama model availability and error handling.
"""
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import asyncio
import json
import os

from osyllabi.ai.client import OllamaClient
from osyllabi.config import AI_CONFIG


class TestOllamaModelAvailability(unittest.TestCase):
    """Test cases for Ollama model availability and handling."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a event loop for testing
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up after tests."""
        self.loop.close()

    def test_default_model_config(self):
        """Test that the default model in AI_CONFIG is set to llama3.1:latest."""
        self.assertEqual(AI_CONFIG['default_model'], 'llama3.1:latest')
    
    def test_client_default_model(self):
        """Test that the OllamaClient uses llama3.1:latest as default model."""
        with patch('osyllabi.ai.client.OllamaClient._validate_server'):
            client = OllamaClient()
            self.assertEqual(client.default_model, 'llama3.1:latest')
    
    @patch('osyllabi.ai.client.OllamaClient.validate_ollama')
    @patch('osyllabi.ai.client.AsyncClient.chat')
    def test_model_not_found_error_handling(self, mock_chat, mock_validate):
        """Test that the client handles model not found errors gracefully."""
        # Setup mocks
        mock_validate.return_value = None
        
        # Mock the 404 error for model not found
        mock_error = Exception("model \"nonexistent_model\" not found, try pulling it first (status code: 404)")
        mock_chat.side_effect = mock_error
        
        # Initialize client and call generate with a non-existent model
        client = OllamaClient()
        result = self.loop.run_until_complete(
            client.async_generate("Test prompt", model="nonexistent_model")
        )
        
        # Should handle the error and return an error message
        self.assertTrue(result.startswith("Error generating text:"))
        self.assertTrue("model \"nonexistent_model\" not found" in result)
    
    @pytest.mark.live_only
    def test_live_model_availability(self):
        """
        Test actual model availability on a running Ollama instance.
        
        This test only runs when the --live flag is provided to pytest.
        """
        client = OllamaClient()
        models = self.loop.run_until_complete(client.list_models())
        
        # Check if our default model is available
        # The Ollama response structure has models with 'model' attribute instead of 'name'
        model_names = [model.model for model in models]  # Access the 'model' attribute directly
        default_model = AI_CONFIG['default_model']
        
        # If model isn't found, print helpful message
        if default_model not in model_names:
            available_models = ", ".join(model_names)
            self.fail(
                f"Default model '{default_model}' not found in available models: {available_models}. "
                f"Please run: ollama pull {default_model}"
            )
    
    @patch('osyllabi.ai.client.OllamaClient.validate_ollama')
    @patch('osyllabi.ai.client.AsyncClient.pull')
    def test_pull_model_functionality(self, mock_pull, mock_validate):
        """Test that the pull_model method works correctly."""
        # Setup mocks
        mock_validate.return_value = None
        mock_pull.return_value = AsyncMock()
        
        # Initialize client
        client = OllamaClient()
        
        # Run the pull_model method
        success = self.loop.run_until_complete(client.pull_model("llama3.1:latest"))
        
        # Verify results
        self.assertTrue(success)
        mock_pull.assert_called_once_with("llama3.1:latest", stream=True)


if __name__ == '__main__':
    unittest.main()
