"""
Unit tests for the Ollama API client.
"""
import unittest
import json
from unittest.mock import patch, MagicMock

from osyllabi.ai.client import OllamaClient
from osyllabi.config import AI_CONFIG


class TestOllamaClient(unittest.TestCase):
    """Test cases for the OllamaClient class."""

    @patch('osyllabi.ai.client.check_for_ollama')
    @patch('osyllabi.ai.client.requests.get')
    def test_client_initialization(self, mock_get, mock_check_ollama):
        """Test that the client initializes correctly."""
        # Setup mock
        mock_check_ollama.return_value = True
        mock_get.return_value.status_code = 200
        
        # Initialize client
        client = OllamaClient()
        
        # Assert configuration from AI_CONFIG was used
        self.assertEqual(client.default_model, AI_CONFIG['default_model'])
        self.assertEqual(client.base_url, AI_CONFIG['ollama_api_url'])
        
        # Verify Ollama check was called
        mock_check_ollama.assert_called_once_with(raise_error=True)

    @patch('osyllabi.ai.client.check_for_ollama')
    @patch('osyllabi.ai.client.requests.post')
    def test_generate_text(self, mock_post, mock_check_ollama):
        """Test text generation with the client."""
        # Setup mocks
        mock_check_ollama.return_value = True
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "message": {
                "content": "Generated text response"
            }
        }
        mock_post.return_value = mock_response
        
        # Initialize client and generate text
        client = OllamaClient()
        result = client.generate("Test prompt")
        
        # Assert response was processed correctly
        self.assertEqual(result, "Generated text response")
        
        # Verify proper payload was sent
        expected_payload = {
            "model": AI_CONFIG['default_model'],
            "messages": [{"role": "user", "content": "Test prompt"}],
            "temperature": AI_CONFIG['temperature'],
            "num_predict": AI_CONFIG['max_tokens'],
            "stream": False
        }
        mock_post.assert_called_once()
        actual_payload = json.loads(mock_post.call_args[1]['json'])
        self.assertEqual(actual_payload["model"], expected_payload["model"])
        self.assertEqual(actual_payload["messages"], expected_payload["messages"])

    @patch('osyllabi.ai.client.check_for_ollama')
    @patch('osyllabi.ai.client.requests.post')
    def test_chat_completion(self, mock_post, mock_check_ollama):
        """Test chat completion with the client."""
        # Setup mocks
        mock_check_ollama.return_value = True
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "Chat response"
            }
        }
        mock_post.return_value = mock_response
        
        # Initialize client and call chat
        client = OllamaClient()
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]
        result = client.chat(messages)
        
        # Assert chat call was made correctly
        mock_post.assert_called_once()
        self.assertEqual(result, mock_response.json.return_value)
        
        # Verify proper endpoint was used
        self.assertTrue(mock_post.call_args[0][0].endswith('/api/chat'))

    @patch('osyllabi.ai.client.check_for_ollama')
    @patch('osyllabi.ai.client.requests.post')
    def test_embedding_generation(self, mock_post, mock_check_ollama):
        """Test embedding generation with the client."""
        # Setup mocks
        mock_check_ollama.return_value = True
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3, 0.4]
        }
        mock_post.return_value = mock_response
        
        # Initialize client and generate embedding
        client = OllamaClient()
        result = client.embed("Text to embed")
        
        # Assert response was processed correctly
        self.assertEqual(result, [0.1, 0.2, 0.3, 0.4])
        
        # Verify proper endpoint was used
        self.assertTrue(mock_post.call_args[0][0].endswith('/api/embeddings'))


if __name__ == '__main__':
    unittest.main()
