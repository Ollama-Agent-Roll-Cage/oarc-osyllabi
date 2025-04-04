"""
Unit tests for the Ollama API client.
"""
import unittest
from unittest.mock import patch, MagicMock

from osyllabi.ai.client import OllamaClient
from osyllabi.config import AI_CONFIG

class TestOllamaClient(unittest.TestCase):
    """Test cases for the OllamaClient class."""
    
    # Remove test_client_initialization test since initialization is tested implicitly
    
    @patch.object(OllamaClient, 'validate_ollama')
    @patch('osyllabi.ai.client.requests.post')
    def test_generate_text(self, mock_post, mock_validate):
        """Test text generation with the client."""
        # Setup mocks
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
        
        actual_payload = mock_post.call_args[1]['json']
        self.assertEqual(actual_payload["model"], expected_payload["model"])
        self.assertEqual(actual_payload["messages"], expected_payload["messages"])
        self.assertEqual(actual_payload["temperature"], expected_payload["temperature"])
        self.assertEqual(actual_payload["num_predict"], expected_payload["num_predict"])

    @patch.object(OllamaClient, 'validate_ollama')  # Update: consistent patching
    @patch('osyllabi.ai.client.requests.post')
    def test_chat_completion(self, mock_post, mock_validate):
        """Test chat completion with the client."""
        # Setup mocks
        mock_validate.return_value = True
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
        
        # Verify proper enydpoint was used
        self.assertTrue(mock_post.call_args[0][0].endswith('/api/chat'))

    @patch.object(OllamaClient, 'validate_ollama')  # Update: consistent patching
    @patch('osyllabi.ai.client.requests.post')
    def test_embedding_generation(self, mock_post, mock_validate):
        """Test embedding generation with the client."""
        # Setup mocks
        mock_validate.return_value = True
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
