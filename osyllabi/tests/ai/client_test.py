"""
Unit tests for the Ollama API client.
"""
import unittest
import asyncio
from unittest.mock import patch, AsyncMock

from osyllabi.ai.client import OllamaClient
from osyllabi.config import AI_CONFIG

class TestOllamaClient(unittest.TestCase):
    """Test cases for the OllamaClient class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a event loop for testing
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up after tests."""
        self.loop.close()
        
    @patch('osyllabi.ai.client.OllamaClient.validate_ollama')
    @patch('osyllabi.ai.client.AsyncClient.chat')
    def test_generate_text(self, mock_chat, mock_validate):
        """Test text generation with the client."""
        # Setup mocks
        mock_validate.return_value = None
        
        chat_result = {
            "message": {
                "role": "assistant", 
                "content": "Generated text response"
            }
        }
        mock_chat.return_value = chat_result
        
        # Initialize client and generate text - use the async method directly
        client = OllamaClient()
        # Run the async generate method
        result = self.loop.run_until_complete(client.async_generate("Test prompt"))
        
        # Assert response was processed correctly
        self.assertEqual(result, "Generated text response")
        
        # Verify proper parameters were sent
        mock_chat.assert_called_once()
        args, kwargs = mock_chat.call_args
        self.assertEqual(kwargs["model"], AI_CONFIG['default_model'])
        self.assertEqual(kwargs["messages"][0]["role"], "user")
        self.assertEqual(kwargs["messages"][0]["content"], "Test prompt")
        self.assertEqual(kwargs["options"]["temperature"], AI_CONFIG['temperature'])
        self.assertEqual(kwargs["options"]["num_predict"], 4000)

    @patch('osyllabi.ai.client.OllamaClient.validate_ollama')
    @patch('osyllabi.ai.client.AsyncClient.chat')
    def test_chat_completion(self, mock_chat, mock_validate):
        """Test chat completion with the client."""
        # Setup mocks
        mock_validate.return_value = None
        
        chat_result = {
            "message": {
                "role": "assistant",
                "content": "Chat response"
            }
        }
        mock_chat.return_value = chat_result
        
        # Initialize client and call chat
        client = OllamaClient()
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]
        result = self.loop.run_until_complete(client.chat(messages))
        
        # Assert chat call was made correctly
        mock_chat.assert_called_once()
        self.assertEqual(result, chat_result)
        
        # Verify proper parameters were used
        args, kwargs = mock_chat.call_args
        self.assertEqual(kwargs["model"], AI_CONFIG['default_model'])
        self.assertEqual(kwargs["messages"], messages)
        self.assertEqual(kwargs["options"]["num_predict"], 4000)
        self.assertEqual(kwargs["options"]["temperature"], AI_CONFIG['temperature'])

    @patch('osyllabi.ai.client.OllamaClient.validate_ollama')
    @patch('osyllabi.ai.client.AsyncClient.embeddings')
    def test_embedding_generation(self, mock_embeddings, mock_validate):
        """Test embedding generation with the client."""
        # Setup mocks
        mock_validate.return_value = None
        
        embedding_result = {
            "embedding": [0.1, 0.2, 0.3, 0.4]
        }
        mock_embeddings.return_value = embedding_result
        
        # Initialize client and generate embedding
        client = OllamaClient()
        result = self.loop.run_until_complete(client.embed("Text to embed"))
        
        # Assert response was processed correctly
        self.assertEqual(result, [0.1, 0.2, 0.3, 0.4])
        
        # Verify proper parameters were used
        mock_embeddings.assert_called_once()
        args, kwargs = mock_embeddings.call_args
        self.assertEqual(kwargs["model"], AI_CONFIG['default_model'])
        self.assertEqual(kwargs["prompt"], "Text to embed")

    @patch('osyllabi.ai.client.OllamaClient.validate_ollama')
    @patch('osyllabi.ai.client.AsyncClient.pull')
    def test_pull_model(self, mock_pull, mock_validate):
        """Test pulling a model with progress tracking."""
        # Mock the validation methods
        mock_validate.return_value = None
        
        # Create mock pull progress
        mock_progress = [
            {"status": "downloading digest"},
            {"digest": "sha256:abc123", "total": 1000, "completed": 200},
            {"digest": "sha256:abc123", "total": 1000, "completed": 500},
            {"digest": "sha256:abc123", "total": 1000, "completed": 1000},
            {"status": "success"}
        ]
        
        # Create an async generator that yields progress items
        async def mock_pull_generator():
            for item in mock_progress:
                yield item
            
        # Make mock_pull return the async generator function instead of the list
        mock_pull.return_value = mock_pull_generator()
        
        # Create the client
        client = OllamaClient()
        
        # Track progress
        progress_updates = []
        def progress_callback(update):
            progress_updates.append(update)
        
        # Run the test
        result = self.loop.run_until_complete(
            client.pull_model("test-model", progress_callback=progress_callback)
        )
        
        # Check results
        self.assertTrue(result)
        self.assertEqual(len(progress_updates), 5)  # 5 progress updates
        
        # Verify the call
        mock_pull.assert_called_once()
        args, kwargs = mock_pull.call_args
        
        # Checking positional argument (model name)
        self.assertEqual(args[0], "test-model")
        self.assertTrue(kwargs["stream"])


if __name__ == '__main__':
    unittest.main()
