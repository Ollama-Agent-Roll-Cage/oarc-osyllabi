"""
AI client for interfacing with Ollama models.
"""
import json
import time
from typing import Any, Dict, List, Optional, Union, Generator

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from osyllabi.utils.log import log
from osyllabi.utils.decorators.singleton import singleton
from osyllabi.config import AI_CONFIG

@singleton
class OllamaClient:
    """
    Client for interacting with Ollama API.
    
    This class provides methods for generating text completions and
    embeddings using Ollama's local API.
    """
    
    @classmethod
    def validate_ollama(cls) -> None:
        """Validate Ollama availability."""
        from osyllabi.utils.utils import check_for_ollama
        check_for_ollama(raise_error=True)
        
    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None
    ):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Base URL for Ollama API
            default_model: Default model to use for requests
            default_temperature: Default temperature for generation
            default_max_tokens: Default maximum tokens for generation
            
        Raises:
            RuntimeError: If Ollama server is not available
        """
        # Call class method through the class to ensure proper mocking
        OllamaClient.validate_ollama()  # Change from self.validate_ollama()
        
        # Use configuration values or defaults
        self.base_url = base_url or AI_CONFIG.get('ollama_api_url', 'http://localhost:11434')
        self.default_model = default_model or AI_CONFIG.get('default_model', 'llama3')
        self.default_temperature = default_temperature if default_temperature is not None else AI_CONFIG.get('temperature', 0.7)
        self.default_max_tokens = default_max_tokens if default_max_tokens is not None else AI_CONFIG.get('max_tokens', 1000)
        
        # Validate server connection
        self._validate_server()

    def _validate_server(self) -> None:
        """
        Validate connection to Ollama server.
        
        Raises:
            RuntimeError: If server is not available
        """
        try:
            response = requests.get(self.base_url, timeout=5)
            if response.status_code != 200:
                raise RuntimeError(f"Ollama server returned unexpected status: {response.status_code}")
                
            log.info(f"Successfully connected to Ollama server at {self.base_url}")
        except requests.RequestException as e:
            error_msg = f"Failed to connect to Ollama server at {self.base_url}: {e}"
            log.error(error_msg)
            raise RuntimeError(error_msg)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True
    )
    def generate(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate a completion from a prompt.
        
        Args:
            prompt: The prompt to generate from
            model: Model to use (defaults to client's default_model)
            system: Optional system prompt for context
            temperature: Controls randomness (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            The generated text or a generator yielding text chunks
            
        Raises:
            ValueError: If the prompt is empty
            RuntimeError: On API errors
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")
            
        model_name = model or self.default_model
        temp = temperature if temperature is not None else self.default_temperature
        tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        # Build request payload - Use the chat endpoint instead of generate
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "num_predict": tokens,
            "stream": stream
        }
        
        # Add system prompt if provided
        if system:
            payload["messages"].insert(0, {"role": "system", "content": system})
            
        # Use the chat endpoint which is more reliable in Ollama
        endpoint = f"{self.base_url}/api/chat"
        
        log.debug(f"Sending request to {endpoint} using model {model_name}")
        
        try:
            if stream:
                return self._stream_chat_response_as_text(endpoint, payload)
            else:
                response = requests.post(endpoint, json=payload)
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "").strip()
                
        except requests.RequestException as e:
            log.error(f"API request failed: {e}")
            raise RuntimeError(f"Failed to generate text: {e}")
    
    def _stream_chat_response_as_text(self, endpoint: str, payload: Dict[str, Any]) -> Generator[str, None, None]:
        """
        Stream response from the chat API as text chunks.
        
        Args:
            endpoint: API endpoint
            payload: Request payload
            
        Returns:
            Generator yielding text chunks
            
        Raises:
            RuntimeError: On API errors
        """
        try:
            response = requests.post(endpoint, json=payload, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
                        
        except requests.RequestException as e:
            log.error(f"Streaming API request failed: {e}")
            raise RuntimeError(f"Failed to stream text: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True
    )
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Generate a chat completion from a list of messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Model to use (defaults to client's default_model)
            temperature: Controls randomness (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Dict with response or a generator yielding response chunks
            
        Raises:
            ValueError: If messages are empty or invalid
            RuntimeError: On API errors
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        for message in messages:
            if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                raise ValueError("Each message must be a dict with 'role' and 'content' keys")
                
        model_name = model or self.default_model
        temp = temperature if temperature is not None else self.default_temperature
        tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        # Build request payload
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temp,
            "num_predict": tokens,
            "stream": stream
        }
        
        endpoint = f"{self.base_url}/api/chat"
        
        log.debug(f"Sending chat request to {endpoint} using model {model_name}")
        
        try:
            if stream:
                return self._stream_chat_response(endpoint, payload)
            else:
                response = requests.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
                
        except requests.RequestException as e:
            log.error(f"Chat API request failed: {e}")
            raise RuntimeError(f"Failed to generate chat response: {e}")
    
    def _stream_chat_response(self, endpoint: str, payload: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """
        Stream chat response from the API.
        
        Args:
            endpoint: API endpoint
            payload: Request payload
            
        Returns:
            Generator yielding response chunks
            
        Raises:
            RuntimeError: On API errors
        """
        try:
            response = requests.post(endpoint, json=payload, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    yield data
                        
        except requests.RequestException as e:
            log.error(f"Streaming chat API request failed: {e}")
            raise RuntimeError(f"Failed to stream chat response: {e}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models in Ollama.
        
        Returns:
            List of dictionaries containing model information
            
        Raises:
            RuntimeError: On API errors
        """
        endpoint = f"{self.base_url}/api/tags"
        
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except requests.RequestException as e:
            log.error(f"Failed to list models: {e}")
            raise RuntimeError(f"Failed to list models: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True
    )
    def embed(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """
        Generate embedding for text using Ollama's embedding API.
        
        Args:
            text: The text to embed
            model: Model to use for embedding (defaults to client's default_model)
            
        Returns:
            List[float]: The embedding vector
            
        Raises:
            ValueError: If the text is empty
            RuntimeError: On API errors
        """
        if not text:
            raise ValueError("Text cannot be empty")
            
        model_name = model or self.default_model
        
        # Build request payload
        payload = {
            "model": model_name,
            "prompt": text
        }
            
        endpoint = f"{self.base_url}/api/embeddings"
        
        log.debug(f"Sending embedding request to {endpoint} using model {model_name}")
        
        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract the embedding from the response
            embedding = result.get("embedding", [])
            if not embedding:
                log.warning("Received empty embedding from Ollama API")
                
            return embedding
                
        except requests.RequestException as e:
            log.error(f"Embedding API request failed: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}")
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True
    )
    def embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using Ollama's embedding API.
        
        Args:
            texts: List of texts to embed
            model: Model to use for embedding (defaults to client's default_model)
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            ValueError: If texts list is empty
            RuntimeError: On API errors
        """
        if not texts:
            return []
            
        # Process texts one by one since Ollama API doesn't support batch embedding
        embeddings = []
        for text in texts:
            try:
                embedding = self.embed(text, model=model)
                embeddings.append(embedding)
            except Exception as e:
                log.error(f"Failed to embed text: {e}")
                # Use a zero vector as fallback
                dimension = 4096  # Default dimension for Ollama embeddings
                if embeddings and len(embeddings[0]) > 0:
                    dimension = len(embeddings[0])
                embeddings.append([0.0] * dimension)
                
        return embeddings
