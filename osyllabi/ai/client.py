"""
AI client for interfacing with Ollama models using the official Ollama package.
"""
import json
import time
import os
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar

from ollama import AsyncClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from osyllabi.utils.log import log
from osyllabi.utils.decorators.singleton import singleton
from osyllabi.config import AI_CONFIG

# Type variable for generic return type annotations
T = TypeVar('T')

@singleton
class OllamaClient:
    """
    Async client for interacting with Ollama API.
    
    This class provides methods for generating text completions,
    chat responses, and embeddings using Ollama's local API.
    """
    
    @classmethod
    async def validate_ollama(cls) -> None:
        """Validate Ollama availability asynchronously."""
        from osyllabi.utils.utils import check_for_ollama
        check_for_ollama(raise_error=True)
        
    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Base URL for Ollama API
            default_model: Default model to use for requests
            default_temperature: Default temperature for generation
            default_max_tokens: Default maximum tokens for generation
            output_dir: Custom output directory path
            
        Raises:
            RuntimeError: If Ollama server is not available
        """
        # Use configuration values or defaults
        self.base_url = base_url or AI_CONFIG.get('ollama_api_url', 'http://localhost:11434')
        self.default_model = default_model or AI_CONFIG.get('default_model', 'llama3')
        self.default_temperature = default_temperature if default_temperature is not None else AI_CONFIG.get('temperature', 0.7)
        self.default_max_tokens = 4000  # Always use 4000 as default max_tokens
        
        # Create output directory with run ID based on epoch time
        self.run_id = str(int(time.time()))
        base_output_dir = output_dir or os.path.join(os.getcwd(), "output")
        self.output_dir = os.path.join(base_output_dir, self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)
        log.info(f"Created output directory for run: {self.output_dir}")
        
        # Initialize Ollama client
        self.client = AsyncClient(host=self.base_url)
        
    async def initialize(self):
        """Initialize the client and validate server connection."""
        # Validate Ollama availability
        await OllamaClient.validate_ollama()
        
        # Test connection to server
        await self._validate_server()
        
    async def _validate_server(self) -> None:
        """
        Validate connection to Ollama server.
        
        Raises:
            RuntimeError: If server is not available
        """
        try:
            # Try to list models as a connection test
            await self.client.list()
            log.info(f"Successfully connected to Ollama server at {self.base_url}")
        except Exception as e:
            error_msg = f"Failed to connect to Ollama server at {self.base_url}: {e}"
            log.error(error_msg)
            raise RuntimeError(error_msg)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def generate(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None,
        stream: bool = False,  # Parameter kept for compatibility, but always False
        callback: Optional[Callable[[str], None]] = None  # Parameter kept for compatibility, but not used
    ) -> str:
        """
        Generate a completion from a prompt using chat API.
        
        Args:
            prompt: The prompt to generate from
            model: Model to use (defaults to client's default_model)
            system: Optional system prompt for context
            temperature: Controls randomness (0-1)
            max_tokens: Maximum tokens to generate
            stream: Not used - always set to False
            callback: Not used
            
        Returns:
            The generated text
            
        Raises:
            ValueError: If the prompt is empty
            RuntimeError: On API errors
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        # Convert parameters to chat format
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # Call chat API (never use streaming)
        response = await self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False  # Never use streaming
        )
        
        # Extract content from chat response
        if response and "message" in response and "content" in response["message"]:
            return response["message"]["content"].strip()
        return ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def chat(
        self, 
        messages: List[Dict[str, Any]], 
        model: Optional[str] = None,
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None,
        stream: bool = False,  # Parameter kept for compatibility, but always False
        callback: Optional[Callable[[Dict[str, Any]], None]] = None  # Parameter kept for compatibility, but not used
    ) -> Dict[str, Any]:
        """
        Generate a chat completion from a list of messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Model to use (defaults to client's default_model)
            temperature: Controls randomness (0-1)
            max_tokens: Maximum tokens to generate
            stream: Not used - always set to False
            callback: Not used
            
        Returns:
            Dict with response
            
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
        num_predict = 4000  # Always use 4000 tokens
        
        log.debug(f"Sending chat request to model {model_name}")
        
        try:
            # Use the ollama library's chat method
            response = await self.client.chat(
                model=model_name,
                messages=messages,
                options={
                    "temperature": temp,
                    "num_predict": num_predict
                }
            )
            return response
                
        except Exception as e:
            log.error(f"Chat API request failed: {e}")
            raise RuntimeError(f"Failed to generate chat response: {e}")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models in Ollama.
        
        Returns:
            List of dictionaries containing model information
            
        Raises:
            RuntimeError: On API errors
        """
        try:
            models = await self.client.list()
            return models["models"]
        except Exception as e:
            log.error(f"Failed to list models: {e}")
            raise RuntimeError(f"Failed to list models: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def embed(
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
        
        log.debug(f"Sending embedding request using model {model_name}")
        
        try:
            # Use the ollama library's embed method
            response = await self.client.embeddings(model=model_name, prompt=text)
            
            # Extract the embedding from the response
            embedding = response.get("embedding", [])
            if not embedding:
                log.warning("Received empty embedding from Ollama API")
                
            return embedding
                
        except Exception as e:
            log.error(f"Embedding API request failed: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}")
            
    async def embed_batch(
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
            
        model_name = model or self.default_model
        
        try:
            # Use the ollama library's batch embedding method
            response = await self.client.embeddings(model=model_name, prompt=texts)
            
            # Extract embeddings from response
            embeddings = response.get("embeddings", [])
            if not embeddings:
                log.warning("Received empty embeddings from Ollama API")
                return [[0.0] * 1536] * len(texts)  # Return empty embeddings
                
            return embeddings
            
        except Exception as e:
            log.error(f"Failed to process batch embeddings: {e}")
            raise RuntimeError(f"Failed to generate batch embeddings: {e}")
    
    def get_output_path(self, filename: str) -> str:
        """
        Get a path in the run's output directory.
        
        Args:
            filename: Name of the file to create in the output directory
            
        Returns:
            Full path to the file in the output directory
        """
        return os.path.join(self.output_dir, filename)
    
    def save_to_output(self, filename: str, content: Union[str, Dict, List]) -> str:
        """
        Save content to a file in the output directory.
        
        Args:
            filename: Name of the file to save
            content: Content to save (string or JSON-serializable object)
            
        Returns:
            Path to the saved file
        """
        filepath = self.get_output_path(filename)
        
        # Create parent directories if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save content based on its type
        if isinstance(content, (dict, list)):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(str(content))
                
        log.debug(f"Saved content to {filepath}")
        return filepath

    async def pull_model(
        self,
        model: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> bool:
        """
        Pull a model from Ollama registry with progress tracking.
        
        Args:
            model: Name of the model to pull
            progress_callback: Optional callback to report progress
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            RuntimeError: On API errors
        """
        log.info(f"Pulling model {model} from Ollama registry")
        
        try:
            # Use the ollama library's pull method with progress callback
            # Note: model is passed as a positional argument, stream as a keyword argument
            async for progress in await self.client.pull(model, stream=True):
                if progress_callback:
                    progress_callback(progress)
                    
                # If progress has a 'completed' field, we can track it
                if "digest" in progress and "completed" in progress and "total" in progress:
                    digest = progress.get("digest", "")
                    completed = progress.get("completed", 0)
                    total = progress.get("total", 0)
                    
                    # Log progress
                    if total > 0:
                        percent = (completed / total) * 100
                        log.debug(f"Pull progress: {percent:.1f}% ({completed}/{total})")
                
                # Log status updates
                if "status" in progress:
                    log.info(f"Pull status: {progress['status']}")
            
            return True
            
        except Exception as e:
            log.error(f"Failed to pull model {model}: {e}")
            raise RuntimeError(f"Failed to pull model: {e}")
