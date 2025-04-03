"""
Constants used throughout the Osyllabi project.
"""

# Log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s%(context)s"

# Exit codes
SUCCESS = 0
FAILURE = 1

# Version
VERSION = "0.1.0"

# Ollama configuration
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:latest"
DEFAULT_EMBEDDING_MODEL = "llama3.1:latest"

# Default resource limits
DEFAULT_MAX_CONCURRENT_REQUESTS = 5
DEFAULT_MAX_FILE_SIZE_MB = 10.0
DEFAULT_MAX_CONTENT_LENGTH = 10000

# Default filesystem paths
DEFAULT_OUTPUT_DIRNAME = "osyllabi_output"
DEFAULT_DATA_DIRNAME = "osyllabi_data"
DEFAULT_CONFIG_DIRNAME = ".osyllabi"

# Default AI generation parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4000
DEFAULT_TOP_P = 1.0