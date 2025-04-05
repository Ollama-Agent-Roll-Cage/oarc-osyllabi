"""
Base agent class for all AI agents in Osyllabi.

This module provides the abstract base class that all specialized agents
should extend, defining core functionality and interfaces.
"""
import abc
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from osyllabi.utils.log import (
    log, INFO, with_context, log_at_level
)


class Agent(abc.ABC):
    """
    Abstract base class for all AI agents.
    
    This class defines the common interface and functionality that
    all agent implementations should provide.
    """
    
    def __init__(
        self,
        name: str,
        model: str = "llama3.1:latest",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        version: str = "1.0"
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name/identifier
            model: LLM model to use for generation
            temperature: Temperature setting for generation
            max_tokens: Maximum tokens for generation
            version: Agent version string
        """
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.version = version
        
        # Session identifier
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        
        # Storage for results and metadata
        self.results = {}
        self.metadata = {}
        self.history = []
        
        # Performance tracking
        self.execution_stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_generation_time": 0.0,
            "avg_response_time": 0.0
        }
        
        # Agent state tracking 
        self.state = {
            "status": "initialized",
            "last_active": datetime.now(),
            "current_task": None
        }
        
    def set_model(self, model: str) -> None:
        """
        Set the model for this agent.
        
        Args:
            model: Model name to use
        """
        self.model = model
        self.log_activity(f"Model changed to: {model}")
        
    def set_temperature(self, temperature: float) -> None:
        """
        Set the temperature for generation.
        
        Args:
            temperature: Temperature value (0.0-1.0)
        """
        self.temperature = max(0.0, min(1.0, temperature))
        self.log_activity(f"Temperature set to: {self.temperature}")
        
    def set_max_tokens(self, max_tokens: int) -> None:
        """
        Set maximum tokens for generation.
        
        Args:
            max_tokens: Maximum number of tokens
        """
        self.max_tokens = max_tokens
        self.log_activity(f"Max tokens set to: {max_tokens}")
        
    def store_result(self, key: str, value: Any) -> None:
        """
        Store a result from the agent's operations.
        
        Args:
            key: Result key/identifier
            value: Result value
        """
        self.results[key] = value
        self._add_to_history("result_stored", {"key": key, "value_type": type(value).__name__})
        
    def get_result(self, key: str, default: Any = None) -> Any:
        """
        Get a stored result.
        
        Args:
            key: Result key/identifier
            default: Default value if key doesn't exist
            
        Returns:
            The stored result or default value
        """
        result = self.results.get(key, default)
        self._add_to_history("result_retrieved", {"key": key, "found": key in self.results})
        return result
        
    def store_metadata(self, key: str, value: Any) -> None:
        """
        Store metadata about agent operations.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
        
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get stored metadata.
        
        Args:
            key: Metadata key
            default: Default value if key doesn't exist
            
        Returns:
            The metadata value or default
        """
        return self.metadata.get(key, default)
    
    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate content based on a prompt.
        
        Args:
            prompt: Prompt to generate from
            
        Returns:
            Generated content
        """
        pass
    
    @abc.abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Process input data and produce output.
        
        Args:
            input_data: Data to process
            
        Returns:
            Processed output
        """
        pass
    
    def log_activity(self, activity: str, level: str = "info") -> None:
        """
        Log agent activity with appropriate context.
        
        Args:
            activity: Activity description
            level: Log level (debug, info, warning, error)
        """
        with with_context(agent=self.name, session=self.session_id):
            log_at_level(getattr(log, level.upper(), INFO), activity)
        
        # Update activity timestamp and history
        self.state["last_active"] = datetime.now()
        self._add_to_history("activity_logged", {"message": activity, "level": level})
    
    def _add_to_history(self, action_type: str, details: Dict[str, Any]) -> None:
        """
        Add an entry to the agent's history.
        
        Args:
            action_type: Type of action
            details: Action details
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action_type,
            "details": details
        }
        self.history.append(entry)
        
    def update_stats(self, tokens_used: int, generation_time: float) -> None:
        """
        Update agent performance statistics.
        
        Args:
            tokens_used: Number of tokens used
            generation_time: Time taken for generation in seconds
        """
        self.execution_stats["total_calls"] += 1
        self.execution_stats["total_tokens"] += tokens_used
        self.execution_stats["total_generation_time"] += generation_time
        self.execution_stats["avg_response_time"] = (
            self.execution_stats["total_generation_time"] / 
            self.execution_stats["total_calls"]
        )
        
    def get_status_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive status report for this agent.
        
        Returns:
            Dict[str, Any]: Status information
        """
        return {
            "agent": {
                "name": self.name,
                "model": self.model,
                "version": self.version,
                "session_id": self.session_id,
                "created_at": self.created_at.isoformat()
            },
            "state": self.state,
            "performance": self.execution_stats,
            "storage": {
                "results_count": len(self.results),
                "metadata_count": len(self.metadata),
                "history_entries": len(self.history)
            }
        }
    
    def set_state(self, status: str, task: Optional[str] = None) -> None:
        """
        Update the agent's state.
        
        Args:
            status: New status string
            task: Current task description (optional)
        """
        previous_status = self.state["status"]
        self.state["status"] = status
        self.state["last_active"] = datetime.now()
        
        if task is not None:
            self.state["current_task"] = task
            
        self._add_to_history("state_changed", {
            "previous": previous_status, 
            "current": status,
            "task": task
        })
        
    def get_history(self, limit: int = 10, action_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get agent history entries.
        
        Args:
            limit: Maximum number of entries to return
            action_type: Optional filter by action type
            
        Returns:
            List[Dict[str, Any]]: History entries
        """
        if action_type:
            filtered = [entry for entry in self.history if entry["action"] == action_type]
            return filtered[-limit:] if limit > 0 else filtered
        else:
            return self.history[-limit:] if limit > 0 else self.history.copy()
    
    def clear_history(self) -> None:
        """Clear the agent's history to free memory."""
        history_size = len(self.history)
        self.history = []
        self.log_activity(f"Cleared history ({history_size} entries)")
