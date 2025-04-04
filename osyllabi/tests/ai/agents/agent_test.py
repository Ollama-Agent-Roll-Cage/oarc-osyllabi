"""
Unit tests for the base Agent class.
"""
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from typing import Any

from osyllabi.ai.agents.agent import Agent


# Concrete implementation for testing
class MockAgent(Agent):
    def __init__(self, name="test_agent", model="test_model", temperature=0.7, max_tokens=1000, version="1.0"):
        super().__init__(name=name, model=model, temperature=temperature, max_tokens=max_tokens, version=version)
        
    def generate(self, prompt: str) -> str:
        return f"Generated content for: {prompt}"
    
    def process(self, input_data: Any) -> Any:
        return f"Processed: {input_data}"


class TestAgent(unittest.TestCase):
    """Test cases for the Agent base class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = MockAgent(
            name="test_agent",
            model="test_model",
            temperature=0.5,
            max_tokens=100,
            version="1.0"
        )

    def test_initialization(self):
        """Test agent initialization and attributes."""
        self.assertEqual(self.agent.name, "test_agent")
        self.assertEqual(self.agent.model, "test_model")
        self.assertEqual(self.agent.temperature, 0.5)
        self.assertEqual(self.agent.max_tokens, 100)
        self.assertEqual(self.agent.version, "1.0")
        self.assertEqual(self.agent.state["status"], "initialized")
        
    def test_set_model(self):
        """Test model setting."""
        self.agent.set_model("new_model")
        self.assertEqual(self.agent.model, "new_model")
        self.assertTrue(any(entry["action"] == "activity_logged" 
                          for entry in self.agent.history))
        
    def test_set_temperature(self):
        """Test temperature setting with bounds checking.""" 
        self.agent.set_temperature(0.8)
        self.assertEqual(self.agent.temperature, 0.8)
        
        # Test bounds
        self.agent.set_temperature(1.5)
        self.assertEqual(self.agent.temperature, 1.0)
        
        self.agent.set_temperature(-0.5)
        self.assertEqual(self.agent.temperature, 0.0)
        
    def test_store_and_get_result(self):
        """Test result storage and retrieval."""
        test_data = {"key": "value"}
        self.agent.store_result("test_result", test_data)
        
        # Test retrieval
        result = self.agent.get_result("test_result")
        self.assertEqual(result, test_data)
        
        # Test default value for missing key
        missing = self.agent.get_result("nonexistent", default="default")
        self.assertEqual(missing, "default")
        
    def test_metadata_operations(self):
        """Test metadata storage and retrieval.""" 
        self.agent.store_metadata("test_meta", "meta_value")
        self.assertEqual(
            self.agent.get_metadata("test_meta"),
            "meta_value"
        )
        
    def test_state_management(self):
        """Test agent state management."""
        self.agent.set_state("processing", "test_task")
        
        self.assertEqual(self.agent.state["status"], "processing")
        self.assertEqual(self.agent.state["current_task"], "test_task")
        
        # Verify history was updated
        last_state_change = next(
            entry for entry in reversed(self.agent.history)
            if entry["action"] == "state_changed"
        )
        self.assertEqual(last_state_change["details"]["current"], "processing")
        
    def test_history_management(self):
        """Test history tracking and retrieval."""
        # Generate some history
        self.agent.log_activity("test activity 1")
        self.agent.log_activity("test activity 2")
        
        # Test history retrieval with limit
        history = self.agent.get_history(limit=1)
        self.assertEqual(len(history), 1)
        
        # Test filtering by action type
        self.agent.store_result("test", "value")
        filtered = self.agent.get_history(action_type="result_stored")
        self.assertTrue(all(entry["action"] == "result_stored" 
                          for entry in filtered))
        
    def test_performance_tracking(self):
        """Test performance statistics tracking."""
        self.agent.update_stats(tokens_used=50, generation_time=1.5)
        self.agent.update_stats(tokens_used=30, generation_time=0.5)
        
        stats = self.agent.execution_stats
        self.assertEqual(stats["total_calls"], 2)
        self.assertEqual(stats["total_tokens"], 80)
        self.assertEqual(stats["total_generation_time"], 2.0)
        self.assertEqual(stats["avg_response_time"], 1.0)
        
    def test_status_report(self):
        """Test status report generation."""
        report = self.agent.get_status_report()
        
        self.assertEqual(report["agent"]["name"], "test_agent")
        self.assertEqual(report["agent"]["model"], "test_model")
        self.assertEqual(report["agent"]["version"], "1.0")
        self.assertTrue("session_id" in report["agent"])
        self.assertTrue("created_at" in report["agent"])
        
    def test_history_clearing(self):
        """Test history clearing functionality."""
        self.agent.log_activity("test activity")
        self.assertTrue(len(self.agent.history) > 0)
        
        self.agent.clear_history()
        self.assertEqual(len(self.agent.history), 1)  # Just the clear activity log
        
    def test_log_activity_levels(self):
        """Test activity logging at different levels."""
        with self.assertLogs(level='DEBUG') as log_context:
            self.agent.log_activity("debug message", level="debug")
            self.agent.log_activity("info message", level="info")
            self.agent.log_activity("warning message", level="warning")
            self.agent.log_activity("error message", level="error")
            
        self.assertEqual(len(log_context.records), 4)
        
    def test_abstract_methods(self):
        """Test that concrete implementation works.""" 
        result = self.agent.generate("test")
        self.assertEqual(result, "Generated content for: test")
        
        result = self.agent.process("data")
        self.assertEqual(result, "Processed: data")


if __name__ == '__main__':
    unittest.main()
