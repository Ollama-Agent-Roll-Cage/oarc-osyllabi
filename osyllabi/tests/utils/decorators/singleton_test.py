"""Tests for the singleton decorator."""
import unittest

from osyllabi.utils.decorators.singleton import singleton


@singleton
class DatabaseConnection:
    def __init__(self, host: str = "localhost"):
        self.host = host
        self.connected = False
        self._initialized = False
    
    def connect(self):
        self.connected = True


@singleton
class Configuration:
    def __init__(self, settings=None):
        self.settings = settings or {}
        if isinstance(settings, dict) and settings:
            self.settings = settings
        self._initialized = False

    def update_settings(self, new_settings):
        self.settings.update(new_settings)


@singleton
class Service:
    def __init__(self, name: str = "default"):
        self.name = name
        self.running = False
    
    def start(self):
        self.running = True
    
    def stop(self):
        self.running = False


class TestSingletonDecorator(unittest.TestCase):
    """Test cases for the singleton decorator."""
    
    def setUp(self):
        """Reset the singleton registry before each test."""
        # Clear singleton instances to start fresh for each test
        from osyllabi.utils.decorators.singleton import _instances
        _instances.clear()

    def test_single_instance(self):
        """Test that only one instance is created."""
        db1 = DatabaseConnection()
        db2 = DatabaseConnection()
        
        self.assertIs(db1, db2)
    
    def test_initialization_once(self):
        """Test that initialization happens only once."""
        # First instance
        db1 = DatabaseConnection("host1")
        self.assertEqual(db1.host, "host1")
        
        # Second instance with different parameters - should be ignored
        db2 = DatabaseConnection("host2")
        self.assertEqual(db2.host, "host1")  # Should still have original value
        
        # Both should be the same instance
        self.assertIs(db1, db2)
    
    def test_singleton_state_modification(self):
        """Test that state modifications affect the singleton."""
        db = DatabaseConnection()
        self.assertFalse(db.connected)
        
        # Modify state
        db.connect()
        self.assertTrue(db.connected)
        
        # Get another reference to the singleton
        another_db = DatabaseConnection()
        
        # Should reflect the modified state
        self.assertTrue(another_db.connected)
        self.assertIs(db, another_db)
    
    def test_multiple_singletons(self):
        """Test that different singleton classes are independent."""
        db = DatabaseConnection()
        config = Configuration()
        
        self.assertIsNot(db, config)
        
        # Each class has its own instance
        db2 = DatabaseConnection()
        config2 = Configuration()
        
        self.assertIs(db, db2)
        self.assertIs(config, config2)
    
    def test_singleton_with_args(self):
        """Test singleton instantiation with arguments."""
        # First instance with args
        conf1 = Configuration({"debug": True})

        # Second instance - args should be ignored
        conf2 = Configuration({"debug": False})

        # Should be same instance with original settings
        self.assertIs(conf1, conf2)
        self.assertEqual(conf1.settings["debug"], True)
