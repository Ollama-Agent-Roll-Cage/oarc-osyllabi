"""
Tests for factory decorator functionality.
"""
import unittest
from argparse import Namespace
from osyllabi.utils.decorators.factory import factory

# Test classes
@factory
class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

@factory 
class CliProcessor:
    def __init__(self, args=None):
        self._result = None
        if args and hasattr(args, 'success'):
            if args.success:
                self._result = (0, "success")
            else:
                self._result = (1, "failure")
    
    def __eq__(self, other):
        """Make this class work with equality comparisons."""
        if isinstance(other, tuple) and hasattr(self, '_result'):
            return self._result == other
        return NotImplemented

class TestFactoryDecorator(unittest.TestCase):
    """Test cases for factory decorator."""

    def test_basic_instantiation(self):
        """Test basic factory creation."""
        # Create instance using factory method
        person = Person.create(name="John", age=30)
        
        # Verify instance
        self.assertIsInstance(person, Person)
        self.assertEqual(person.name, "John")
        self.assertEqual(person.age, 30)

    def test_factory_with_args(self):
        """Test factory with positional arguments."""
        person = Person.create("Jane", 25)
        self.assertEqual(person.name, "Jane")
        self.assertEqual(person.age, 25)

    def test_factory_with_kwargs(self):
        """Test factory with keyword arguments."""
        person = Person.create(age=35, name="Bob")
        self.assertEqual(person.name, "Bob")
        self.assertEqual(person.age, 35)

    def test_cli_args_processing(self):
        """Test factory with CLI args processing."""
        # Test successful case
        args = Namespace(success=True)
        result = CliProcessor.create(args=args)
        self.assertEqual(result, (0, "success"))

        # Test failure case
        args = Namespace(success=False)
        result = CliProcessor.create(args=args)
        self.assertEqual(result, (1, "failure"))

    def test_factory_preservation(self):
        """Test that factory preserves original class functionality."""
        # Direct instantiation should still work
        person = Person("Direct", 40)
        self.assertIsInstance(person, Person)
        self.assertEqual(person.name, "Direct")
        self.assertEqual(person.age, 40)

if __name__ == "__main__":
    unittest.main()
