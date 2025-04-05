"""
Tests for data file extraction functionality.
"""
import unittest
from unittest.mock import patch, MagicMock, Mock, mock_open
from pathlib import Path
import json
import io

from osyllabi.generator.resource.data import DataFileExtractor


class TestDataFileExtractor(unittest.TestCase):
    """Test cases for the DataFileExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = DataFileExtractor()
        
    def test_initialization(self):
        """Test initialization of DataFileExtractor."""
        self.assertIn('.json', self.extractor.format_handlers)
        self.assertIn('.csv', self.extractor.format_handlers)
        self.assertIn('.yml', self.extractor.format_handlers)
        self.assertIn('.yaml', self.extractor.format_handlers)
        self.assertIn('.docx', self.extractor.format_handlers)
        
    def test_supports_method(self):
        """Test supports method for different file types."""
        mock_path = MagicMock(spec=Path)
        
        # Test JSON support
        mock_path.is_file.return_value = True
        mock_path.suffix.lower.return_value = '.json'
        self.assertTrue(self.extractor.supports(mock_path))
        
        # Test CSV support
        mock_path.suffix.lower.return_value = '.csv'
        self.assertTrue(self.extractor.supports(mock_path))
        
        # Test YAML support
        mock_path.suffix.lower.return_value = '.yml'
        self.assertTrue(self.extractor.supports(mock_path))
        
        # Test unsupported file type
        mock_path.suffix.lower.return_value = '.txt'
        self.assertFalse(self.extractor.supports(mock_path))
        
        # Test non-file resource
        mock_path.is_file.return_value = False
        mock_path.suffix.lower.return_value = '.json'
        self.assertFalse(self.extractor.supports(mock_path))
        
    @patch('builtins.open', new_callable=mock_open, read_data='{"name": "Test", "value": 42}')
    @patch('pathlib.Path')
    def test_handle_json(self, mock_path_class, mock_file):
        """Test JSON file handling."""
        # Setup mock
        mock_path = mock_path_class.return_value
        mock_path.suffix.lower.return_value = '.json'
        mock_path.stat.return_value.st_size = 100
        
        # Extract content
        result = self.extractor._handle_json(mock_path)
        
        # Verify extraction results
        self.assertEqual(result["content_type"], "json")
        self.assertIn("```json", result["content"])
        self.assertEqual(result["metadata"]["structure"]["type"], "object")
        self.assertEqual(result["metadata"]["structure"]["keys_count"], 2)
        self.assertIn("name", result["metadata"]["structure"]["keys"])
        self.assertIn("value", result["metadata"]["structure"]["keys"])
        
    @patch('builtins.open', new_callable=mock_open, read_data='Invalid JSON')
    @patch('pathlib.Path')
    def test_handle_invalid_json(self, mock_path_class, mock_file):
        """Test handling of invalid JSON file."""
        # Setup mock
        mock_path = mock_path_class.return_value
        mock_path.suffix.lower.return_value = '.json'
        mock_path.stat.return_value.st_size = 100
        
        # Extract content from invalid JSON
        result = self.extractor._handle_json(mock_path)
        
        # Verify extraction results for invalid JSON
        self.assertEqual(result["content_type"], "text")
        self.assertIn("Invalid JSON content", result["content"])
        self.assertIn("error", result["metadata"])
        
    @patch('builtins.open', new_callable=mock_open, read_data='')
    @patch('pathlib.Path')
    @patch('osyllabi.generator.resource.data.pd')
    @patch('osyllabi.generator.resource.data.PANDAS_AVAILABLE')
    def test_handle_csv_with_pandas(self, pandas_available_mock, pd_mock, path_mock, open_mock):
        """Test CSV handling with pandas available."""
        # Set pandas available
        pandas_available_mock.return_value = True
        self.extractor.pandas_available = True

        # Setup mock path
        mock_path = path_mock.return_value
        mock_path.suffix.lower.return_value = '.csv'
        mock_path.stat.return_value.st_size = 200
        mock_path.name = 'test.csv'
        
        # Setup mock DataFrame
        mock_df = MagicMock()
        mock_df.__len__.return_value = 10
        mock_df.columns = ['name', 'age', 'city']
        mock_df.select_dtypes.return_value.columns = ['age']
        mock_df.head.return_value.to_string.return_value = "  name  age city\n0 Alice  25  NYC\n1 Bob    30  LA"
        
        # Mock stats for numeric column
        mock_col_stats = MagicMock()
        mock_col_stats.min.return_value = 25
        mock_col_stats.max.return_value = 50
        mock_col_stats.mean.return_value = 35
        mock_col_stats.median.return_value = 35
        mock_df.__getitem__.return_value = mock_col_stats
        
        # Setup pandas read_csv to return our mock DataFrame
        pd_mock.read_csv.return_value = mock_df
        
        # Extract content
        result = self.extractor._handle_csv(mock_path)
        
        # Verify extraction results
        self.assertEqual(result["content_type"], "csv")
        self.assertEqual(result["title"], 'test.csv')
        self.assertIn("CSV Data Analysis", result["content"])
        self.assertIn("Rows: 10", result["content"])
        self.assertIn("Columns: 3", result["content"])
        self.assertEqual(result["metadata"]["row_count"], 10)
        self.assertEqual(result["metadata"]["column_count"], 3)
        self.assertEqual(result["metadata"]["headers"], ['name', 'age', 'city'])
        self.assertIn("statistics", result["metadata"])
        
    @patch('osyllabi.generator.resource.data.csv.reader')
    @patch('builtins.open', new_callable=mock_open, read_data='name,age,city\nAlice,25,NYC\nBob,30,LA')
    @patch('pathlib.Path')
    def test_handle_csv_without_pandas(self, mock_path_class, mock_file, mock_csv_reader):
        """Test CSV handling without pandas."""
        # Setup mock path
        mock_path = mock_path_class.return_value
        mock_path.suffix.lower.return_value = '.csv'
        mock_path.stat.return_value.st_size = 200
        mock_path.name = 'test.csv'
        
        # Setup CSV reader mock
        mock_csv_reader.return_value = [
            ['name', 'age', 'city'],
            ['Alice', '25', 'NYC'],
            ['Bob', '30', 'LA']
        ]
        
        # Make pandas unavailable for CSV handling
        self.extractor.pandas_available = False
        
        # Extract content
        result = self.extractor._handle_csv(mock_path)
        
        # Verify extraction results
        self.assertEqual(result["content_type"], "csv")
        self.assertEqual(result["title"], 'test.csv')
        self.assertIn("CSV Data Analysis", result["content"])
        self.assertIn("Rows: 3", result["content"])
        self.assertIn("Columns: 3", result["content"])
        self.assertEqual(result["metadata"]["row_count"], 3)
        self.assertEqual(result["metadata"]["column_count"], 3)
        self.assertEqual(result["metadata"]["headers"], ['name', 'age', 'city'])
        
    @patch('pathlib.Path')
    def test_analyze_json_object(self, mock_path_class):
        """Test JSON structure analysis for objects."""
        json_data = {
            "name": "Test Object",
            "values": [1, 2, 3],
            "nested": {"key": "value"}
        }
        
        analysis = self.extractor._analyze_json(json_data)
        
        self.assertEqual(analysis["type"], "object")
        self.assertEqual(analysis["keys_count"], 3)
        self.assertTrue(analysis["has_nested"])
        self.assertIn("name", analysis["keys"])
        self.assertIn("values", analysis["keys"])
        self.assertIn("nested", analysis["keys"])
        
    @patch('pathlib.Path')
    def test_analyze_json_array(self, mock_path_class):
        """Test JSON structure analysis for arrays."""
        json_data = [
            {"id": 1, "name": "Item 1"},
            {"id": 2, "name": "Item 2"},
            {"id": 3, "name": "Item 3"}
        ]
        
        analysis = self.extractor._analyze_json(json_data)
        
        self.assertEqual(analysis["type"], "array")
        self.assertEqual(analysis["item_count"], 3)
        self.assertTrue(analysis["has_nested"])
        self.assertIn("dict", analysis["sample_types"])
        
    @patch('pathlib.Path')
    def test_analyze_json_primitive(self, mock_path_class):
        """Test JSON structure analysis for primitive types."""
        json_data = "Simple string value"
        
        analysis = self.extractor._analyze_json(json_data)
        
        self.assertEqual(analysis["type"], "str")
        self.assertEqual(analysis["value"], "Simple string value")
        
    @patch('builtins.open', new_callable=mock_open, read_data='name: Test\nvalue: 42')
    @patch('pathlib.Path')
    @patch('osyllabi.generator.resource.data.yaml')
    @patch('osyllabi.generator.resource.data.YAML_AVAILABLE')
    def test_handle_yaml_with_pyyaml(self, yaml_available_mock, yaml_mock, path_mock, open_mock):
        """Test YAML handling with PyYAML available."""
        # Set yaml available
        yaml_available_mock.return_value = True
        self.extractor.yaml_available = True

        # Setup mock path
        mock_path = path_mock.return_value
        mock_path.suffix.lower.return_value = '.yml'
        mock_path.stat.return_value.st_size = 100
        
        # Setup yaml.safe_load mock
        yaml_mock.safe_load.return_value = {"name": "Test", "value": 42}
        
        # Extract content
        result = self.extractor._handle_yaml(mock_path)
        
        # Verify extraction results
        self.assertEqual(result["content_type"], "yaml")
        self.assertIn("```json", result["content"])
        
    @patch('builtins.open', new_callable=mock_open, read_data='name: Test\nvalue: 42')
    @patch('pathlib.Path')
    def test_handle_yaml_without_pyyaml(self, mock_path_class, mock_file):
        """Test YAML handling without PyYAML."""
        # Setup mock
        mock_path = mock_path_class.return_value
        mock_path.suffix.lower.return_value = '.yml'
        mock_path.stat.return_value.st_size = 100
        
        # Make PyYAML unavailable
        self.extractor.yaml_available = False
        
        # Extract content
        result = self.extractor._handle_yaml(mock_path)
        
        # Verify extraction results
        self.assertEqual(result["content_type"], "yaml")
        self.assertIn("```yaml", result["content"])
        
    @patch('builtins.open', new_callable=mock_open, read_data='')
    @patch('pathlib.Path')
    @patch('osyllabi.generator.resource.data.Document')
    @patch('osyllabi.generator.resource.data.DOCX_AVAILABLE')
    def test_handle_docx_with_python_docx(self, docx_available_mock, document_mock, path_mock, open_mock):
        """Test DOCX handling with python-docx available."""
        # Set docx available
        docx_available_mock.return_value = True
        self.extractor.docx_available = True

        # Setup mock path
        mock_path = path_mock.return_value
        mock_path.suffix.lower.return_value = '.docx'
        mock_path.stat.return_value.st_size = 1000
        
        # Setup mock Document
        mock_doc = MagicMock()
        document_mock.return_value = mock_doc
        
        # Setup paragraphs with some content
        mock_paragraph1 = MagicMock()
        mock_paragraph1.text = "Heading 1"
        mock_paragraph1.style.name = "Heading 1"
        
        mock_paragraph2 = MagicMock()
        mock_paragraph2.text = "Regular text paragraph"
        mock_paragraph2.style.name = "Normal"
        
        mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2]
        mock_doc.sections = [MagicMock()]
        
        # Extract content
        result = self.extractor._handle_docx(mock_path)
        
        # Verify extraction results
        self.assertEqual(result["content_type"], "docx")
        self.assertIn("Heading 1", result["content"])
        self.assertIn("Regular text paragraph", result["content"])
        self.assertEqual(result["metadata"]["paragraph_count"], 2)
        self.assertEqual(result["metadata"]["section_count"], 1)
        self.assertIn("structure", result["metadata"])
        self.assertEqual(result["metadata"]["structure"][0]["level"], 1)
        self.assertEqual(result["metadata"]["structure"][0]["title"], "Heading 1")
        
    @patch('pathlib.Path')
    def test_handle_docx_without_python_docx(self, mock_path_class):
        """Test DOCX handling without python-docx."""
        # Setup mock path
        mock_path = mock_path_class.return_value
        mock_path.suffix.lower.return_value = '.docx'
        
        # Make python-docx unavailable
        self.extractor.docx_available = False
        
        # Extract content should return None when docx module is unavailable
        result = self.extractor._handle_docx(mock_path)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
