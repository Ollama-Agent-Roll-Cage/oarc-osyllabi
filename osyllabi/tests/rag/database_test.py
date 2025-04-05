"""
Unit tests for the vector database functionality.
"""
import unittest
import tempfile
import warnings
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from osyllabi.rag.database import VectorDatabase

# Filter out the NumPy deprecation warning from FAISS
warnings.filterwarnings(
    "ignore",
    message="numpy.core._multiarray_umath is deprecated",
    category=DeprecationWarning
)

class TestVectorDatabase(unittest.TestCase):
    """Rewritten tests for our in-memory pandas VectorDatabase."""

    def setUp(self):
        self.db = VectorDatabase()

    def test_add_document(self):
        chunks = ["Chunk 1", "Chunk 2"]
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        chunk_ids = self.db.add_document(chunks, vectors, {}, source="doc.txt")
        self.assertEqual(len(chunk_ids), 2)

    def test_get_document_sources(self):
        self.db.add_document(["Sample"], [[0.1, 0.2]], {}, "docA.txt")
        self.db.add_document(["Another"], [[0.3, 0.4]], {}, "docB.txt")
        sources = self.db.get_document_sources()
        self.assertIn("docA.txt", sources)
        self.assertIn("docB.txt", sources)

    def test_get_stats(self):
        self.db.add_document(["1.1", "1.2"], [[0.1, 0.2], [0.3, 0.4]], {}, "doc1.txt")
        stats = self.db.get_stats()
        self.assertEqual(stats["document_count"], 1)
        self.assertEqual(stats["chunk_count"], 2)
        self.assertEqual(stats["embedding_dimension"], 2)

    def test_remove_document(self):
        self.db.add_document(["Text 1"], [[0.1, 0.2]], {}, "test_doc.txt")
        self.db.remove_document("test_doc.txt")
        self.assertEqual(len(self.db.data), 0)

    def test_search(self):
        self.db.add_document(["Text A", "Text B"], [[0.0, 1.0], [0.8, 0.1]], {}, "doc.txt")
        results = self.db.search([0.0, 1.0], top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("Text A", results[0]["text"])

    def test_search_with_source_filter(self):
        self.db.add_document(["Chunk from doc1"], [[0.1, 0.2, 0.3]], {}, source="doc1.txt")
        self.db.add_document(["Chunk from doc2"], [[0.5, 0.6, 0.7]], {}, source="doc2.txt")
        results = self.db.search([0.1, 0.2, 0.3], top_k=1, source_filter="doc2.txt")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "doc2.txt")

    def test_search_with_threshold(self):
        self.db.add_document(["T1", "T2"], [[1.0, 0.0], [0.0, 1.0]], {}, "doc.txt")
        # T1 is 100% similar to query if it's the same vector
        results = self.db.search([1.0, 0.0], threshold=0.5)
        self.assertEqual(len(results), 1)
        self.assertIn("T1", results[0]["text"])

    def test_close_and_reopen(self):
        self.db.add_document(["CloseTest"], [[0.5, 0.5]], {}, "close_doc.txt")
        self.db.close()
        newdb = VectorDatabase()  # fresh in-memory
        self.assertEqual(len(newdb.data), 0)


if __name__ == "__main__":
    unittest.main()
