"""
Vector database for RAG capabilities in Osyllabi.

This module provides a lightweight SQLite-based vector database for storing
and retrieving embeddings for Retrieval-Augmented Generation.
"""
import json
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional

from osyllabi.utils.log import log


class VectorDatabase:
    """
    SQLite-based vector database for curriculum RAG.
    
    This class provides functionality to store and retrieve vectors and
    document chunks for retrieval-augmented generation.
    """
    
    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize the database
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Initialize schema
        self._initialize_schema()
        
    def _initialize_schema(self) -> None:
        """Set up database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Create chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT,
                metadata TEXT
            )
        """)
        
        # Create vectors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY,
                chunk_id INTEGER NOT NULL,
                vector BLOB NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id)
            )
        """)
        
        # Create indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
        
        self.conn.commit()
        
    def add_document(
        self,
        text_chunks: List[str],
        vectors: List[List[float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        """
        Add document chunks and their vectors to the database.
        
        Args:
            text_chunks: List of text chunks
            vectors: List of vector embeddings
            metadata: Additional metadata for the document
            
        Returns:
            List of chunk IDs
        """
        if len(text_chunks) != len(vectors):
            raise ValueError("Number of chunks and vectors must match")
            
        # Prepare metadata
        meta_json = json.dumps(metadata) if metadata else None
        source = metadata.get("source", "unknown") if metadata else "unknown"
        
        # Add to database
        cursor = self.conn.cursor()
        chunk_ids = []
        
        for chunk, vector in zip(text_chunks, vectors):
            # Insert chunk
            cursor.execute(
                "INSERT INTO chunks (text, source, metadata) VALUES (?, ?, ?)",
                (chunk, source, meta_json)
            )
            chunk_id = cursor.lastrowid
            chunk_ids.append(chunk_id)
            
            # Insert vector as binary blob
            vector_bytes = np.array(vector, dtype=np.float32).tobytes()
            cursor.execute(
                "INSERT INTO vectors (chunk_id, vector) VALUES (?, ?)",
                (chunk_id, vector_bytes)
            )
        
        self.conn.commit()
        return chunk_ids
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Find most similar vectors using cosine similarity.
        
        Args:
            query_vector: Query embedding
            top_k: Maximum number of results
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of dictionaries with chunks and metadata
        """
        # Convert query vector to numpy array
        query_np = np.array(query_vector, dtype=np.float32)
        
        # Fetch all vectors for comparison
        # In a production system, this would use an approximate nearest
        # neighbor library like FAISS or HNSWLib instead of loading all vectors
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT v.id, v.chunk_id, v.vector, c.text, c.source, c.metadata
            FROM vectors v
            JOIN chunks c ON v.chunk_id = c.id
        """)
        
        results = []
        for row in cursor.fetchall():
            vec_id, chunk_id, vector_bytes, text, source, meta_str = row
            
            # Convert binary vector to numpy array
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_np, vector)
            
            # Only include results above threshold
            if similarity >= threshold:
                # Parse metadata
                metadata = json.loads(meta_str) if meta_str else {}
                
                results.append({
                    "id": chunk_id,
                    "text": text,
                    "source": source,
                    "metadata": metadata,
                    "similarity": float(similarity)
                })
        
        # Sort by similarity (descending) and take top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def get_document_count(self) -> int:
        """Get the number of unique documents in the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT source) FROM chunks")
        return cursor.fetchone()[0]
    
    def get_chunk_count(self) -> int:
        """Get the total number of chunks in the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        return cursor.fetchone()[0]
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            
    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()
