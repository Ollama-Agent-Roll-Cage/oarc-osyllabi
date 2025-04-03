"""
Vector database for RAG capabilities in Osyllabi.

This module provides a lightweight SQLite-based vector database for storing
and retrieving embeddings for Retrieval-Augmented Generation.
"""
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from osyllabi.utils.log import log
from osyllabi.utils.vector.operations import cosine_similarity

# Default table names
DEFAULT_CHUNKS_TABLE = "chunks"
DEFAULT_VECTORS_TABLE = "vectors"


class VectorDatabase:
    """
    SQLite-based vector database for curriculum RAG.
    
    This class provides functionality to store and retrieve vectors and
    document chunks for retrieval-augmented generation.
    """
    
    def __init__(
        self,
        db_path: Union[str, Path],
        chunks_table: str = DEFAULT_CHUNKS_TABLE,
        vectors_table: str = DEFAULT_VECTORS_TABLE
    ):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file
            chunks_table: Name for the chunks table
            vectors_table: Name for the vectors table
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Table names
        self.chunks_table = chunks_table
        self.vectors_table = vectors_table
        
        # Performance optimization: set pragmas for better performance
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("PRAGMA cache_size = 10000")  # 10MB cache
        
        # Initialize schema
        self._initialize_schema()
        
        # Query stats
        self.stats = {
            "queries": 0,
            "total_time": 0.0,
            "last_query_time": 0.0
        }
        
    def _initialize_schema(self) -> None:
        """Set up database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Create chunks table with more fields for better metadata
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.chunks_table} (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT,
                metadata TEXT,
                chunk_index INTEGER,  
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                hash TEXT
            )
        """)
        
        # Create vectors table with dimension field
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.vectors_table} (
                id INTEGER PRIMARY KEY,
                chunk_id INTEGER NOT NULL,
                vector BLOB NOT NULL,
                dimension INTEGER,
                FOREIGN KEY (chunk_id) REFERENCES {self.chunks_table}(id) ON DELETE CASCADE
            )
        """)
        
        # Create indices for performance
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_chunks_source ON {self.chunks_table}(source)")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_chunks_hash ON {self.chunks_table}(hash)")
        
        self.conn.commit()
        
    def add_document(
        self,
        text_chunks: List[str],
        vectors: List[List[float]],
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        dedup: bool = True
    ) -> List[int]:
        """
        Add document chunks and their vectors to the database.
        
        Args:
            text_chunks: List of text chunks
            vectors: List of vector embeddings
            metadata: Additional metadata for the document
            source: Source identifier (file path, URL, etc.)
            dedup: Whether to check for and skip duplicate chunks
            
        Returns:
            List of chunk IDs
        """
        if len(text_chunks) != len(vectors):
            raise ValueError(f"Number of chunks ({len(text_chunks)}) and vectors ({len(vectors)}) must match")
            
        # Prepare metadata
        meta_json = json.dumps(metadata) if metadata else None
        src = source or metadata.get("source", "") if metadata else ""
        
        # Add to database
        cursor = self.conn.cursor()
        chunk_ids = []
        
        # Process in chunks for better performance with large documents
        batch_size = 1000
        for batch_start in range(0, len(text_chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(text_chunks))
            
            # Start a transaction for the batch
            cursor.execute("BEGIN TRANSACTION")
            
            try:
                for i in range(batch_start, batch_end):
                    chunk = text_chunks[i]
                    vector = vectors[i]
                    
                    # Skip empty chunks
                    if not chunk or not chunk.strip():
                        continue
                    
                    # Generate a simple hash for deduplication
                    chunk_hash = self._hash_text(chunk)
                    
                    # Check for duplicates if requested
                    chunk_id = None
                    if dedup:
                        cursor.execute(
                            f"SELECT id FROM {self.chunks_table} WHERE hash = ?",
                            (chunk_hash,)
                        )
                        result = cursor.fetchone()
                        if result:
                            chunk_id = result[0]
                            chunk_ids.append(chunk_id)
                            continue
                    
                    # Insert chunk
                    cursor.execute(
                        f"INSERT INTO {self.chunks_table} (text, source, metadata, chunk_index, hash) VALUES (?, ?, ?, ?, ?)",
                        (chunk, src, meta_json, i - batch_start, chunk_hash)
                    )
                    chunk_id = cursor.lastrowid
                    chunk_ids.append(chunk_id)
                    
                    # Insert vector as binary blob
                    vector_bytes = np.array(vector, dtype=np.float32).tobytes()
                    cursor.execute(
                        f"INSERT INTO {self.vectors_table} (chunk_id, vector, dimension) VALUES (?, ?, ?)",
                        (chunk_id, vector_bytes, len(vector))
                    )
                
                # Commit the transaction
                self.conn.commit()
                
            except Exception as e:
                # Rollback on error
                self.conn.rollback()
                log.error(f"Database error during batch insert: {e}")
                raise
        
        log.debug(f"Added {len(chunk_ids)} chunks to database")
        return chunk_ids
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        threshold: float = 0.0,
        source_filter: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find most similar vectors using cosine similarity.
        
        Args:
            query_vector: Query embedding
            top_k: Maximum number of results
            threshold: Minimum similarity score (0-1)
            source_filter: Optional source(s) to filter results
            
        Returns:
            List of dictionaries with chunks and metadata
        """
        start_time = time.time()
        
        # Convert query vector to numpy array
        query_np = np.array(query_vector, dtype=np.float32)
        
        # Apply source filter if provided
        source_clause = ""
        source_params = []
        
        if source_filter:
            if isinstance(source_filter, str):
                source_clause = f"AND c.source = ?"
                source_params = [source_filter]
            elif isinstance(source_filter, list) and source_filter:
                placeholders = ', '.join(['?'] * len(source_filter))
                source_clause = f"AND c.source IN ({placeholders})"
                source_params = source_filter
        
        # Fetch vectors for comparison
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT v.id, v.chunk_id, v.vector, c.text, c.source, c.metadata, c.chunk_index
            FROM {self.vectors_table} v
            JOIN {self.chunks_table} c ON v.chunk_id = c.id
            WHERE 1=1 {source_clause}
        """, source_params)
        
        results = []
        for row in cursor.fetchall():
            vec_id, chunk_id, vector_bytes, text, source, meta_str, chunk_index = row
            
            # Convert binary vector to numpy array
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(query_np, vector)
            
            # Only include results above threshold
            if similarity >= threshold:
                # Parse metadata
                metadata = json.loads(meta_str) if meta_str else {}
                
                results.append({
                    "id": chunk_id,
                    "text": text,
                    "source": source,
                    "metadata": metadata,
                    "similarity": float(similarity),
                    "chunk_index": chunk_index
                })
        
        # Sort by similarity (descending) and take top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        results = results[:top_k]
        
        # Update stats
        query_time = time.time() - start_time
        self.stats["queries"] += 1
        self.stats["total_time"] += query_time
        self.stats["last_query_time"] = query_time
        
        log.debug(f"Vector search found {len(results)} results in {query_time:.3f}s")
        return results
    
    def get_document_count(self) -> int:
        """Get the number of unique documents in the database."""
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(DISTINCT source) FROM {self.chunks_table}")
        return cursor.fetchone()[0]
    
    def get_chunk_count(self) -> int:
        """Get the total number of chunks in the database."""
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.chunks_table}")
        return cursor.fetchone()[0]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict with database statistics
        """
        # Add up-to-date counts
        stats = {
            "document_count": self.get_document_count(),
            "chunk_count": self.get_chunk_count(),
            "avg_query_time": self.stats["total_time"] / self.stats["queries"] if self.stats["queries"] > 0 else 0,
            "query_count": self.stats["queries"],
            "last_query_time": self.stats["last_query_time"]
        }
        
        return stats
    
    def _hash_text(self, text: str) -> str:
        """Generate a simple hash of text for deduplication."""
        import hashlib
        # Use first 100 chars for faster comparison
        sample = text[:100].strip().lower()
        return hashlib.md5(sample.encode('utf-8')).hexdigest()
    
    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            
    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()
